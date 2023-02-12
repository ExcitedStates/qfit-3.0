import os.path
from copy import copy
from itertools import product
from numbers import Real
from struct import unpack as _unpack, pack as _pack
from sys import byteorder as _BYTEORDER
import logging
import time

import numpy as np
from scipy.ndimage import map_coordinates
from iotbx.reflection_file_reader import any_reflection_file
import iotbx.ccp4_map
import iotbx.mrcfile
from cctbx import sgtbx
from scitbx.array_family import flex

from .spacegroups import SpaceGroup
from .unitcell import UnitCell
from ._extensions import extend_to_p1


logger = logging.getLogger(__name__)


class GridParameters:
    def __init__(self, voxelspacing=(1, 1, 1), offset=(0, 0, 0)):
        if isinstance(voxelspacing, Real):
            voxelspacing = [voxelspacing] * 3
        self.voxelspacing = np.asarray(voxelspacing, np.float64)
        self.offset = np.asarray(offset, np.int32)

    def copy(self):
        return GridParameters(self.voxelspacing.copy(), self.offset.copy())

    def __str__(self):
        return f"voxelspacing={list(self.voxelspacing)} grid={list(self.offset)}"


class Resolution:
    def __init__(self, high=None, low=None):
        self.high = high
        self.low = low

    def copy(self):
        return Resolution(self.high, self.low)


class _BaseVolume:
    def __init__(self, array, grid_parameters=None, origin=(0, 0, 0)):
        self.array = array
        if grid_parameters is None:
            grid_parameters = GridParameters()
        self.grid_parameters = grid_parameters
        self.origin = np.asarray(origin, np.float64)

    @property
    def shape(self):
        return self.array.shape

    @property
    def offset(self):
        return self.grid_parameters.offset

    @property
    def voxelspacing(self):
        return self.grid_parameters.voxelspacing

    def tofile(self, file_name, fmt=None):
        if fmt is None:
            fmt = os.path.splitext(file_name)[-1][1:]
        if fmt in ("ccp4", "map", "mrc"):
            self.write_map_file(file_name)
        else:
            raise ValueError(f"Format '{fmt}' is not supported.")

    def as_cctbx_map(self):
        """
        Convert the density to a CCTBX map-like flex array.
        """
        density_np = np.ascontiguousarray(np.swapaxes(self.array, 0, 2))
        density = flex.double(density_np)
        (ox, oy, oz) = (int(x) for x in self.grid_parameters.offset)
        (nx, ny, nz) = density.accessor().focus()
        grid = flex.grid((ox, oy, oz), (nx + ox, ny + oy, nz + oz))
        density.reshape(grid)
        return density

    def write_map_file(self, file_name):
        density = self.as_cctbx_map()
        if file_name.endswith(".ccp4"):
            writer = iotbx.ccp4_map.write_ccp4_map
        else:
            writer = iotbx.mrcfile.write_ccp4_map
        writer(
            file_name=file_name,
            unit_cell=self.unit_cell.to_cctbx(),
            space_group=self.unit_cell.space_group.as_cctbx_group(),
            unit_cell_grid=[int(x) for x in self.unit_cell_shape],
            map_data=density,
            labels=flex.std_string(["qfit"]),
        )


# XXX currently unused
class EMMap(_BaseVolume):

    """A non-periodic volume. Has no notion of a unit cell or space group."""

    def __init__(self, array, grid_parameters=None, origin=(0, 0, 0)):
        super().__init__(array, grid_parameters, origin)

    # FIXME replace with ccp4 input
    # @classmethod
    # def fromfile(cls, fid, fmt=None):
    #    p = parse_volume(fid)
    #    density = p.density
    #    grid_parameters = GridParameters(p.voxelspacing)
    #    origin = p.origin
    #    return cls(density, grid_parameters=grid_parameters, origin=origin)

    @classmethod
    def zeros(cls, shape, grid_parameters=None, origin=None):
        array = np.zeros(shape, dtype=np.float64)
        return cls(array, grid_parameters, origin)

    @classmethod
    def zeros_like(cls, volume):
        array = np.zeros_like(volume.array)
        return cls(array, volume.grid_parameters, volume.origin)

    def copy(self):
        return EMMap(
            self.array.copy(),
            grid_parameters=self.grid_parameters.copy(),
            origin=self.origin.copy(),
        )

    def interpolate(self, xyz, order=1):
        # Transform xyz to grid coor.
        grid_coor = xyz - self.origin
        grid_coor /= self.grid_parameters.voxelspacing
        values = map_coordinates(self.array, grid_coor.T[::-1], order=order)
        return values

    def extract(self, xyz, padding=3):
        grid_coor = xyz - self.origin
        grid_coor /= self.voxelspacing
        grid_padding = padding / self.voxelspacing
        lb = grid_coor.min(axis=0) - grid_padding
        ru = grid_coor.max(axis=0) + grid_padding
        lb = np.floor(lb).astype(int)
        lb = np.maximum(lb, 0)
        ru = np.ceil(ru).astype(int)
        array = self.array[lb[2] : ru[2], lb[1] : ru[1], lb[0] : ru[0]].copy()
        grid_parameters = GridParameters(self.voxelspacing)
        origin = self.origin + lb * self.voxelspacing
        return EMMap(array, grid_parameters=grid_parameters, origin=origin)


class XMap(_BaseVolume):

    """A periodic volume with a unit cell and space group."""

    def __init__(
        self,
        array,
        grid_parameters=None,
        unit_cell=None,
        resolution=None,
        hkl=None,
        origin=None,
    ):
        super().__init__(array, grid_parameters)
        self.unit_cell = unit_cell
        self.hkl = hkl
        self.resolution = resolution
        self.cutoff_dict = {}
        if origin is None:
            self.origin = np.zeros(3, np.float64)
        else:
            self.origin = np.asarray(origin)

    @staticmethod
    def fromfile(fname, fmt=None, resolution=None, label="FWT,PHWT"):
        if fmt is None:
            fmt = os.path.splitext(fname)[1]
        if fmt in (".ccp4", ".mrc", ".map"):
            return XMap.from_mapfile(fname, resolution)
        elif fmt == ".mtz":
            return XMap.from_mtz(fname, resolution, label)
        else:
            raise RuntimeError("File format not recognized.")

    @staticmethod
    def from_mapfile(fname, resolution):
        if resolution is None:
            raise ValueError(
                f"{fname} is a CCP4/MRC/MAP file. Please provide a resolution (use the '-r'/'--resolution' flag)."
            )
        if fname.endswith(".ccp4"):
            map_io = iotbx.ccp4_map.map_reader(fname)
        else:
            map_io = iotbx.mrcfile.map_reader(fname)
        uc_params = map_io.unit_cell().parameters()
        origin = (0, 0, 0)  # map_io.nxstart_nystart_nzstart
        unit_cell = UnitCell(*uc_params, map_io.space_group_number)
        abc = uc_params[0:3]
        shape = map_io.unit_cell_grid
        # Reorder axis so that nx is fastest changing.
        # NOTE CCTBX handles axis conventions internally, so we always run
        # this swap here (without needing to check the map file header again)
        density = np.swapaxes(map_io.data.as_numpy_array(), 0, 2)
        cell_shape = tuple(reversed(shape))
        voxelspacing = tuple(length / n for length, n in zip(abc, shape))
        offset = map_io.data.accessor().origin()
        grid_parameters = GridParameters(voxelspacing, offset)
        resolution = Resolution(high=resolution)
        return XMap(
            density,
            grid_parameters,
            unit_cell=unit_cell,
            resolution=resolution,
            origin=origin,
        )

    @staticmethod
    def from_mtz(fname, resolution=None, label="FWT,PHWT"):
        from .transformer import SFTransformer

        mtz_in = any_reflection_file(fname)
        miller_arrays = {a.info().label_string(): a for a in mtz_in.as_miller_arrays()}
        map_coeffs = miller_arrays.get(label, None)
        if not map_coeffs:
            raise KeyError(f"Could not find columns '{label}' in MTZ file.")
        hkl = np.asarray(list(map_coeffs.indices()), np.int32)
        unit_cell = UnitCell(*map_coeffs.unit_cell().parameters())
        space_group = SpaceGroup.from_cctbx(map_coeffs.space_group_info())
        unit_cell.space_group = space_group
        f = map_coeffs.amplitudes().data().as_numpy_array().copy()
        phi = map_coeffs.phases().data().as_numpy_array().copy() * 180 / np.pi
        fft = SFTransformer(hkl, f, phi, unit_cell)
        grid = fft()
        abc = [getattr(unit_cell, x) for x in "a b c".split()]
        voxelspacing = [x / n for x, n in zip(abc, grid.shape[::-1])]
        logger.debug(f"MTZ unit cell: {unit_cell}")
        grid_parameters = GridParameters(voxelspacing)
        resolution = Resolution(high=map_coeffs.d_min(), low=map_coeffs.d_max_min()[0])
        logger.debug(f"MTZ Resolution: {map_coeffs.d_min()}")
        logger.debug(f"Map size: {grid.size}")
        logger.debug(f"Map Grid: {grid_parameters}")
        return XMap(
            grid, grid_parameters, unit_cell=unit_cell, resolution=resolution, hkl=hkl
        )

    @classmethod
    def zeros_like(cls, xmap):
        array = np.zeros_like(xmap.array)
        try:
            uc = xmap.unit_cell.copy()
        except AttributeError:
            uc = None
        hkl = copy(xmap.hkl)
        return cls(
            array,
            grid_parameters=xmap.grid_parameters.copy(),
            unit_cell=uc,
            hkl=hkl,
            resolution=xmap.resolution.copy(),
            origin=xmap.origin.copy(),
        )

    def asymmetric_unit_cell(self):
        raise NotImplementedError

    @property
    def unit_cell_shape(self):
        shape = np.round(self.unit_cell.abc / self.grid_parameters.voxelspacing).astype(
            int
        )
        return shape

    def canonical_unit_cell(self):
        """
        Perform space group symmetry expansion to fill in density values for
        the entire unit cell.
        """
        if self.is_canonical_unit_cell():
            return self
        shape = np.round(self.unit_cell.abc / self.grid_parameters.voxelspacing).astype(
            int
        )[::-1]
        array = np.zeros(shape, np.float64)
        grid_parameters = GridParameters(self.voxelspacing)
        out = XMap(
            array,
            grid_parameters=grid_parameters,
            unit_cell=self.unit_cell,
            hkl=self.hkl,
            resolution=self.resolution,
        )
        offset = np.asarray(self.offset, np.int32)
        for symop in self.unit_cell.space_group.symop_list:
            transform = np.hstack((symop.R, symop.t.reshape(3, -1)))
            transform[:, -1] *= out.shape[::-1]
            extend_to_p1(self.array, offset, transform, out.array)
        return out

    def is_canonical_unit_cell(self):
        return np.allclose(self.shape, self.unit_cell_shape[::-1]) and np.allclose(
            self.offset, 0
        )

    def extract(self, orth_coor, padding=3.0):
        """Create a copy of the map around the atomic coordinates provided.

        Args:
            orth_coor (np.ndarray[(n_atoms, 3), dtype=np.float]):
                a collection of Cartesian atomic coordinates
            padding (float): amount of padding (in Angstrom) to add around the
                returned electron density map
        Returns:
            XMap: the new map object around the coordinates
        """
        if not self.is_canonical_unit_cell():
            raise RuntimeError("XMap should contain full unit cell.")
        logger.debug(f"Extracting map around {len(orth_coor)} atoms")

        # Convert atomic Cartesian coordinates to voxelgrid coordinates
        grid_coor = orth_coor @ self.unit_cell.orth_to_frac.T
        grid_coor *= self.unit_cell_shape
        grid_coor -= self.offset

        # How many voxels are we padding by?
        grid_padding = padding / self.voxelspacing

        # What are the voxel-coords of the lower and upper extrema that we will extract?
        lb = grid_coor.min(axis=0) - grid_padding
        ru = grid_coor.max(axis=0) + grid_padding
        lb = np.floor(lb).astype(int)
        ru = np.ceil(ru).astype(int)
        shape = (ru - lb)[::-1]
        logger.debug(f"From old map size (voxels): {self.shape}")
        logger.debug(f"Extract between corners:    {lb[::-1]}, {ru[::-1]}")
        logger.debug(f"New map size (voxels):      {shape}")

        # Make new GridParameters, make sure to update offset
        grid_parameters = GridParameters(self.voxelspacing, self.offset + lb)
        offset = grid_parameters.offset

        # Use index math to get appropriate region of map
        #     - Create a tuple of axis-indexes
        #     - Perform wrapping maths on the indexes
        #     - Apply new index to the original map to get re-mapped map
        # This is ~500--1000x faster than working element-by-element
        # (BTR: I don't understand why we're indexing jki and not ijk)
        ranges = [range(axis_len) for axis_len in shape]
        ixgrid = np.ix_(*ranges)
        ixgrid = tuple(
            (dimension_index + offset) % wrap_to
            for dimension_index, offset, wrap_to in zip(
                ixgrid, offset[::-1], self.unit_cell_shape[::-1]
            )
        )
        density_map = self.array[ixgrid]

        logger.debug(f"Extracted map size: {density_map.size}")
        logger.debug(f"Extracted map grid: {grid_parameters}")
        return XMap(
            density_map,
            grid_parameters=grid_parameters,
            unit_cell=self.unit_cell,
            resolution=self.resolution,
            hkl=self.hkl,
            origin=self.origin,
        )

    def interpolate(self, xyz):
        # Transform xyz to grid coor.
        uc = self.unit_cell
        orth_to_grid = uc.orth_to_frac * self.unit_cell_shape.reshape(3, 1)
        if not np.allclose(self.origin, 0):
            xyz = xyz - self.origin
        grid_coor = orth_to_grid @ xyz.T
        grid_coor -= self.offset.reshape(3, 1)
        if self.is_canonical_unit_cell():
            grid_coor %= self.unit_cell_shape.reshape(3, 1)
        values = map_coordinates(self.array, grid_coor[::-1], order=1)
        return values

    def set_space_group(self, space_group):
        self.unit_cell.set_space_group(space_group)


class ASU:

    """Assymetric Unit Cell"""

    def __init__(
        self, array, grid_parameters=None, unit_cell=None, resolution=None, hkl=None
    ):
        raise NotImplementedError
        super().__init__(array, grid_parameters, unit_cell, resolution, hkl)
