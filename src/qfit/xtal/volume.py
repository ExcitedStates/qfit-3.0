from abc import ABC, abstractmethod
import os.path
from copy import copy, deepcopy
from numbers import Real
import logging

import numpy as np
from scipy.ndimage import map_coordinates
from iotbx.reflection_file_reader import any_reflection_file
import iotbx.ccp4_map
import iotbx.mrcfile
from cctbx import maptbx, crystal
from scitbx.array_family import flex
import boost_adaptbx.boost.python as bp
asu_map_ext = bp.import_ext("cctbx_asymmetric_map_ext")

from qfit.xtal.unitcell import UnitCell
from qfit.xtal.spacegroups import SpaceGroup
from qfit.xtal.transformer import fft_map_coefficients
from qfit._extensions import extend_to_p1  # pylint: disable=import-error,no-name-in-module


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


class _BaseVolume(ABC):
    def __init__(self, array, grid_parameters=None, origin=(0, 0, 0)):
        self.array = array
        if grid_parameters is None:
            grid_parameters = GridParameters()
        self.grid_parameters = grid_parameters
        self.origin = np.asarray(origin, np.float64)
        self._n_real = tuple(int(x) for x in self.array.shape[::-1])

    @abstractmethod
    def write_map_file(self, file_name):
        ...

    @property
    def shape(self):
        return self.array.shape

    def n_real(self):
        """Alternative to 'shape' for CCTBX compatibility"""
        return self._n_real

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

    def value_at(self, i, j, k):
        """
        Returns the value of the real-space map at coordinates (i,j,k),
        accounting for internal storage conventions
        """
        return self.array[k][j][i]

    def set_values_from_flex_array(self, real_map):
        self.array[:] = np.swapaxes(real_map.as_numpy_array(), 0, 2)

    def mask_with_value(self, real_sel, value):
        real_sel_np = np.swapaxes(real_sel.as_numpy_array(), 0, 2)
        self.array[real_sel_np] += value


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
        source_transformer=None,
    ):
        super().__init__(array, grid_parameters)
        self.unit_cell = unit_cell
        self.hkl = hkl
        self.resolution = resolution
        self.cutoff_dict = {}
        self._source_transformer = source_transformer
        if origin is None:
            self.origin = np.zeros(3, np.float64)
        else:
            self.origin = np.asarray(origin)

    @staticmethod
    def fromfile(fname, fmt=None, resolution=None, label="FWT,PHWT",
                 transformer="cctbx"):
        if fmt is None:
            fmt = os.path.splitext(fname)[1]
        if fmt in (".ccp4", ".mrc", ".map"):
            return XMap.from_mapfile(fname, resolution)
        elif fmt == ".mtz":
            return XMap.from_mtz(fname,
                                 resolution=resolution,
                                 label=label,
                                 transformer=transformer)
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
        shape = map_io.unit_cell_grid
        return XMap.from_cctbx_map(map_io.data,
                                   shape,
                                   unit_cell,
                                   resolution,
                                   origin)

    @staticmethod
    def from_cctbx_map(map_data, shape, unit_cell, resolution, origin):
        abc = unit_cell.abc
        voxelspacing = tuple(length / n for length, n in zip(abc, shape))
        offset = map_data.accessor().origin()
        grid_parameters = GridParameters(voxelspacing, offset)
        resolution = Resolution(high=resolution)
        density = np.swapaxes(map_data.as_numpy_array(), 0, 2)
        return XMap(
            density,
            grid_parameters,
            unit_cell=unit_cell,
            resolution=resolution,
            origin=origin,
        )

    @staticmethod
    def from_mtz(fname,
                 resolution=None,
                 label="FWT,PHWT",
                 resolution_factor=1/4,
                 transformer="cctbx"):
        mtz_in = any_reflection_file(fname)
        miller_arrays = {a.info().label_string(): a for a in mtz_in.as_miller_arrays()}
        map_coeffs = miller_arrays.get(label, None)
        if not map_coeffs:
            raise KeyError(f"Could not find columns '{label}' in MTZ file.")
        unit_cell = UnitCell(*map_coeffs.unit_cell().parameters())
        space_group = SpaceGroup.from_cctbx(map_coeffs.space_group_info())
        unit_cell.space_group = space_group
        grid = fft_map_coefficients(map_coeffs,
                                    nyquist=1/(2*resolution_factor),
                                    transformer=transformer)
        abc = unit_cell.abc
        voxelspacing = [x / n for x, n in zip(abc, grid.shape[::-1])]
        logger.debug(f"MTZ unit cell: {unit_cell}")
        grid_parameters = GridParameters(voxelspacing)
        resolution = Resolution(high=map_coeffs.d_min(),
                                low=map_coeffs.d_max_min()[0])
        logger.debug(f"MTZ Resolution: {map_coeffs.d_min()}")
        logger.debug(f"Map size: {grid.size}")
        logger.debug(f"Map Grid: {grid_parameters}")
        hkl = np.asarray(list(map_coeffs.indices()), np.int32)
        return XMap(
            grid,
            grid_parameters,
            unit_cell=unit_cell,
            resolution=resolution,
            hkl=hkl,
            source_transformer=transformer
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

    def get_p1_crystal_symmetry(self):
        return crystal.symmetry(unit_cell=self.unit_cell.to_cctbx(),
                                space_group_symbol="P1")

    def _expand_to_p1(self):
        """
        Use CCTBX's asymmetric map handling to expand to P1.
        """
        cctbx_map = self.as_cctbx_map()
        asu_map = asu_map_ext.asymmetric_map(
            self.unit_cell.space_group.as_cctbx_group().type(),
            cctbx_map,
            [int(x) for x in self.unit_cell_shape])
        expanded = asu_map.symmetry_expanded_map()
        maptbx.unpad_in_place(expanded)
        shape = expanded.focus()
        logger.info("Expanded map with shape %s to %s", tuple(self.shape),
                    tuple(shape))
        return XMap.from_cctbx_map(
            map_data=expanded,
            shape=shape,
            unit_cell=self.unit_cell,
            resolution=self.resolution,
            origin=self.origin)

    def _expand_to_p1_non_symmetric(self):
        """
        Old qFit implementation of symmetry expansion - this is still run
        by default when the old transformer is selected, whether or not the
        map fills a P1 box already.
        """
        shape = self.unit_cell.abc / self.grid_parameters.voxelspacing
        shape = np.round(shape).astype(int)[::-1]
        logger.info("Expanding map with shape %s to %s", tuple(self.shape),
                    tuple(shape))
        array = np.zeros(shape, np.float64)
        grid_parameters = GridParameters(self.voxelspacing)
        out = XMap(
            array,
            grid_parameters=grid_parameters,
            unit_cell=self.unit_cell,
            hkl=self.hkl,
            resolution=self.resolution,
            source_transformer=self._source_transformer
        )
        offset = np.asarray(self.offset, np.int32)
        for symop in self.unit_cell.space_group.symop_list:
            transform = np.hstack((symop.R, symop.t.reshape(3, -1)))
            transform[:, -1] *= out.shape[::-1]
            logger.debug("Transform for symop %s is %s", symop, transform)
            extend_to_p1(self.array, offset, transform, out.array)
        return out

    def canonical_unit_cell(self, expand_to_p1=None):
        """
        Perform space group symmetry expansion to fill in density values for
        the entire unit cell.
        """
        # TODO experiment with the effect of running the expansion; to
        # completely reproduce the original behavior of qfit-3.0 we
        # need to also run extend_to_p1 even if the map is already complete
        if self.is_canonical_unit_cell() and expand_to_p1 != True:
            logger.info("Skipping map extension")
            return self
        else:
            return self._expand_to_p1_non_symmetric()
            # CCTBX version
            #return self._expand_to_p1()

    def is_canonical_unit_cell(self):
        return (np.allclose(self.shape, self.unit_cell_shape[::-1]) and
                np.allclose(self.offset, 0))

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

    def get_unit_cell_grid_selection(self):
        """
        Generate the selection to re-extract a region from a P1 box, covering
        the same indices as the current map region.
        """
        shape = self.array.shape
        offset = self.grid_parameters.offset
        ranges = [range(axis_len) for axis_len in shape]
        ixgrid = np.ix_(*ranges)
        return tuple(
            (dimension_index + offset) % wrap_to
            for dimension_index, offset, wrap_to in zip(
                ixgrid, offset[::-1], self.unit_cell_shape[::-1]
            )
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

    def save_mask(self, mask, file_name):
        """
        (Internal debugging method) Save an atom mask corresponding to the
        current map as a map file with all masked points set to 1.
        """
        tmp_map = deepcopy(self)
        tmp_map.array[:] = 0.0
        tmp_map.array[mask] = 1.0
        logger.debug("Saving atom mask to %s", os.path.abspath(file_name))
        tmp_map.write_map_file(file_name)

    def save_masked_map(self, mask, file_name, map_data=None):
        """
        (Internal debugging method) Save the masked region of the current
        map (or an equivalently sized array).
        """
        tmp_map = deepcopy(self)
        if map_data is not None:
            tmp_map.array[:] = map_data
        tmp_map.array[~mask] = 0.0
        logger.debug("Saving masked map to %s", os.path.abspath(file_name))
        tmp_map.write_map_file(file_name)
