"""
CCTBX-based implementations of map coefficients FFT and model-based map
calculation.  The old qFit transformer functions have been moved to
legacy_transformers.py.
"""

from abc import ABC, abstractmethod
import logging

import numpy as np
from cctbx import maptbx, masks, miller
from cctbx.sgtbx import space_group_info
from cctbx.uctbx import unit_cell
from cctbx.xray import structure_factors
from cctbx.array_family import flex as flex_array
from cctbx.xray import ext as xray_ext
from mmtbx.utils import shift_origin

logger = logging.getLogger(__name__)


# XXX test only
def get_naive_shape(map_coeffs, nyquist):
    """
    Get map gridding parameters equivalent to the legacy transformer behavior,
    where the shape is as close as possible to strict d_min/4, regardless of
    crystal symmetry.
    """
    import qfit.xtal.legacy_transformer
    logger.debug("Running legacy QFit FFT to get naive map shape")
    xmap = qfit.xtal.legacy_transformer.fft_map_coefficients(
        map_coeffs=map_coeffs,
        nyquist=nyquist)
    shape = xmap.shape[::-1]
    logger.debug("Naive shape from qfit: %s", shape)
    crystal_gridding = maptbx.crystal_gridding(
        unit_cell             = map_coeffs.unit_cell(),
        space_group_info      = map_coeffs.space_group_info(),
        pre_determined_n_real = shape)
    logger.debug("Using pre-determined gridding: %s",
                 crystal_gridding.n_real())
    return crystal_gridding


def fft_map_coefficients(map_coeffs,
                         nyquist=2,
                         transformer="cctbx",
                         use_naive_shape=False):
    """
    Transform CCTBX map coefficients (e.g. from an MTZ file) to a real map,
    using symmetry-aware FFT gridding; alternately, use the legacy Qfit
    transformer.  Returns the equivalent 3D numpy array with the X,Z axes
    swapped.
    """
    # TODO experiment with effect of transformer behavior - how essential is
    # the old FFT gridding to reproduce previous results?
    import qfit.xtal.legacy_transformer
    if transformer == "cctbx":
        crystal_gridding = symmetry_flags = None
        if use_naive_shape:
            crystal_gridding = get_naive_shape(map_coeffs, nyquist)
        else:
            symmetry_flags = maptbx.use_space_group_symmetry
        fft_map = map_coeffs.fft_map(
            resolution_factor=1/(2*nyquist),
            crystal_gridding=crystal_gridding,
            symmetry_flags=symmetry_flags)
        real_map = fft_map.apply_sigma_scaling().real_map_unpadded()
        logger.info(f"FFT map dimensions from CCTBX: {real_map.focus()}")
        return np.swapaxes(real_map.as_numpy_array(), 0, 2)
    elif transformer == "qfit":
        return qfit.xtal.legacy_transformer.fft_map_coefficients(
            map_coeffs=map_coeffs,
            nyquist=nyquist)
    else:
        raise NotImplementedError(f"No transformer for '{transformer}'")


class _BaseTransformer(ABC):
    """
    Abstract base class for all transformer implementations.
    """
    def __init__(
        self,
        structure,
        xmap,
        rmax=3.0,
        em=False,
        smin=None,
        smax=None,
        simple=False,
    ):
        self.structure = structure
        self.xmap = xmap
        self.rmax = rmax
        self.em = em
        self.smin = smin
        self.smax = smax
        self.simple = simple

    def get_masked_selection(self):
        return self.xmap.array > 0

    # not abstract, just no-op by default
    def initialize(self):
        ...

    def reset(self, rmax=None, full=False):
        if full:
            self.xmap.array.fill(0)
        else:
            self.mask(rmax=rmax, value=0.0)

    @abstractmethod
    def density(self):
        ...

    @abstractmethod
    def get_conformers_densities(self, coor_set, b_set):
        ...

    def mask(self, rmax=None, value=1.0):
        """
        Compute an atom mask around the current structure.  Modifies
        self.xmap in place by adding values of 1 for masked points, 0 for
        points outside the mask.  Assumes that the map has previously
        been reset to 0 with self.reset(full=True).
        """
        if rmax is None:
            rmax = self.rmax
        xrs, _ = self._get_xray_structure_in_box()
        mask_sel = self._get_cctbx_mask(xrs, rmax)
        self.xmap.mask_with_value(mask_sel, value)

    def _get_xray_structure(self):
        symm = self.structure.crystal_symmetry
        if not symm:
            symm = self.xmap.get_p1_crystal_symmetry()
        xrs = self.structure.to_xray_structure(
            active_only=True,
            crystal_symmetry=symm)
        if self.em:
            #logger.debug("Switching to electron structure factor table")
            xrs.discard_scattering_type_registry()
            xrs.scattering_type_registry(table="electron")
        else:
            xrs.scattering_type_registry(table="n_gaussian")
        return xrs

    def _get_xray_structure_in_box(self):
        # XXX the logic in here is a simplified version of the approach in
        # mmtbx.utils.extract_box_around_model_and_map.  in the future it
        # would be better to use that wrapper directly in qfit, in place
        # of the calls to xmap.extract()
        xrs = self._get_xray_structure()
        if self.xmap.is_canonical_unit_cell() and self.em:
            return xrs.expand_to_p1(), 0
        # XXX note that a lot of this math is technically unnecessary if the
        # map is already a canonical P1 box, but dealing with this up front
        # allows us to modify the extracted cctbx.xray.structure object in
        # place with new coordinates and/or Bs
        origin = tuple(int(x) for x in self.xmap.grid_parameters.offset)
        uc_grid = tuple(int(x) for x in self.xmap.unit_cell_shape)
        n_real = self.xmap.n_real()
        #logger.debug(f"Computing mask with n_real={n_real} origin={origin} uc_grid={uc_grid}")
        ucp = xrs.unit_cell().parameters()
        box_cell_abc = [ucp[i]*(n_real[i]/uc_grid[i]) for i in range(3)]
        uc_box = unit_cell(box_cell_abc + list(ucp)[3:])
        #logger.debug(f"New unit cell: {uc_box.parameters()}")
        sg_p1 = space_group_info("P1")
        # XXX unlike the original qFit implementation, I am not even
        # attempting to deal with space group symmetry right now.  weirdly,
        # it's not clear if this even matters, since the old implementation
        # seems slightly buggy
        xrs_p1_box = xrs.customized_copy(space_group_info=sg_p1)
        sites_start = xrs_p1_box.sites_cart()
        # this applies the shift to the xrs_p1_box object
        soo = shift_origin(
            xray_structure=xrs_p1_box,
            n_xyz=uc_grid,
            origin_grid_units=origin)
        sites_cart = soo.xray_structure.sites_cart()
        delta_xyz_cart = sites_cart - sites_start
        sites_frac = uc_box.fractionalize(sites_cart)
        xrs_shifted = xrs_p1_box.customized_copy(unit_cell=uc_box)
        xrs_shifted.set_sites_frac(sites_frac)
        return xrs_shifted, delta_xyz_cart

    def _get_cctbx_mask(self, xrs, rmax, sites_cart=None):
        """
        Compute an atom mask using cctbx.masks.  This method assumes that
        translations to account for boxed maps have already been performed.
        """
        n_real = self.xmap.n_real()
        if sites_cart is not None:
            sites_frac = xrs.unit_cell().fractionalize(sites_cart)
        else:
            sites_frac = xrs.sites_frac()
        if isinstance(rmax, (float, int)):
            rmax = flex_array.double(sites_frac.size(), float(rmax))
        # this mask is inverted, i.e. the region of interest has value 0
        mask_sel = masks.around_atoms(
            xrs.unit_cell(),
            1,
            sites_frac,
            rmax,
            n_real,
            0,
            0).data == 0
        return mask_sel

    def get_conformers_mask(self, coor_set, rmax):
        """
        Get the combined map mask (as a numpy boolean array) for a series of
        coordinates for the current structure.
        """
        assert len(coor_set) > 0
        self.reset(full=True)
        logger.debug(f"Masking {len(coor_set)} conformations")
        xrs, dxyz = self._get_xray_structure_in_box()
        n_sites = xrs.scatterers().size()
        rmax = flex_array.double(n_sites, rmax)
        total_sel = None
        active_flag = self.structure.active
        for coor in coor_set:
            # XXX the active selection may be smaller the coordinate array!
            if coor.size != n_sites:
                coor = coor[active_flag]
            sites_cart = flex_array.vec3_double(coor.tolist()) + dxyz
            mask_sel = self._get_cctbx_mask(xrs, rmax, sites_cart)
            if total_sel is None:
                total_sel = mask_sel
            else:
                total_sel |= mask_sel
        self.xmap.mask_with_value(total_sel, 1.0)
        mask = self.xmap.array > 0
        self.reset(full=True)
        return mask


class Transformer(_BaseTransformer):

    """
    Manager to transform a structure to a density or equivalent atom mask,
    using CCTBX map sampling functions.  Modifies the xmap object in place.
    """

    def get_conformers_densities(self, coor_set, b_set):
        """
        Iterate over paired lists of candidate conformation coordinates and
        B-factors and compute the density for each, without modifying the
        internal data structures.
        """
        xrs, dxyz = self._get_xray_structure_in_box()
        active_flag = self.structure.active
        for (coor, b) in zip(coor_set, b_set):
            if coor.size != xrs.scatterers().size():
                coor = coor[active_flag]
                b = b[active_flag]
            sites_cart = flex_array.vec3_double(coor.tolist())
            xrs.set_sites_cart(sites_cart + dxyz)
            xrs.set_b_iso(values=flex_array.double(b))
            # FIXME workaround to match behavior in volume.py
            yield np.swapaxes(self.get_density(xrs).as_numpy_array(), 0, 2)
            #yield self.density(xrs)
            #self.xmap.tofile("density_new.ccp4")

    def get_density(self, xrs=None):
        """
        Compute the current model electron density using cctbx.xray map
        sampling function, without any FFTs, and return the density grid
        as a 3D numpy array.  Does not modify the internal data structures.
        """
        if not xrs:
            xrs, _ = self._get_xray_structure_in_box()
        n_real = self.xmap.n_real()
        u_base = xray_ext.calc_u_base(
            d_min=self.xmap.resolution.high,
            grid_resolution_factor=0.25)
        # XXX this is an internal method used by the CCTBX map coefficient
        # function, which recycles the map through an FFT (very computationally
        # expensive).
        sampled_density = xray_ext.sampled_model_density(
            unit_cell=xrs.unit_cell(),
            scatterers=xrs.scatterers(),
            scattering_type_registry=xrs.scattering_type_registry(),
            fft_n_real=n_real,
            fft_m_real=n_real,
            u_base=u_base,
            wing_cutoff=1e-3,
            exp_table_one_over_step_size=-100,
            force_complex=False,
            use_u_base_as_u_extra=True,
            sampled_density_must_be_positive=False,
            tolerance_positive_definite=1e-5)
        return sampled_density.real_map_unpadded()

    def density(self, xrs=None):
        """Compute the electron density and update the internal map object"""
        map_array = self.get_density(xrs)
        self.xmap.set_values_from_flex_array(map_array)
        return self.xmap.array


class FFTTransformer(Transformer):
    """
    Alternative transformer for cases where we want to use the same set of
    reflections (h,k,l) as the input map coefficients, and can tolerate the
    overhead of running two FFTs.  Currently this is only used in scaler.py.
    """

    def __init__(self, structure, xmap, hkl=None, em=False, **kwds):
        super().__init__(structure, xmap, em=em)
        if hkl is None:
            hkl = self.xmap.hkl
        assert hkl is not None
        self.hkl = hkl

    def density(self):
        """
        Compute the electron density using via CCTBX structure factor FFT.
        """
        xrs = self.structure.to_xray_structure(active_only=True)
        if self.em:
            logger.debug("Switching to electron structure factor table")
            xrs.discard_scattering_type_registry()
            xrs.scattering_type_registry(table="electron")
        assert self.structure.crystal_symmetry is not None
        reflections = miller.set(xrs.crystal_symmetry(),
                                 flex_array.miller_index(self.hkl),
                                 anomalous_flag=False)
        # XXX This is currently in closer agreement with the old "legacy"
        # implementation, which does not use FFT, but it is much slower
        sfs = structure_factors.from_scatterers(
            crystal_symmetry=xrs.crystal_symmetry(),
            d_min=reflections.d_min(),
            cos_sin_table=False,
            quality_factor=None,
            u_base=None,
            b_base=None,
            wing_cutoff=None)(
                xray_structure=xrs,
                miller_set=reflections,
                algorithm="fft")
        grid = maptbx.crystal_gridding(
            unit_cell = xrs.unit_cell(),
            space_group_info = xrs.space_group_info(),
            pre_determined_n_real = self.xmap.n_real())
        fcalc_map = miller.fft_map(
            crystal_gridding = grid,
            fourier_coefficients = sfs.f_calc())
        # FIXME This does not end up on the desired scale
        real_map = fcalc_map.apply_volume_scaling().real_map_unpadded()
        negative_sel = real_map < 0
        real_map.set_selected(negative_sel, 0)
        self.xmap.set_values_from_flex_array(real_map)
        return self.xmap.array


def get_transformer(impl_name="cctbx", *args, **kwds):
    """
    Instantiate a Transformer class using the specified implementation.
    """
    if impl_name == "qfit":
        from qfit.xtal.legacy_transformer import Transformer as QfitTransformer
        return QfitTransformer(*args, **kwds)
    elif impl_name == "cctbx":
        return Transformer(*args, **kwds)
    else:
        raise NotImplementedError(f"No transformer for '{impl_name}'")


def get_fft_transformer(impl_name="cctbx", *args, **kwds):
    if impl_name == "qfit":
        from qfit.xtal.legacy_transformer import FFTTransformer as QfitFFTTransformer
        return QfitFFTTransformer(*args, **kwds)
    elif impl_name == "cctbx":
        return FFTTransformer(*args, **kwds)
    else:
        raise NotImplementedError(f"No transformer for '{impl_name}'")
