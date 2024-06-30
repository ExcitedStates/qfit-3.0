import logging

from cctbx import maptbx, masks, miller
from cctbx.sgtbx import space_group_info
from cctbx.uctbx import unit_cell
from cctbx.xray import structure_factors
from cctbx.array_family import flex as flex_array
from cctbx.xray import ext as xray_ext
from mmtbx.utils import shift_origin

logger = logging.getLogger(__name__)


def fft_map_coefficients(map_coeffs, nyquist=2):
    """
    Transform CCTBX map coefficients (e.g. from an MTZ file) to a real map,
    using symmetry-aware FFT gridding.  Returns the equivalent 3D numpy array.
    """
    fft_map = map_coeffs.fft_map(
        resolution_factor=1/(2*nyquist),
        symmetry_flags=maptbx.use_space_group_symmetry)
    real_map = fft_map.apply_sigma_scaling().real_map_unpadded()
    logger.info(f"FFT map dimensions from CCTBX: {real_map.focus()}")
    return real_map.as_numpy_array()


class Transformer:

    """
    Manager to transform a structure to a density or equivalent atom mask,
    using CCTBX map sampling functions.  Modifies the xmap object in place.
    """
    def __init__(
        self,
        structure,
        xmap,
        rmax=3.0,
        em=False,
    ):
        self.structure = structure
        self.xmap = xmap
        self.rmax = rmax
        self.em = em

    def reset(self, rmax=None, full=False):
        if full:
            self.xmap.array.fill(0)
        else:
            self.mask(rmax=rmax, value=0.0)

    def get_masked_selection(self):
        return self.xmap.array > 0

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
        return self._mask_cctbx(xrs, rmax, value)

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

    def _mask_cctbx(self, xrs, rmax, value=1.0):
        """
        Compute an atom mask using cctbx.masks.  This method accounts for
        map cutouts and origin shifts to match the original qFit behavior,
        by temporarily translating the masked structure to fit in a P1 box
        corresponding to the map extents.
        """
        n_real = self.xmap.n_real()
        sites_frac = xrs.sites_frac()
        # this mask is inverted, i.e. the region of interest has value 0
        mask_sel = masks.around_atoms(
            xrs.unit_cell(),
            1,
            sites_frac,
            flex_array.double(sites_frac.size(), rmax),
            n_real,
            0,
            0).data == 0
        self.xmap.mask_with_value(mask_sel, value)

    def get_conformers_mask(self, coor_set, rmax):
        """
        Get the combined map mask (as a numpy boolean array) for a series of
        coordinates for the current structure.
        """
        assert len(coor_set) > 0
        self.reset(full=True)
        logger.debug(f"Masking {len(coor_set)} conformations")
        xrs, dxyz = self._get_xray_structure_in_box()
        for coor in coor_set:
            # XXX the active selection may be smaller the coordinate array!
            if coor.size != xrs.scatterers().size():
                coor = coor[self.structure.active]
            xrs.set_sites_cart(flex_array.vec3_double((coor + dxyz).tolist()))
            self._mask_cctbx(xrs, rmax)
        mask = self.xmap.array > 0
        self.reset(full=True)
        return mask

    def get_conformers_densities(self, coor_set, b_set):
        """
        Iterate over paired lists of candidate conformation coordinates and
        B-factors and compute the density for each, without duplicating
        memory or array operations.
        """
        xrs, dxyz = self._get_xray_structure_in_box()
        for (coor, b) in zip(coor_set, b_set):
            if coor.size != xrs.scatterers().size():
                coor = coor[self.structure.active]
                b = b[self.structure.active]
            xrs.set_sites_cart(flex_array.vec3_double((coor + dxyz).tolist()))
            xrs.set_b_iso(values=flex_array.double(b))
            yield self.density(xrs)
            self.reset(full=True)

    # TODO figure out why this produces inferior results versus FFT
    def density(self, xrs=None):
        """
        Compute the current model electron density using cctbx.xray map
        sampling function, without any FFTs
        """
        if not xrs:
            xrs, _ = self._get_xray_structure_in_box()
        n_real = self.xmap.n_real()
        u_base = xray_ext.calc_u_base(
            d_min=self.xmap.resolution.high,
            grid_resolution_factor=0.25)
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
        real_map = sampled_density.real_map_unpadded()
        self.xmap.set_values_from_flex_array(real_map)
        return self.xmap.array


# XXX this is much slower, but it's useful as a reference implementation
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

    def density(self, xrs=None):
        """
        Compute the electron density using via CCTBX structure factor FFT.
        """
        assert self.structure.crystal_symmetry is not None
        if not xrs:
            xrs = self._get_xray_structure()
        reflections = miller.set(xrs.crystal_symmetry(),
                                 flex_array.miller_index(self.hkl),
                                 anomalous_flag=False)
        # XXX This is currently in closer agreement with the previous
        # implementation, which did not use FFT, but it is much slower
        sfs = structure_factors.from_scatterers(
            crystal_symmetry=xrs.crystal_symmetry(),
            d_min=self.xmap.resolution.high,
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
        real_map = fcalc_map.apply_volume_scaling().real_map_unpadded()
        self.xmap.set_values_from_flex_array(real_map)
        return self.xmap.array


def get_transformer(impl_name="cctbx", *args, **kwds):
    """
    Instantiate a Transformer class using the specified implementation.
    """
    if impl_name == "fft":
        return FFTTransformer(*args, **kwds)
    else:
        return Transformer(*args, **kwds)
