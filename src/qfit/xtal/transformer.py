import logging

import numpy as np
from scipy.integrate import fixed_quad
from cctbx import maptbx, masks, miller
from cctbx.sgtbx import space_group_info
from cctbx.uctbx import unit_cell
from cctbx.xray import structure_factors
from cctbx.array_family import flex as flex_array
from cctbx.xray import ext as xray_ext
from mmtbx.utils import shift_origin

from qfit.xtal.atomsf import ATOM_STRUCTURE_FACTORS, ELECTRON_SCATTERING_FACTORS
from qfit._extensions import dilate_points  # pylint: disable=import-error,no-name-in-module

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


class CCTBXTransformer:

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
        smin=None,     # XXX unused, for API compatibility
        smax=None,     # XXX unused, for API compatibility
        simple=False,  # XXX unused, for API compatibility
    ):
        self.structure = structure
        self.xmap = xmap
        self.rmax = rmax
        self.em = em

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
        return self._mask_cctbx(rmax, value)

    def _get_structure_in_box(self):
        # XXX the logic in here is a simplified version of the approach in
        # mmtbx.utils.extract_box_around_model_and_map.  in the future it
        # would be better to use that wrapper directly in qfit, in place
        # of the calls to xmap.extract()
        xrs = self.structure.to_xray_structure(active_only=True)
        if self.xmap.is_canonical_unit_cell():
            return xrs.expand_to_p1()
        origin = tuple(int(x) for x in self.xmap.grid_parameters.offset)
        uc_grid = tuple(int(x) for x in self.xmap.unit_cell_shape)
        n_real = self.xmap.n_real()
        #logger.debug(f"Computing mask with n_real={n_real} origin={origin} uc_grid={uc_grid}")
        ucp = self.structure.crystal_symmetry.unit_cell().parameters()
        box_cell_abc = [ucp[i]*(n_real[i]/uc_grid[i]) for i in range(3)]
        uc_box = unit_cell(box_cell_abc + list(ucp)[3:])
        #logger.debug(f"New unit cell: {uc_box.parameters()}")
        sg_p1 = space_group_info("P1")
        # XXX unlike the original qFit implementation, I am not even
        # attempting to deal with space group symmetry right now.  weirdly,
        # it's not clear if this even matters, since the old implementation
        # seems slightly buggy
        xrs_p1_box = xrs.customized_copy(space_group_info=sg_p1)
        # this applies the shift to the xrs_p1_box object
        soo = shift_origin(
            xray_structure=xrs_p1_box,
            n_xyz=uc_grid,
            origin_grid_units=origin)
        sites_cart = soo.xray_structure.sites_cart()
        sites_frac = uc_box.fractionalize(sites_cart)
        xrs_shifted = xrs_p1_box.customized_copy(unit_cell=uc_box)
        xrs_shifted.set_sites_frac(sites_frac)
        return xrs_shifted

    def _mask_cctbx(self, rmax, value):
        """
        Compute an atom mask using cctbx.masks.  This method accounts for
        map cutouts and origin shifts to match the original qFit behavior,
        by temporarily translating the masked structure to fit in a P1 box
        corresponding to the map extents.
        """
        xrs = self._get_structure_in_box()
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
        for coor in coor_set:
            self.structure.coor = coor
            self.mask(rmax)
        mask = self.xmap.array > 0
        self.reset(full=True)
        return mask

    def get_conformer_density(self, coor, b):
        self.structure.coor = coor
        self.structure.b = b
        self.density()
        return self.xmap.array

    # XXX unused, for API compatibility
    def initialize(self):
        ...

    def reset(self, rmax=None, full=False):
        if full:
            self.xmap.array.fill(0)
        else:
            self.mask(rmax=rmax, value=0.0)

    # TODO figure out why this produces inferior results
    def density(self):
        """
        Compute the current model electron density using cctbx.xray map
        sampling function, without any FFTs
        """
        xrs = self._get_structure_in_box()
        if self.em:
            #logger.debug("Switching to electron structure factor table")
            xrs.discard_scattering_type_registry()
            xrs.scattering_type_registry(table="electron")
        else:
            xrs.scattering_type_registry(table="n_gaussian")
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


class FFTTransformer(CCTBXTransformer):
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
        # XXX This is currently in closer agreement with the old "classic"
        # implementation below, which does not use FFT, but it is much slower
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


# TODO retire this
class QfitTransformer(CCTBXTransformer):

    """Transform a structure to a density."""

    def __init__(
        self,
        structure,
        xmap,
        smin=None,
        smax=None,
        rmax=3.0,
        rstep=0.01,
        simple=False,
        em=False,
    ):
        super().__init__(structure, xmap, rmax=rmax, em=em)
        self.smin = smin
        self.smax = smax
        self.rstep = rstep
        self.simple = simple
        self.asf_range = 6
        if self.em == True:
            self.asf_range = 5
            self._asf = ELECTRON_SCATTERING_FACTORS
        else:
            self._asf = ATOM_STRUCTURE_FACTORS

        self._initialized = False

        if not simple and smax is None and self.xmap.resolution.high is not None:
            self.smax = 1 / (2 * self.xmap.resolution.high)
        if not simple:
            rlow = self.xmap.resolution.low
            if rlow is None:
                rlow = 1000
            self.smin = 1 / (2 * rlow)

        # Calculate transforms
        uc = xmap.unit_cell
        self.lattice_to_cartesian = uc.frac_to_orth / uc.abc
        self.cartesian_to_lattice = uc.orth_to_frac * uc.abc.reshape(3, 1)
        self.grid_to_cartesian = self.lattice_to_cartesian * self.xmap.voxelspacing
        structure_coor = self.structure.coor
        self._grid_coor = np.zeros_like(structure_coor)
        self._grid_coor_rot = np.zeros_like(structure_coor)

    def _coor_to_grid_coor(self):
        if np.allclose(self.xmap.origin, 0):
            coor = self.structure.coor
        else:
            coor = self.structure.coor - self.xmap.origin
        np.dot(coor, self.cartesian_to_lattice.T, self._grid_coor)
        self._grid_coor /= self.xmap.voxelspacing
        self._grid_coor -= self.xmap.offset

    def initialize(self):
        self.radial_densities = []
        for atom in self.structure.atoms:
            elem = atom.element.strip()
            if self.simple:
                rdens = self.simple_radial_density(elem, atom.b)[1]
            else:
                rdens = self.radial_density(elem, atom.b)[1]
            self.radial_densities.append(rdens)
        self.radial_densities = np.ascontiguousarray(self.radial_densities)
        self._initialized = True

    def density(self):
        """Transform structure to a density in a xmap."""
        if not self._initialized:
            self.initialize()

        self._coor_to_grid_coor()
        lmax = np.asarray(
            [self.rmax / vs for vs in self.xmap.voxelspacing], dtype=np.float64
        )
        active = self.structure.active
        q = self.structure.q
        for symop in self.xmap.unit_cell.space_group.symop_list:
            np.dot(self._grid_coor, symop.R.T, self._grid_coor_rot)
            # FIXME this can't be right...
            self._grid_coor_rot += symop.t * self.xmap.shape[::-1]
            dilate_points(
                self._grid_coor_rot,
                active,
                q,
                lmax,
                self.radial_densities,
                self.rstep,
                self.rmax,
                self.grid_to_cartesian,
                self.xmap.array,
            )
        #self.xmap.tofile("density_qfit.ccp4")

    def simple_radial_density(self, element, bfactor):
        """Calculate electron density as a function of radius."""

        # assert bfactor > 0, "B-factor should be bigger than 0"

        try:
            asf = self._asf[element.capitalize()]
        except KeyError:
            print("Unknown element:", element.capitalize())
            asf = self._asf["C"]
        four_pi2 = 4 * np.pi * np.pi
        bw = []
        for i in range(self.asf_range):
            divisor = asf[1][i] + bfactor
            if divisor <= 1e-4:
                bw.append(0)
            else:
                bw.append(-four_pi2 / (asf[1][i] + bfactor))
        aw = [asf[0][i] * (-bw[i] / np.pi) ** 1.5 for i in range(self.asf_range)]
        r = np.arange(0, self.rmax + self.rstep + 1, self.rstep)
        r2 = r * r
        density = np.zeros_like(r2)
        for i in range(self.asf_range):
            try:
                #exp_factor = bw[i] * r2
                density += aw[i] * np.exp(bw[i] * r2)
            except FloatingPointError:
                pass
        return r, density

    def radial_density(self, element, bfactor):
        """Calculate electron density as a function of radius."""
        r = np.arange(0, self.rmax + self.rstep + 1, self.rstep)
        density = np.zeros_like(r)
        for n, x in enumerate(r):
            asf = self._asf[element.capitalize()]
            args = (x, asf, bfactor, self.em)
            integrand, _ = fixed_quad(
                self._scattering_integrand, self.smin, self.smax, args=args, n=50
            )
            density[n] = integrand
        return r, density

    @staticmethod
    def _scattering_integrand(s, r, asf, bfactor, em):
        """Integral function to be approximated to obtain radial density."""
        s2 = s * s
        if em == True:
            f = (
                asf[0][0] * np.exp(-asf[1][0] * s2)
                + asf[0][1] * np.exp(-asf[1][1] * s2)
                + asf[0][2] * np.exp(-asf[1][2] * s2)
                + asf[0][3] * np.exp(-asf[1][3] * s2)
                + asf[0][4] * np.exp(-asf[1][4] * s2)
            )
        else:
            f = (
                asf[0][0] * np.exp(-asf[1][0] * s2)
                + asf[0][1] * np.exp(-asf[1][1] * s2)
                + asf[0][2] * np.exp(-asf[1][2] * s2)
                + asf[0][3] * np.exp(-asf[1][3] * s2)
                + asf[0][4] * np.exp(-asf[1][4] * s2)
                + asf[0][5]
            )
        w = 8 * f * np.exp(-bfactor * s2) * s
        a = 4 * np.pi * s
        if r > 1e-4:
            return w / r * np.sin(a * r)
        else:
            # Return 4th order Tayler expansion to prevent singularity
            return w * a * (1 - a * a * r * r / 6.0)


# XXX leaving the old implementation as the default for now
class Transformer(QfitTransformer):
    ...


def get_transformer(impl_name, *args, **kwds):
    """
    Instantiate a Transformer class using the specified implementation.
    """
    if impl_name == "fft":
        return FFTTransformer(*args, **kwds)
    elif impl_name == "cctbx":
        return CCTBXTransformer(*args, **kwds)
    else:
        return QfitTransformer(*args, **kwds)
