import logging

import numpy as np
from numpy.fft import rfftn, irfftn
from scipy.integrate import fixed_quad
from cctbx.maptbx import use_space_group_symmetry, crystal_gridding
from cctbx.masks import around_atoms as mask_around_atoms
from cctbx.miller import fft_map
from cctbx.sgtbx import space_group_info
from cctbx.uctbx import unit_cell
from cctbx.array_family import flex
from mmtbx.utils import shift_origin

from qfit.xtal.atomsf import ATOM_STRUCTURE_FACTORS, ELECTRON_SCATTERING_FACTORS
from qfit._extensions import dilate_points  # pylint: disable=import-error,no-name-in-module


logger = logging.getLogger(__name__)


def fft_map_coefficients(map_coeffs, nyquist=2):
    """
    Transform CCTBX map coefficients to a real map with swapped X and Z axes
    """
    fft_map = map_coeffs.fft_map(
        resolution_factor=1/(2*nyquist),
        symmetry_flags=use_space_group_symmetry)
    real_map = fft_map.apply_sigma_scaling().real_map_unpadded()
    logger.info(f"FFT map dimensions before axis swap: {real_map.focus()}")
    return np.swapaxes(real_map.as_numpy_array(), 0, 2)


class FFTTransformer:

    """
    Transform a structure in a map via FFT.  Modifies the xmap object in place.
    """

    def __init__(self, structure, xmap, hkl=None, em=False, b_add=None):
        self.structure = structure
        self.xmap = xmap
        self.asf_range = 6
        self.em = em
        if self.em == True:
            self.asf_range = 5
        if hkl is None:
            hkl = self.xmap.hkl
        self.hkl = hkl
        self.b_add = b_add
        h, k, l = self.hkl.T
        fft_mask = np.ones(self.xmap.shape, dtype=bool)
        fft_mask.fill(True)
        sg = self.xmap.unit_cell.space_group
        hmax = 0
        hsym = np.zeros_like(h)
        ksym = np.zeros_like(k)
        lsym = np.zeros_like(l)
        for symop in sg.iter_primitive_symops():
            for n, msym in enumerate((hsym, ksym, lsym)):
                msym.fill(0)
                rot = np.asarray(symop.R.T)[n]
                for r, m in zip(rot, (h, k, l)):
                    if r != 0:
                        msym += int(r) * m
            fft_mask[lsym, ksym, hsym] = False
            fft_mask[-lsym, -ksym, -hsym] = False
        # Keep the density on absolute level
        fft_mask[0, 0, 0] = False
        hmax = fft_mask.shape[-1] // 2 + 1
        # self._fft_mask = fft_mask
        self._fft_mask = fft_mask[:, :, :hmax].copy()
        self.hkl = hkl
        if self.b_add is not None:
            b_original = self.structure.b
            self.structure.b += self.b_add
        self._transformer = Transformer(
            self.structure, self.xmap, simple=True, em=self.em
        )
        self._transformer.initialize()
        if b_add is not None:
            self.structure.b = b_original

    def initialize(self):
        self._transformer.initialize()

    def mask(self, *args, **kwargs):
        self._transformer.mask(*args, **kwargs)

    def reset(self, *args, **kwargs):
        self._transformer.reset(*args, **kwargs)

    def density(self):
        self._transformer.density()
        fft_grid = rfftn(self.xmap.array)
        fft_grid[self._fft_mask] = 0
        grid = irfftn(fft_grid, self.xmap.shape)
        if self.b_add is not None:
            pass
        self.xmap.array[:] = grid.real


class Transformer:

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
        self.structure = structure
        self.xmap = xmap
        self.smin = smin
        self.smax = smax
        self.rmax = rmax
        self.rstep = rstep
        self.simple = simple
        self.em = em
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
        n_real = tuple(int(x) for x in self.xmap.shape[::-1])
        logger.debug(f"Computing mask with n_real={n_real} origin={origin} uc_grid={uc_grid}")
        ucp = self.structure.crystal_symmetry.unit_cell().parameters()
        box_cell_abc = [ucp[i]*(n_real[i]/uc_grid[i]) for i in range(3)]
        uc_box = unit_cell(box_cell_abc + list(ucp)[3:])
        logger.debug(f"New unit cell: {uc_box.parameters()}")
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
        # XXX the logic in here is a simplified version of the approach in
        # mmtbx.utils.extract_box_around_model_and_map.  in the future it
        # would be better to use that wrapper directly in qfit, in place
        # of the calls to xmap.extract()
        xrs = self._get_structure_in_box()
        n_real = tuple(int(x) for x in self.xmap.shape[::-1])
        sites_frac = xrs.sites_frac()
        # this mask is inverted, i.e. the region of interest has value 0
        mask_sel = mask_around_atoms(
            xrs.unit_cell(),
            1,
            sites_frac,
            flex.double(sites_frac.size(), rmax),
            n_real,
            0,
            0).data == 0
        mask_sel_np = np.swapaxes(mask_sel.as_numpy_array(), 0, 2)
        assert np.sum(mask_sel_np) > 0
        self.xmap.array[mask_sel_np] += value

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
        #self.xmap.tofile("mask_cctbx.ccp4")
        self.reset(full=True)
        return mask

    def get_conformer_density(self, coor, b):
        self.structure.coor = coor
        self.structure.b = b
        self.density()
        return self.xmap.array

    def reset(self, rmax=None, full=False):
        if full:
            self.xmap.array.fill(0)
        else:
            self.mask(rmax=rmax, value=0.0)

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
        if False:
            return self._density_cctbx_fft()
        else:
            return self._density_qfit()

    def _density_cctbx_fft(self):
        """
        Compute the electron density using via CCTBX structure factors.
        """
        xrs = self._get_structure_in_box()
        if self.em:
            logger.debug("Switching to electron structure factor table")
            xrs.discard_scattering_type_registry()
            xrs.scattering_type_registry(table="electron")
        fcalc = xrs.structure_factors(
            anomalous_flag=False,
            d_min=self.xmap.resolution.high,
            algorithm="fft").f_calc()
        n_real = tuple(int(x) for x in self.xmap.shape[::-1])
        grid = crystal_gridding(
            unit_cell = xrs.unit_cell(),
            space_group_info = xrs.space_group_info(),
            pre_determined_n_real = n_real)
        fcalc_map = fft_map(
            crystal_gridding = grid,
            fourier_coefficients = fcalc)
        real_map = fcalc_map.apply_volume_scaling().real_map_unpadded()
        self.xmap.array[:] = np.swapaxes(real_map.as_numpy_array(), 0, 2)
        #self.xmap.tofile("density_cctbx.ccp4")

    def _density_qfit(self):
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
