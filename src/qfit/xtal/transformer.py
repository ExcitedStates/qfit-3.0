import logging

import numpy as np
from numpy.fft import rfftn, irfftn
from scitbx.array_family import flex
from scipy.integrate import fixed_quad
from cctbx.maptbx import use_space_group_symmetry
import cctbx.masks
from cctbx.array_family import flex

from qfit.xtal.atomsf import ATOM_STRUCTURE_FACTORS, ELECTRON_SCATTERING_FACTORS
from qfit._extensions import dilate_points, mask_points  # pylint: disable=import-error,no-name-in-module


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

    def mask(self, rmax=None, value=1.0, implementation="qfit"):
        if rmax is None:
            rmax = self.rmax
        # TODO deprecate and remove this
        if implementation != "cctbx":
            self._coor_to_grid_coor()
            lmax = np.asarray(
                [rmax / vs for vs in self.xmap.voxelspacing], dtype=np.float64
            )
            active = self.structure.active
            for symop in self.xmap.unit_cell.space_group.symop_list:
                np.dot(self._grid_coor, symop.R.T, self._grid_coor_rot)
                # XXX why use xmap.shape when this might be an extracted
                # map?  shouldn't it be the unit cell shape here?
                self._grid_coor_rot += symop.t * self.xmap.shape[::-1]
                mask_points(
                    self._grid_coor_rot,
                    active,
                    lmax,
                    rmax,
                    self.grid_to_cartesian,
                    value,
                    self.xmap.array,
                )
        else:
            xrs = self.structure.to_xray_structure(active_only=True)
            sites_frac = xrs.expand_to_p1().sites_frac()
            n_real = tuple(int(x) for x in self.xmap.unit_cell_shape)
            # this mask is inverted, i.e. the region of interest has value 0
            mask_sel = cctbx.masks.around_atoms(
                xrs.unit_cell(),
                xrs.space_group().order_z(),
                sites_frac,
                flex.double(sites_frac.size(), rmax),
                n_real,
                0,
                0).data == 0
            mask_sel_np = np.swapaxes(mask_sel.as_numpy_array(), 0, 2)
            if not self.xmap.is_canonical_unit_cell():
                sel2 = self.xmap.get_unit_cell_grid_selection()
                mask_sel_np = mask_sel_np[sel2]
            self.xmap.array[mask_sel_np] = value

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
