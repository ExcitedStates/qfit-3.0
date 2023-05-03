import numpy as np
import copy
from numpy.fft import fftn, ifftn, rfftn, irfftn
from scipy.integrate import fixed_quad

from .atomsf import ATOM_STRUCTURE_FACTORS, ELECTRON_SCATTERING_FACTORS
from ._extensions import dilate_points, mask_points, correlation_gradients


class SFTransformer:
    def __init__(self, hkl, f, phi, unit_cell):
        self.hkl = hkl
        self.f = f
        self.phi = phi

        self.unit_cell = unit_cell
        self.space_group = unit_cell.space_group

        abc = self.unit_cell.abc
        hkl_max = np.abs(hkl).max(axis=0)
        self._resolution = np.min(abc / hkl_max)

    def __call__(self, nyquist=2):

        h, k, l = self.hkl.T

        naive_voxelspacing = self._resolution / (2 * nyquist)
        naive_shape = np.ceil(self.unit_cell.abc / naive_voxelspacing)
        naive_shape = naive_shape[::-1].astype(int)
        shape = naive_shape
        # TODO make grid of unit cell a multiple of small primes
        # efficient_multiples = [2, 3, 5, 7, 11, 13]
        # shape = [closest_upper_multiple(x, efficient_multiples) for x in naive_shape]
        fft_grid = np.zeros(shape, dtype=np.complex128)

        start_sf = self._f_phi_to_complex(self.f, self.phi)
        symops = self.space_group.symop_list[: self.space_group.num_primitive_sym_equiv]
        two_pi = 2 * np.pi
        hsym = np.zeros_like(h)
        ksym = np.zeros_like(k)
        lsym = np.zeros_like(l)
        for symop in symops:
            for n, msym in enumerate((hsym, ksym, lsym)):
                msym.fill(0)
                rot = np.asarray(symop.R.T)[n]
                for r, m in zip(rot, (h, k, l)):
                    if r != 0:
                        msym += int(r) * m
            if np.allclose(symop.t, 0):
                sf = start_sf
            else:
                delta_phi = np.rad2deg(np.inner((-two_pi * symop.t), self.hkl))
                delta_phi = np.asarray(delta_phi).ravel()
                sf = self._f_phi_to_complex(self.f, self.phi + delta_phi)
            fft_grid[lsym, ksym, hsym] = sf
            fft_grid[-lsym, -ksym, -hsym] = sf.conj()
        # grid = np.fft.ifftn(fft_grid)
        nx = shape[-1]
        grid = np.fft.irfftn(fft_grid[:, :, : nx // 2 + 1])
        grid -= grid.mean()
        grid /= grid.std()
        return grid

    def _f_phi_to_complex(self, f, phi):
        sf = np.nan_to_num((f * np.exp(-1j * np.deg2rad(phi))).astype(np.complex64))
        return sf


class FFTTransformer:

    """Transform a structure in a map via FFT"""

    def __init__(self, structure, xmap, hkl=None, em=False, b_add=None):
        self.structure = structure
        self.xmap = xmap
        self.asf_range = 6
        self.em = em
        if self.em == True:
            scattering="electron"
            self.asf_range = 5
        if hkl is None:
            hkl = self.xmap.hkl
        self.hkl = hkl
        self.b_add = b_add
        h, k, l = self.hkl.T
        fft_mask = np.ones(self.xmap.shape, dtype=bool)
        fft_mask.fill(True)
        sg = self.xmap.unit_cell.space_group
        symops = sg.symop_list[: sg.num_primitive_sym_equiv]
        hmax = 0
        hsym = np.zeros_like(h)
        ksym = np.zeros_like(k)
        lsym = np.zeros_like(l)
        for symop in symops:
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
        grid = irfftn(fft_grid)
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

    def mask(self, rmax=None, value=1.0):
        self._coor_to_grid_coor()
        if rmax is None:
            rmax = self.rmax
        lmax = np.asarray(
            [rmax / vs for vs in self.xmap.voxelspacing], dtype=np.float64
        )
        active = self.structure.active
        for symop in self.xmap.unit_cell.space_group.symop_list:
            np.dot(self._grid_coor, symop.R.T, self._grid_coor_rot)
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

    def reset(self, rmax=None, full=False):
        if full:
            self.xmap.array.fill(0)
        else:
            self.mask(rmax=rmax, value=0.0)

    def initialize(self, derivative=False):
        self.radial_densities = []
        for n in range(self.structure.natoms):
            if self.simple:
                rdens = self.simple_radial_density(
                    self.structure.e[n], self.structure.b[n]
                )[1]
            else:
                rdens = self.radial_density(self.structure.e[n], self.structure.b[n])[1]
            self.radial_densities.append(rdens)
        self.radial_densities = np.ascontiguousarray(self.radial_densities)

        if derivative:
            self.radial_derivatives = np.zeros_like(self.radial_densities)
            if self.simple:
                for n, (e, b) in enumerate(zip(self.structure.e, self.structure.b)):
                    self.radial_derivatives[n] = self.simple_radial_derivative(e, b)[1]
            else:
                # self.radial_derivatives[n] = self.radial_derivative(e, b)[1]
                self.radial_derivatives = np.gradient(
                    self.radial_densities, self.rstep, edge_order=2, axis=1
                )

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
                exp_factor = bw[i] * r2
                density += aw[i] * np.exp(bw[i] * r2)
            except FloatingPointError:
                pass
        return r, density

    def simple_radial_derivative(self, element, bfactor):
        """Calculate gradient."""
        r = np.arange(0, self.rmax + self.rstep + 1, self.rstep)
        r2 = r * r
        asf = self._asf[element.capitalize()]
        four_pi2 = 4 * np.pi * np.pi
        bw = [-four_pi2 / (asf[1][i] + bfactor) for i in range(self.asf_range)]
        aw = [asf[0][i] * (-bw[i] / np.pi) ** 1.5 for i in range(self.asf_range)]
        derivative = np.zeros(r.size, np.float64)
        for i in range(self.asf_range):
            derivative += (2.0 * bw[i] * aw[i]) * np.exp(bw[i] * r2)
        derivative *= r
        return r, derivative

    def correlation_gradients(self, target):

        self._coor_to_grid_coor()
        lmax = np.asarray(
            [self.rmax / vs for vs in self.xmap.voxelspacing], dtype=np.float64
        )
        gradients = np.zeros(self.structure.coor)

        active = self.structure.active
        q = self.structure.q
        correlation_gradients(
            self._grid_coor,
            active,
            q,
            lmax,
            self.radial_derivatives,
            self.rstep,
            self.rmax,
            self.grid_to_cartesian,
            target.array,
            gradients,
        )
        return gradients

    def radial_density(self, element, bfactor):
        """Calculate electron density as a function of radius."""
        r = np.arange(0, self.rmax + self.rstep + 1, self.rstep)
        density = np.zeros_like(r)
        for n, x in enumerate(r):
            asf = self._asf[element.capitalize()]
            args = (x, asf, bfactor)
            # Use a fixed number of quadrature points, 50 is more than enough
            # integrand, err = quadrature(self._scattering_integrand, self.smin,
            #                            self.smax, args=args)#, tol=1e-5, miniter=13, maxiter=15)
            integrand, err = fixed_quad(
                self._scattering_integrand, self.smin, self.smax, args=args, n=50
            )
            density[n] = integrand
        return r, density

    def radial_derivative(self, element, bfactor):
        r = np.arange(0, self.rmax + self.rstep + 1, self.rstep)
        derivative = np.zeros_like(r)
        for n, x in enumerate(r):
            asf = self._asf[element.capitalize()]
            args = (x, asf, bfactor)
            integrand, err = quadrature(
                self._scattering_integrand_derivative, self.smin, self.smax, args=args
            )
            derivative[n] = integrand
        return r, derivative

    @staticmethod
    def _scattering_integrand(s, r, asf, bfactor):
        """Integral function to be approximated to obtain radial density."""
        s2 = s * s
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

    @staticmethod
    def _scattering_integrand_derivative(s, r, asf, bfactor):
        s2 = s * s
        f = asf[0][5]
        for a, b in zip(asf[0], asf[1]):
            # f += asf[0][i] * np.exp(-asf[1][i] * s2)
            f += a * np.exp(-b * s2)
        a = 4 * np.pi * s
        w = 8 * f * np.exp(-bfactor * s2) * s
        ar = a * r
        if r > 1e-4:
            return w / r * (a * np.cos(ar) - np.sin(ar) / r)
        else:
            # Return 4th order Tayler expansion
            ar2 = ar * ar
            a3 = a * a * a
            return w * a3 * r * (ar2 - 8) / 24.0
