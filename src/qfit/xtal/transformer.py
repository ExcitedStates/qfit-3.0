import logging

from cctbx import maptbx
import numpy as np
import copy
from numpy.fft import fftn, ifftn, rfftn, irfftn
from scipy.integrate import fixed_quad

# from cctbx import maptbx, masks, miller
# from cctbx.sgtbx import space_group_info
# from cctbx.uctbx import unit_cell
# from cctbx.xray import structure_factors
# from cctbx.array_family import flex as flex_array
# from cctbx.xray import ext as xray_ext
# from mmtbx.utils import shift_origin



from .atomsf import ATOM_STRUCTURE_FACTORS, ELECTRON_SCATTERING_FACTORS
from .._extensions import dilate_points, mask_points, correlation_gradients 

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

def get_transformer(impl_name="cctbx", *args, **kwds):
    """
    Instantiate a Transformer class using the specified implementation.
    """
    if impl_name == "fft":
        return FFTTransformer(*args, **kwds)
    else:
        return Transformer(*args, **kwds)

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
            scattering = "electron"
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
            args = (x, asf, bfactor, self.em)
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
            args = (x, asf, bfactor, self.em)
            integrand, err = quadrature(
                self._scattering_integrand_derivative, self.smin, self.smax, args=args
            )
            derivative[n] = integrand
        return r, derivative

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


### CCTBX

# def fft_map_coefficients(map_coeffs, nyquist=2):
#     """
#     Transform CCTBX map coefficients (e.g. from an MTZ file) to a real map,
#     using symmetry-aware FFT gridding.  Returns the equivalent 3D numpy array.
#     """
#     fft_map = map_coeffs.fft_map(
#         resolution_factor=1/(2*nyquist),
#         symmetry_flags=maptbx.use_space_group_symmetry)
#     real_map = fft_map.apply_sigma_scaling().real_map_unpadded()
#     logger.info(f"FFT map dimensions from CCTBX: {real_map.focus()}")
#     return real_map.as_numpy_array()


# class Transformer:

#     """
#     Manager to transform a structure to a density or equivalent atom mask,
#     using CCTBX map sampling functions.  Modifies the xmap object in place.
#     """
#     def __init__(
#         self,
#         structure,
#         xmap,
#         rmax=3.0,
#         em=False,
#         smin=None,     # XXX unused, for API compatibility
#         smax=None,     # XXX unused, for API compatibility
#         simple=False,  # XXX unused, for API compatibility
#     ):
#         self.structure = structure
#         self.xmap = xmap
#         self.rmax = rmax
#         self.em = em

#     def get_masked_selection(self):
#         return self.xmap.array > 0

#     def mask(self, rmax=None, value=1.0):
#         """
#         Compute an atom mask around the current structure.  Modifies
#         self.xmap in place by adding values of 1 for masked points, 0 for
#         points outside the mask.  Assumes that the map has previously
#         been reset to 0 with self.reset(full=True).
#         """
#         if rmax is None:
#             rmax = self.rmax
#         return self._mask_cctbx(rmax, value)

#     def _get_structure_in_box(self):
#         # XXX the logic in here is a simplified version of the approach in
#         # mmtbx.utils.extract_box_around_model_and_map.  in the future it
#         # would be better to use that wrapper directly in qfit, in place
#         # of the calls to xmap.extract()
#         symm = self.structure.crystal_symmetry
#         if not symm:
#             symm = self.xmap.get_p1_crystal_symmetry()
#         xrs = self.structure.to_xray_structure(
#             active_only=True,
#             crystal_symmetry=symm)
#         if self.xmap.is_canonical_unit_cell():
#             return xrs.expand_to_p1()
#         origin = tuple(int(x) for x in self.xmap.grid_parameters.offset)
#         uc_grid = tuple(int(x) for x in self.xmap.unit_cell_shape)
#         n_real = self.xmap.n_real()
#         #logger.debug(f"Computing mask with n_real={n_real} origin={origin} uc_grid={uc_grid}")
#         ucp = symm.unit_cell().parameters()
#         box_cell_abc = [ucp[i]*(n_real[i]/uc_grid[i]) for i in range(3)]
#         uc_box = unit_cell(box_cell_abc + list(ucp)[3:])
#         #logger.debug(f"New unit cell: {uc_box.parameters()}")
#         sg_p1 = space_group_info("P1")
#         # XXX unlike the original qFit implementation, I am not even
#         # attempting to deal with space group symmetry right now.  weirdly,
#         # it's not clear if this even matters, since the old implementation
#         # seems slightly buggy
#         xrs_p1_box = xrs.customized_copy(space_group_info=sg_p1)
#         # this applies the shift to the xrs_p1_box object
#         soo = shift_origin(
#             xray_structure=xrs_p1_box,
#             n_xyz=uc_grid,
#             origin_grid_units=origin)
#         sites_cart = soo.xray_structure.sites_cart()
#         sites_frac = uc_box.fractionalize(sites_cart)
#         xrs_shifted = xrs_p1_box.customized_copy(unit_cell=uc_box)
#         xrs_shifted.set_sites_frac(sites_frac)
#         return xrs_shifted

#     def _mask_cctbx(self, rmax, value):
#         """
#         Compute an atom mask using cctbx.masks.  This method accounts for
#         map cutouts and origin shifts to match the original qFit behavior,
#         by temporarily translating the masked structure to fit in a P1 box
#         corresponding to the map extents.
#         """
#         xrs = self._get_structure_in_box()
#         n_real = self.xmap.n_real()
#         sites_frac = xrs.sites_frac()
#         # this mask is inverted, i.e. the region of interest has value 0
#         mask_sel = masks.around_atoms(
#             xrs.unit_cell(),
#             1,
#             sites_frac,
#             flex_array.double(sites_frac.size(), rmax),
#             n_real,
#             0,
#             0).data == 0
#         self.xmap.mask_with_value(mask_sel, value)

#     def get_conformers_mask(self, coor_set, rmax):
#         """
#         Get the combined map mask (as a numpy boolean array) for a series of
#         coordinates for the current structure.
#         """
#         assert len(coor_set) > 0
#         self.reset(full=True)
#         logger.debug(f"Masking {len(coor_set)} conformations")
#         for coor in coor_set:
#             self.structure.coor = coor
#             self.mask(rmax)
#         mask = self.xmap.array > 0
#         self.reset(full=True)
#         return mask

#     def get_conformer_density(self, coor, b):
#         self.structure.coor = coor
#         self.structure.b = b
#         self.density()
#         return self.xmap.array

#     # XXX unused, for API compatibility
#     def initialize(self):
#         ...

#     def reset(self, rmax=None, full=False):
#         if full:
#             self.xmap.array.fill(0)
#         else:
#             self.mask(rmax=rmax, value=0.0)

#     def density(self):
#         """
#         Compute the current model electron density using cctbx.xray map
#         sampling function, without any FFTs
#         """
#         xrs = self._get_structure_in_box()
#         if self.em:
#             #logger.debug("Switching to electron structure factor table")
#             xrs.discard_scattering_type_registry()
#             xrs.scattering_type_registry(table="electron")
#         else:
#             xrs.scattering_type_registry(table="n_gaussian")
#         n_real = self.xmap.n_real()
#         u_base = xray_ext.calc_u_base(
#             d_min=self.xmap.resolution.high,
#             grid_resolution_factor=0.25)
#         sampled_density = xray_ext.sampled_model_density(
#             unit_cell=xrs.unit_cell(),
#             scatterers=xrs.scatterers(),
#             scattering_type_registry=xrs.scattering_type_registry(),
#             fft_n_real=n_real,
#             fft_m_real=n_real,
#             u_base=u_base,
#             wing_cutoff=1e-3,
#             exp_table_one_over_step_size=-100,
#             force_complex=False,
#             use_u_base_as_u_extra=True,
#             sampled_density_must_be_positive=False,
#             tolerance_positive_definite=1e-5)
#         real_map = sampled_density.real_map_unpadded()
#         self.xmap.set_values_from_flex_array(real_map)


# class FFTTransformer(Transformer):
#     """
#     Alternative transformer for cases where we want to use the same set of
#     reflections (h,k,l) as the input map coefficients, and can tolerate the
#     overhead of running two FFTs.  Currently this is only used in scaler.py.
#     """

#     def __init__(self, structure, xmap, hkl=None, em=False, **kwds):
#         super().__init__(structure, xmap, em=em)
#         if hkl is None:
#             hkl = self.xmap.hkl
#         assert hkl is not None
#         self.hkl = hkl

#     def density(self):
#         """
#         Compute the electron density using via CCTBX structure factor FFT.
#         """
#         xrs = self.structure.to_xray_structure(active_only=True)
#         if self.em:
#             logger.debug("Switching to electron structure factor table")
#             xrs.discard_scattering_type_registry()
#             xrs.scattering_type_registry(table="electron")
#         assert self.structure.crystal_symmetry is not None
#         reflections = miller.set(xrs.crystal_symmetry(),
#                                  flex_array.miller_index(self.hkl),
#                                  anomalous_flag=False)
#         # XXX This is currently in closer agreement with the old "classic"
#         # implementation below, which does not use FFT, but it is much slower
#         sfs = structure_factors.from_scatterers(
#             crystal_symmetry=xrs.crystal_symmetry(),
#             d_min=self.xmap.resolution.high,
#             cos_sin_table=False,
#             quality_factor=None,
#             u_base=None,
#             b_base=None,
#             wing_cutoff=None)(
#                 xray_structure=xrs,
#                 miller_set=reflections,
#                 algorithm="fft")
#         grid = maptbx.crystal_gridding(
#             unit_cell = xrs.unit_cell(),
#             space_group_info = xrs.space_group_info(),
#             pre_determined_n_real = self.xmap.n_real())
#         fcalc_map = miller.fft_map(
#             crystal_gridding = grid,
#             fourier_coefficients = sfs.f_calc())
#         real_map = fcalc_map.apply_volume_scaling().real_map_unpadded()
#         self.xmap.set_values_from_flex_array(real_map)


# def get_transformer(impl_name="cctbx", *args, **kwds):
#     """
#     Instantiate a Transformer class using the specified implementation.
#     """
#     if impl_name == "fft":
#         return FFTTransformer(*args, **kwds)
#     else:
#         return Transformer(*args, **kwds)
