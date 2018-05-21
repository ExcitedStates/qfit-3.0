"""Classes for handling unit cell transformation."""

import numpy as np
import numpy.linalg as la

from . import spacegroups


class UnitCell:

    """Class for storing and performing calculations on unit cell parameters.
    The constructor expects alpha, beta, and gamma to be in degrees.
    """

    def __init__(self, a=1.0, b=1.0, c=1.0,
                 alpha=90.0, beta=90.0, gamma=90.0,
                 space_group="P1", shape=None):

        self.a = a
        self.b = b
        self.c = c

        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

        self.set_space_group(space_group)
        self.shape = shape

        self._sin_alpha = np.sin(np.deg2rad(self.alpha))
        self._sin_beta  = np.sin(np.deg2rad(self.beta))
        self._sin_gamma = np.sin(np.deg2rad(self.gamma))

        self._cos_alpha = np.cos(np.deg2rad(self.alpha))
        self._cos_beta  = np.cos(np.deg2rad(self.beta))
        self._cos_gamma = np.cos(np.deg2rad(self.gamma))

        self.orth_to_frac = self.calc_fractionalization_matrix()
        self.frac_to_orth = self.calc_orthogonalization_matrix()

        ## check our math!
        #assert np.allclose(self.orth_to_frac, la.inv(self.frac_to_orth))

    def __str__(self):
        return "UnitCell(a=%f, b=%f, c=%f, alpha=%f, beta=%f, gamma=%f)" % (
            self.a, self.b, self.c, self.alpha, self.beta, self.gamma)

    def calc_v(self):
        """Calculates the volume of the rhombohedral created by the
        unit vectors a1/|a1|, a2/|a2|, a3/|a3|.
        """
        return np.sqrt(1 -
                       (self._cos_alpha * self._cos_alpha) -
                       (self._cos_beta  * self._cos_beta)  -
                       (self._cos_gamma * self._cos_gamma) +
                       (2 * self._cos_alpha * self._cos_beta * self._cos_gamma)
                      )

    def calc_volume(self):
        """Calculates the volume of the unit cell.
        """
        return self.a * self.b * self.c * self.calc_v()

    def calc_reciprocal_unit_cell(self):
        """Corresponding reciprocal unit cell.
        """
        V = self.calc_volume()

        ra  = (self.b * self.c * self._sin_alpha) / V
        rb  = (self.a * self.c * self._sin_beta)  / V
        rc  = (self.a * self.b * self._sin_gamma) / V

        ralpha = np.arccos(
            (self._cos_beta  * self._cos_gamma - self._cos_alpha) / (self._sin_beta  * self._sin_gamma))
        rbeta  = np.arccos(
            (self._cos_alpha * self._cos_gamma - self._cos_beta)  / (self._sin_alpha * self._sin_gamma))
        rgamma = np.arccos(
            (self._cos_alpha * self._cos_beta  - self._cos_gamma) / (self._sin_alpha * self._sin_beta))

        return UnitCell(ra, rb, rc, ralpha, rbeta, rgamma)

    def calc_orthogonalization_matrix(self):
        """Cartesian to fractional coordinates.
        """

        v = self.calc_v()

        f11 = self.a
        f12 = self.b * self._cos_gamma
        f13 = self.c * self._cos_beta
        f22 = self.b * self._sin_gamma
        f23 = (self.c * (self._cos_alpha - self._cos_beta * self._cos_gamma)) / (self._sin_gamma)
        f33 = (self.c * v) / self._sin_gamma

        orth_to_frac = np.array([ [f11, f12, f13],
                                     [0.0, f22, f23],
                                     [0.0, 0.0, f33] ], float)

        return orth_to_frac

    def calc_fractionalization_matrix(self):
        """Fractional to Cartesian coordinates.
        """

        v = self.calc_v()

        o11 = 1.0 / self.a
        o12 = - self._cos_gamma / (self.a * self._sin_gamma)
        o13 = (self._cos_gamma * self._cos_alpha - self._cos_beta) / (self.a * v * self._sin_gamma)
        o22 = 1.0 / (self.b * self._sin_gamma)
        o23 = (self._cos_gamma * self._cos_beta - self._cos_alpha) / (self.b * v * self._sin_gamma)
        o33 = self._sin_gamma / (self.c * v)

        frac_to_orth = np.array([ [o11, o12, o13],
                                     [0.0, o22, o23],
                                     [0.0, 0.0, o33] ], float)

        return frac_to_orth

    def calc_orth_to_frac(self, v):
        """Calculates and returns the fractional coordinate vector of
        orthogonal vector v.
        """
        return np.dot(self.orth_to_frac, v)

    def calc_frac_to_orth(self, v):
        """Calculates and returns the orthogonal coordinate vector of
        fractional vector v.
        """
        return np.dot(self.frac_to_orth, v)

    def calc_orth_symop(self, symop):
        """Calculates the orthogonal space symmetry operation (return SymOp)
        given a fractional space symmetry operation (argument SymOp).
        """
        RF  = np.dot(symop.R, self.orth_to_frac)
        ORF = np.dot(self.frac_to_orth, RF)
        Ot  = np.dot(self.frac_to_orth, symop.t)
        return spacegroups.SymOp(ORF, Ot)

    def calc_orth_symop2(self, symop):
        """Calculates the orthogonal space symmetry operation (return SymOp)
        given a fractional space symmetry operation (argument SymOp).
        """
        RF  = np.dot(symop.R, self.orth_to_frac)
        ORF = np.dot(self.frac_to_orth, RF)
        Rt  = np.dot(symop.R, symop.t)
        ORt = np.dot(self.frac_to_orth, Rt)

        return spacegroups.SymOp(ORF, ORt)

    def calc_cell(self, xyz):
        """Returns the cell integer 3-Tuple where the xyz fractional
        coordinates are located.
        """
        if xyz[0] < 0.0:
            cx = int(xyz[0] - 1.0)
        else:
            cx = int(xyz[0] + 1.0)

        if xyz[1] < 0.0:
            cy = int(xyz[1] - 1.0)
        else:
            cy = int(xyz[1] + 1.0)

        if xyz[2] < 0.0:
            cz = int(xyz[2] - 1.0)
        else:
            cz = int(xyz[2] + 1.0)

        return (cx, cy, cz)

    def iter_struct_orth_symops(self, struct):
        """Iterate over the orthogonal-space symmetry operations which will
        place a symmetry related structure near the argument struct.
        """
        ## compute the centroid of the structure
        cent = np.zeros(3, float)
        for n, atm in enumerate(struct.iter_all_atoms()):
            cent += atm.position
        centroid = cent / n

        ccell = self.calc_cell(self.calc_orth_to_frac(centroid))
        centroid_cell = np.array(ccell, float)

        ## compute the distance from the centroid to the farthest point from
        ## it in the structure.
        max_dist = 0.0
        for frag in struct.iter_amino_acids():
            for atm in frag.iter_atoms():
                dist = la.norm.length(atm.position - centroid)
                max_dist = max(max_dist, dist)
        max_dist2 = 2.0 * max_dist + 5.0

        cube = range(-3, 4)
        for symop in self.space_group.iter_symops():
            for i, j, k in product(cube, repeat=3):

                 cell_t  = np.array([i, j, k], float)
                 symop_t = spacegroups.SymOp(symop.R, symop.t+cell_t)

                 xyz       = self.calc_orth_to_frac(centroid)
                 xyz_symm  = symop_t(xyz)
                 centroid2 = self.calc_frac_to_orth(xyz_symm)

                 if la.norm(centroid - centroid2) <= max_dist2:
                     yield self.calc_orth_symop(symop_t)

    def set_space_group(self, space_group):
        self.space_group = spacegroups.GetSpaceGroup(space_group)


def strRT(R, T):
    """Returns a string for a rotation/translation pair in a readable form.
    """
    x  = "[%6.3f %6.3f %6.3f %6.3f]\n" % (
        R[0,0], R[0,1], R[0,2], T[0])
    x += "[%6.3f %6.3f %6.3f %6.3f]\n" % (
        R[1,0], R[1,1], R[1,2], T[1])
    x += "[%6.3f %6.3f %6.3f %6.3f]\n" % (
        R[2,0], R[2,1], R[2,2], T[2])

    return x


## <testing>
#def test_module():
#    print("=================================================")
#    print("TEST CASE #1: Triclinic unit cell")
#    print()
#
#    uc = UnitCell(7.877, 7.210, 7.891, 105.563, 116.245, 79.836)
#
#    e = np.array([[1.0, 0.0, 0.0],
#                     [0.0, 1.0, 0.0],
#                     [0.0, 0.0, 1.0]], float)
#
#
#    print uc
#    print("volume                   = ", uc.calc_v())
#    print("cell volume              = ", uc.calc_volume())
#    print("fractionalization matrix =\n", uc.calc_fractionalization_matrix())
#    print("orthogonalization matrix =\n", uc.calc_orthogonalization_matrix())
#
#    print("orth * e =\n", np.dot(
#        uc.calc_orthogonalization_matrix(), e))
#
#
#    print "calc_frac_to_orth"
#    vlist = [
#        np.array([0.0, 0.0, 0.0]),
#        np.array([0.5, 0.5, 0.5]),
#        np.array([1.0, 1.0, 1.0]),
#        np.array([-0.13614, 0.15714, -0.07165]) ]
#
#    for v in vlist:
#        ov = uc.calc_frac_to_orth(v)
#        v2 = uc.calc_orth_to_frac(ov)
#        print "----"
#        print "    ",v
#        print "    ",ov
#        print "    ",v2
#        print "----"
#
#
#    print "================================================="
#
#    print
#
#    print "================================================="
#    print "TEST CASE #2: Reciprocal of above unit cell "
#    print
#
#    ruc = uc.calc_reciprocal_unit_cell()
#    print ruc
#    print "volume      = ", ruc.calc_v()
#    print "cell volume = ", ruc.calc_volume()
#
#    print "================================================="
#
#    print
#
#    print "================================================="
#    print "TEST CASE #3: Orthogonal space symmetry operations"
#
#    unitx = UnitCell(a           = 64.950,
#                     b           = 64.950,
#                     c           = 68.670,
#                     alpha       = 90.00,
#                     beta        = 90.00,
#                     gamma       = 120.00,
#                     space_group = "P 32 2 1")
#    print unitx
#    print
#
#    for symop in unitx.space_group.iter_symops():
#        print "Fractional Space SymOp:"
#        print symop
#        print "Orthogonal Space SymOp:"
#        print unitx.calc_orth_symop(symop)
#        print
#
#if __name__ == "__main__":
#    test_module()
## </testing>
