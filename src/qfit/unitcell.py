"""Classes for handling unit cell transformation."""

from itertools import product

from cctbx import uctbx
import numpy as np
import numpy.linalg as la

from . import spacegroups


class UnitCell:

    """Class for storing and performing calculations on unit cell parameters.
    The constructor expects alpha, beta, and gamma to be in degrees.
    """

    def __init__(
        self,
        a=1.0,
        b=1.0,
        c=1.0,
        alpha=90.0,
        beta=90.0,
        gamma=90.0,
        space_group="P1"
    ):
        self._uctbx_cell = uctbx.unit_cell((a, b, c, alpha, beta, gamma))
        self.a = a
        self.b = b
        self.c = c

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.set_space_group(space_group)

        self.orth_to_frac = self.calc_fractionalization_matrix()
        self.frac_to_orth = self.calc_orthogonalization_matrix()

        ## check our math!
        # assert np.allclose(self.orth_to_frac, la.inv(self.frac_to_orth))

    def __str__(self):
        return "UnitCell(a=%f, b=%f, c=%f, alpha=%f, beta=%f, gamma=%f)" % (
            self.a,
            self.b,
            self.c,
            self.alpha,
            self.beta,
            self.gamma,
        )

    def to_cctbx(self):
        return self._uctbx_cell

    @staticmethod
    def from_cctbx(uc):
        return UnitCell(*(uc.parameters()))

    def copy(self):
        return UnitCell(
            self.a,
            self.b,
            self.c,
            self.alpha,
            self.beta,
            self.gamma,
            self.space_group.number,
        )

    @property
    def abc(self):
        return np.asarray([self.a, self.b, self.c], float)

    def calc_v(self):
        """Calculates the volume of the rhombohedral created by the
        unit vectors a1/|a1|, a2/|a2|, a3/|a3|.
        """
        return self._uctbx_cell.volume() / (self.a * self.b * self.c)

    def calc_volume(self):
        """Calculates the volume of the unit cell."""
        return self._uctbx_cell.volume()

    def calc_orthogonalization_matrix(self):
        """Cartesian to fractional coordinates."""
        return np.reshape(self._uctbx_cell.orthogonalization_matrix(), (3, 3))

    def calc_fractionalization_matrix(self):
        """Fractional to Cartesian coordinates."""
        return np.reshape(self._uctbx_cell.fractionalization_matrix(), (3, 3))

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
        RF = np.dot(symop.R, self.orth_to_frac)
        ORF = np.dot(self.frac_to_orth, RF)
        Ot = np.dot(self.frac_to_orth, symop.t)
        return spacegroups.SymOp(ORF, Ot)

    def calc_orth_symop2(self, symop):
        """Calculates the orthogonal space symmetry operation (return SymOp)
        given a fractional space symmetry operation (argument SymOp).
        """
        RF = np.dot(symop.R, self.orth_to_frac)
        ORF = np.dot(self.frac_to_orth, RF)
        Rt = np.dot(symop.R, symop.t)
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

    def iter_struct_orth_symops(self, structure, target=None, cushion=3):
        """Iterate over the orthogonal-space symmetry operations which will
        place a symmetry related structure near the argument struct.
        """
        ## compute the centroid of the structure
        coor = structure.coor
        centroid = coor.mean(axis=0)
        centroid_frac = self.calc_orth_to_frac(centroid)

        ## compute the distance from the centroid to the farthest point from
        ## it in the structure.
        diff = coor - centroid
        longest_dist_sq = (diff * diff).sum(axis=1).max()
        longest_dist = np.sqrt(longest_dist_sq)

        if target is not None:
            target_coor = target.coor
            target_centroid = target_coor.mean(axis=0)
            diff = target_coor - target_centroid
            target_longest_dist_sq = (diff * diff).sum(axis=1).max()
            target_longest_dist = np.sqrt(target_longest_dist_sq)
        else:
            target_centroid = centroid
            target_longest_dist = longest_dist

        max_dist = longest_dist + target_longest_dist + cushion
        cube = range(-3, 4)
        for symop in self.space_group.iter_symops():
            for cell_t in product(cube, repeat=3):
                cell_t = np.array(cell_t, float)
                symop_t = spacegroups.SymOp(symop.R, symop.t + cell_t)

                xyz_symm = symop_t(centroid_frac)
                centroid2 = self.calc_frac_to_orth(xyz_symm)

                if la.norm(target_centroid - centroid2) <= max_dist:
                    yield self.calc_orth_symop(symop_t)

    def set_space_group(self, space_group):
        self.space_group = spacegroups.getSpaceGroup(space_group)
