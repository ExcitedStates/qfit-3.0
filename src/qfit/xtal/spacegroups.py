"""Symmetry operations as functions on vectors or arrays."""

import numpy as np
from cctbx.sgtbx import space_group_info

from libtbx.utils import Sorry


class SymOp:
    """A subclass of the tuple class for performing one symmetry operation."""

    def __init__(self, R, t):
        self.R = R
        self.t = t

    def __str__(self):
        x = "[%6.3f %6.3f %6.3f %6.3f]\n" % (
            self.R[0, 0],
            self.R[0, 1],
            self.R[0, 2],
            self.t[0],
        )
        x += "[%6.3f %6.3f %6.3f %6.3f]\n" % (
            self.R[1, 0],
            self.R[1, 1],
            self.R[1, 2],
            self.t[1],
        )
        x += "[%6.3f %6.3f %6.3f %6.3f]" % (
            self.R[2, 0],
            self.R[2, 1],
            self.R[2, 2],
            self.t[2],
        )
        return x

    def __call__(self, vec):
        """Return the symmetry operation on argument vector and."""
        return np.dot(self.R, vec) + self.t

    def __eq__(self, symop):
        return np.allclose(self.R, symop[0]) and np.allclose(self.t, symop[1])

    def is_identity(self):
        """Returns True if this SymOp is an identity symmetry operation
        (no rotation, no translation), otherwise returns False.
        """
        value = np.allclose(self.R, np.identity(3, np.float64)) and np.allclose(
            self.t, np.zeros(3, np.float64)
        )
        return value


class SpaceGroup:
    """Contains the various names and symmetry operations for one space
    group.
    """

    def __init__(
        self,
        number=None,
        num_sym_equiv=None,
        num_primitive_sym_equiv=None,
        short_name=None,
        point_group_name=None,
        crystal_system=None,
        pdb_name=None,
        symop_list=None,
    ):
        self.number = number
        self.num_sym_equiv = num_sym_equiv
        self.num_primitive_sym_equiv = num_primitive_sym_equiv
        self.short_name = short_name
        self.point_group_name = point_group_name
        self.crystal_system = crystal_system
        self.pdb_name = pdb_name
        self.symop_list = symop_list

    def __repr__(self):
        return '"' + self.short_name + '"'

    def __str__(self):
        return self.short_name

    def iter_symops(self):
        """Iterates over all SymOps in the SpaceGroup."""
        return iter(self.symop_list)

    def iter_primitive_symops(self):
        """Iterate over symops required by the FFT module."""
        return iter(self.symop_list[: self.num_primitive_sym_equiv])

    def check_group_name(self, name):
        """Checks if the given name is a name for this space group, returns
        True or False. The space group name can be in several forms:
        the short name, the longer PDB-style name, or the space group number.
        """
        # In case a SpaceGroup instance is passed
        try:
            name = name.short_name
        except AttributeError:
            pass
        if name == self.short_name:
            return True
        if name == self.pdb_name:
            return True
        if name == self.point_group_name:
            return True
        if name == self.number:
            return True
        return False

    def iter_equivalent_positions(self, vec):
        """Iterate the symmetry equivalent positions of the argument vector.
        The vector must already be in fractional coordinates, and the symmetry
        equivalent vectors are also in fractional coordinates.
        """
        for symop in self.symop_list:
            yield symop(vec)

    @staticmethod
    def from_symbol(symbol):
        sg_info = space_group_info(symbol)
        return SpaceGroup.from_cctbx(sg_info)

    @staticmethod
    def from_cctbx(space_group_info):
        def _to_symop(op):
            n12 = op.as_double_array()
            return SymOp(
                np.array([n12[0:3], n12[3:6], n12[6:9]], np.float64),
                np.array(n12[9:12], np.float64),
            )

        group = space_group_info.group()
        point_group_type = group.point_group_type()
        if point_group_type[0] == "-":
            point_group_type = point_group_type[1] + "bar" + point_group_type[2:]
        num_sym_equiv = group.n_equivalent_positions()
        num_primitive_sym_equiv = group.order_p()
        symops = [_to_symop(op) for op in group.all_ops()]
        return SpaceGroup(
            number=space_group_info.type().number(),
            num_sym_equiv=num_sym_equiv,
            num_primitive_sym_equiv=num_primitive_sym_equiv,
            short_name=str(space_group_info).replace(" ", ""),
            point_group_name=f"PG{point_group_type}",
            crystal_system=group.crystal_system().upper(),
            pdb_name=str(space_group_info),
            symop_list=symops,
        )

    def to_cctbx(self):
        return space_group_info(self.number)

    def as_cctbx_group(self):
        return self.to_cctbx().group()


def getSpaceGroup(name):
    """Returns the SpaceGroup instance for the given name. If the space group
    is not found, return the P1 space group as default.
    """
    try:
        return SpaceGroup.from_symbol(name)
    except RuntimeError as e:
        raise Sorry(f"{e}") from e


# XXX unused, but preserving in case of future need
def SymOpFromString(string):
    """Return the SymOp described by string."""

    ops = string.split(",")
    tr = "Tr"
    rot = "Rot"
    for op in ops:
        parts = op.split("+")
        # Determine the translational part
        if "/" in parts[-1]:
            t = parts.pop()
            nom, denom = t.split("/")
            tr += f"_{nom}{denom}"
        else:
            tr += "_0"

        # Get the rotational part
        rot += "_"
        r = list("".join(parts).strip())[::-1]
        while r:
            l = r.pop()
            if l == "-":
                rot += "m"
                l = r.pop()
            rot += l

    g = globals()
    R = g[rot]
    T = g[tr]
    return SymOp(R, T)
