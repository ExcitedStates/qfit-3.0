import logging
from collections import defaultdict
from collections.abc import Iterable
from operator import eq, gt, ge, le, lt

import numpy as np
from molmass.elements import ELEMENTS

from .math import dihedral_angle
from .pdbfile import write_pdb, write_mmcif
from .selector import _Selector

logger = logging.getLogger(__name__)


class _BaseStructure:
    REQUIRED_ATTRIBUTES = [
        "record",
        "name",
        "b",
        "q",
        "coor",
        "resn",
        "resi",
        "icode",
        "e",
        "charge",
        "chain",
        "altloc",
    ]
    OTHER_ATTRIBUTES = ["link_data", "crystal_symmetry", "unit_cell", "file_format"]
    _DTYPES = [str, str, float, float, float, str, int, str, str, str, str, float]
    _selector = _Selector()
    _COMPARISON_DICT = {"==": eq, "!=": eq, ">": gt, ">=": ge, "<=": le, "<": lt}

    def __init__(self, data, selection=None, parent=None, **kwargs):
        self.parent = parent
        self.data = data
        self._selection = selection
        # Save extra kwargs for general extraction and duplication methods.
        self._kwargs = kwargs
        self.link_data = {}
        self.crystal_symmetry = None
        self.unit_cell = None
        self.file_format = None  # "pdb"  # default is PDB
        for attr, array in data.items():
            hattr = "_" + attr
            setattr(self, hattr, array)
            prop = self._structure_property(hattr)
            setattr(self.__class__, attr, prop)
        self._x, self._y, self._z = self._coor.T
        for attr in "xyz":
            hattr = "_" + attr
            prop = self._structure_property(hattr)
            setattr(self.__class__, attr, prop)

        for key, value in kwargs.items():
            if key in self.OTHER_ATTRIBUTES:
                setattr(self, key, value)

        if selection is None:
            self.natoms = self._coor.shape[0]
        else:
            self.natoms = self._selection.size

    def _structure_property(self, property_name, docstring=None):
        def getter(self):
            if self._selection is None:
                return self.__getattribute__(property_name).copy()
            else:
                return self.__getattribute__(property_name)[self._selection]

        def setter(self, value):
            if self._selection is None:
                getattr(self, property_name)[:] = value
            else:
                getattr(self, property_name)[self._selection] = value

        return property(getter, setter, doc=docstring)

    def _get_property(self, ptype):
        elements, ind = np.unique(self.e, return_inverse=True)
        values = []
        for e in elements:
            try:
                value = getattr(ELEMENTS[e.capitalize()], ptype)
            except KeyError:
                logger.warning(
                    "Unknown element {:s}. Using Carbon parameter instead.".format(e)
                )
                value = getattr(ELEMENTS["C"], ptype)
            values.append(value)
        out = np.asarray(values, dtype=np.float64)[ind]
        return out

    def get_selected_data(self):
        if self._selection is None:
            return self.data
        else:
            return {k: v[self._selection] for k, v in self.data.items()}

    @property
    def covalent_radius(self):
        return self._get_property("covrad")

    @property
    def vdw_radius(self):
        return self._get_property("vdwrad")

    def copy(self):
        data = {}
        for attr in self.data:
            data[attr] = getattr(self, attr).copy()
        return self.__class__(data, parent=None, selection=None, **self._kwargs)

    def get_dihedral_angle(self, coor):
        return dihedral_angle(coor)

    def extract(self, *args):
        if not isinstance(args[0], str):
            selection = args[0]
        else:
            selection = self.select(*args)
        return self.__class__(
            self.data, selection=selection, parent=self, **self._kwargs
        )

    def rotate(self, R):
        """Rotate structure"""
        coor = np.dot(self.coor, R.T)  # pylint: disable=access-member-before-definition
        self.coor = coor

    def rmsd(self, structure):
        coor1 = self.coor
        coor2 = structure.coor
        if coor1.shape != coor2.shape:
            raise ValueError("Coordinate shapes are not equivalent")
        if "TYR" in self.resn:
            idx_cd1 = structure.name.tolist().index("CD1")
            idx_cd2 = structure.name.tolist().index("CD2")
            idx_ce1 = structure.name.tolist().index("CE1")
            idx_ce2 = structure.name.tolist().index("CE2")
            coor3 = np.copy(coor2)
            coor3[idx_cd1], coor3[idx_cd2] = coor2[idx_cd2], coor2[idx_cd1]
            coor3[idx_ce1], coor3[idx_ce2] = coor2[idx_ce2], coor2[idx_ce1]
            diff = (coor1 - coor2).ravel()
            diff2 = (coor1 - coor3).ravel()
            return min(
                np.sqrt(3 * np.inner(diff, diff) / diff.size),
                np.sqrt(3 * np.inner(diff2, diff2) / diff2.size),
            )
        if "PHE" in self.resn:
            idx_cd1 = structure.name.tolist().index("CD1")
            idx_cd2 = structure.name.tolist().index("CD2")
            idx_ce1 = structure.name.tolist().index("CE1")
            idx_ce2 = structure.name.tolist().index("CE2")
            coor3 = np.copy(coor2)
            coor3[idx_cd1], coor3[idx_cd2] = coor2[idx_cd2], coor2[idx_cd1]
            coor3[idx_ce1], coor3[idx_ce2] = coor2[idx_ce2], coor2[idx_ce1]
            diff = (coor1 - coor2).ravel()
            diff2 = (coor1 - coor3).ravel()
            return min(
                np.sqrt(3 * np.inner(diff, diff) / diff.size),
                np.sqrt(3 * np.inner(diff2, diff2) / diff2.size),
            )
        else:
            diff = (coor1 - coor2).ravel()
            return np.sqrt(3 * np.inner(diff, diff) / diff.size)

    def select(self, string, values=None, comparison="=="):
        if values is None:
            self._selector.set_structure(self)
            selection = self._selector(string)
        else:
            selection = self._simple_select(string, values, comparison)
        return selection

    def _simple_select(self, attr, values, comparison_str):
        data = getattr(self, attr)
        comparison = self._COMPARISON_DICT[comparison_str]
        if not isinstance(values, Iterable) or isinstance(values, str):
            values = (values,)
        mask = np.zeros(self.natoms, bool)
        for value in values:
            mask2 = comparison(data, value)
            np.logical_or(mask, mask2, out=mask)
        if comparison_str == "!=":
            np.logical_not(mask, out=mask)
        if self._selection is None:
            selection = np.flatnonzero(mask)
        else:
            selection = self._selection[mask]
        return selection

    def tofile(self, fname, cryst=None):
        if fname.endswith(".pdb") or fname.endswith(".pdb.gz"):
            return self.to_pdb_file(fname, cryst)
        elif fname.endswith(".cif") or fname.endswith(".cif.gz"):
            return self.to_mmcif_file(fname, cryst)
        else:
            raise ValueError("Don't know how to write format for '{}'!".format(fname))

    def to_pdb_file(self, fname, cryst=None):
        if cryst != None:
            self.crystal_symmetry = cryst
        return write_pdb(fname, self)

    def to_mmcif_file(self, fname, cryst=None):
        if cryst != None:
            self.crystal_symmetry = cryst
        return write_mmcif(fname, self)

    def translate(self, translation):
        """Translate atoms"""
        self.coor += translation
