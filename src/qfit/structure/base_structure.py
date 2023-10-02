from abc import ABC
import logging
from collections.abc import Iterable
from operator import eq, gt, ge, le, lt

import numpy as np
from molmass.elements import ELEMENTS

from .pdbfile import write_pdb, write_mmcif, ANISOU_FIELDS
from .selector import AtomSelector
from .math import adp_ellipsoid_axes

logger = logging.getLogger(__name__)


class BaseStructure(ABC):
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
    _selector = AtomSelector()
    _COMPARISON_DICT = {"==": eq, "!=": eq, ">": gt, ">=": ge, "<=": le, "<": lt}

    def __init__(self, data, selection=None, parent=None, **kwargs):
        self._data = data
        if selection is not None and not isinstance(selection, np.ndarray):
            selection = np.array(selection)
        self._selection = selection
        self.parent = parent
        # Save extra kwargs for general extraction and duplication methods.
        self._kwargs = kwargs
        self.link_data = kwargs.get("link_data", {})
        self.crystal_symmetry = kwargs.get("crystal_symmetry", None)
        self.unit_cell = kwargs.get("unit_cell", None)
        self.file_format = kwargs.get("file_format", None)
        self.total_length = data["coor"].shape[0]
        if selection is None:
            self.natoms = self.total_length
        else:
            self.natoms = self._selection.size

    def _get_element_property(self, ptype):
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

    @property
    def selection(self):
        return self._selection

    def get_selected_data(self, keys=None):
        if self._selection is None:
            return self._data
        return {k: v[self._selection]
                for k, v in self._data.items()
                if (keys is None or k in keys)}

    def get_selected_structure(self, selection):
        """
        Return a copy of the same type, containing only the selected atoms.
        'selection' should be a numpy boolean or uint array.
        """
        data = {}
        for attr in self._data:
            array1 = self.get_array(attr)
            data[attr] = array1[selection]
        return self.__class__(data,
                              parent=None,
                              selection=None,
                              **self._kwargs)

    def with_symmetry(self, crystal_symmetry):
        kwargs = dict(self._kwargs)
        kwargs["crystal_symmetry"] = crystal_symmetry
        return self.__class__(self._data, **kwargs)

    def combine(self, other):
        """
        Combines two structures into one and returns the result (of the same
        type as 'self').
        """
        data = {}
        for attr in self._data:
            array1 = self.get_array(attr)
            array2 = other.get_array(attr)
            combined = np.concatenate((array1, array2))
            data[attr] = combined
        return self.__class__(data, crystal_symmetry=self.crystal_symmetry)

    @property
    def covalent_radius(self):
        return self._get_element_property("covrad")

    @property
    def vdw_radius(self):
        return self._get_element_property("vdwrad")

    def _copy(self, class_def=None):
        if class_def is None:
            class_def = self.__class__
        data = {}
        for attr in self._data:
            data[attr] = self.get_array(attr).copy()
        return class_def(data, parent=None, selection=None, **self._kwargs)

    def copy(self):
        return self._copy()

    def copy_as(self, class_def):
        return self._copy(class_def)

    def extract(self, *args):
        if not isinstance(args[0], str):
            selection = args[0]
        else:
            selection = self.select(*args)
        return self.__class__(
            self._data, selection=selection, parent=self, **self._kwargs
        )

    def rotate(self, R):
        """Rotate structure in place"""
        coor = np.dot(self.coor, R.T)  # pylint: disable=access-member-before-definition
        self.coor = coor

    # FIXME this is not used anywhere, but the logic to handle ring flips
    # is something we should incorporate in the overall program
    def rmsd(self, structure):
        coor1 = self.coor
        coor2 = structure.coor
        if coor1.shape != coor2.shape:
            raise ValueError("Coordinate shapes are not equivalent")
        resnames = set(self.resn)
        if "TYR" in resnames:
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
        elif "PHE" in resnames:
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
        data = self.get_array(attr)
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

    def get_array(self, key):
        if self._selection is None:
            return self._data[key].copy()
        return self._data[key][self._selection]

    def _set_array(self, key, value, selection=None):
        if selection is None:
            selection = self._selection
        if selection is None:
            self._data[key][:] = value
        else:
            self._data[key][selection] = value

    def set_occupancies(self, values, selection=None):
        self._set_array("q", values, selection)

    def set_xyz(self, values, selection=None):
        self._set_array("coor", values, selection)

    def get_xyz(self, selection):
        return self._data["coor"][selection]

    def set_altloc(self, values, selection=None):
        self._set_array("altloc", values, selection)

    def get_name(self, selection):
        return self._data["name"][selection]

    @property
    def record(self):
        return self.get_array("record")

    @property
    def name(self):
        return self.get_array("name")

    @name.setter
    def name(self, value):
        self._set_array("name", value)

    @property
    def b(self):
        return self.get_array("b")

    @b.setter
    def b(self, value):
        self._set_array("b", value)

    @property
    def q(self):
        return self.get_array("q")

    @q.setter
    def q(self, value):
        self._set_array("q", value)

    @property
    def coor(self):
        return self.get_array("coor")

    @coor.setter
    def coor(self, value):
        self._set_array("coor", value)

    @property
    def resn(self):
        return self.get_array("resn")

    @property
    def resi(self):
        return self.get_array("resi")

    @property
    def icode(self):
        return self.get_array("icode")

    @property
    def e(self):
        return self.get_array("e")

    @property
    def charge(self):
        return self.get_array("charge")

    @property
    def chain(self):
        return self.get_array("chain")

    @property
    def altloc(self):
        return self.get_array("altloc")

    @altloc.setter
    def altloc(self, value):
        self._set_array("altloc", value)

    @property
    def atomid(self):
        return self.get_array("atomid")

    @property
    def active(self):
        return self.get_array("active")

    @active.setter
    def active(self, value):
        self._set_array("active", value)

    def clear_active(self):
        self._data["active"][:] = False

    def set_active(self, selection=None, value=True):
        if selection is None:
            self._data["active"][:] = value
        else:
            self._data["active"][selection] = value

    def extract_anisous(self):
        data = self.get_selected_data(keys=set(ANISOU_FIELDS))
        return [(
            (data["u00"][x], data["u01"][x], data["u02"][x]),
            (data["u01"][x], data["u11"][x], data["u12"][x]),
            (data["u02"][x], data["u12"][x], data["u22"][x]),
        ) for x in range(self.natoms)]

    def get_adp_ellipsoid_axes(self):
        """
        Identify sampling directions based on the ellipsoid axes of the
        refined ADPs.  Should only be used on single-atom structures.
        """
        try:
            u_matrix = self.extract_anisous()[0]
            directions = adp_ellipsoid_axes(u_matrix)
            logger.debug(f"[_sample_backbone] u_matrix = {u_matrix}")
            logger.debug(f"[_sample_backbone] directions = {directions}")
            return directions
        except KeyError:
            logger.info(
                f"Got KeyError for directions at Cβ. Treating as isotropic B, using x,y,z vectors."
            )
            # TODO: Probably choose to put one of these as Cβ-Cα, C-N, and then (Cβ-Cα × C-N)
            return np.identity(3)

    def _add_atom(self, name, element, coor, **kwargs):
        """
        Append a single atom to the structure.  This is only used by the
        RotamerResidue subclass, when building out an incomplete residue,
        but the implementation lives here now to hide the details of the
        internal data
        """
        index = self._selection[-1]
        if index < len(self._data["record"]):
            index = len(self._data["record"]) - 1
        for attr in self._data:
            if attr == "e":
                self._data[attr] = np.append(self._data[attr], element)
            elif attr == "atomid":
                self._data[attr] = np.append(self._data[attr], index + 1)
            elif attr == "name":
                self._data["name"] = np.append(self._data["name"], name)
            elif attr == "coor":
                self._data["coor"] = np.append(self._data["coor"],
                                               np.expand_dims(coor, axis=0),
                                               axis=0)
            elif attr in kwargs:
                self._data[attr] = kwargs[attr]
            else:
                self._data[attr] = np.append(self._data[attr],
                                             self.get_array(attr)[-1])
        self._selection = np.append(self._selection, index + 1).astype(np.uint64)
        self.natoms += 1


class BaseMonomer(BaseStructure):
    """
    Base class for any single 'residue', of any chemical type (amino acid,
    nucleic acid, or ligand).  May be multi-conformer as long as the
    composition is homogenous.
    """

    @property
    def resname(self):
        return self.resn[0]
