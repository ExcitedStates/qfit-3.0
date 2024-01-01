from abc import ABC
import logging
from collections.abc import Iterable
from operator import eq, gt, ge, le, lt

from scitbx.array_family import flex
import numpy as np
from molmass.elements import ELEMENTS

from .pdbfile import write_pdb, write_mmcif, load_combined_atoms, get_pdb_hierarchy
from .math import adp_ellipsoid_axes
from qfit.xtal import UnitCell

logger = logging.getLogger(__name__)

ANISOU_SCALE = 10000

def _as_size_t(selection):
    """
    Make sure an atom selection has the type scitbx.array_family.flex.size_t
    """
    if isinstance(selection, flex.size_t):
        return selection
    elif isinstance(selection, flex.bool):
        return selection.iselection()
    elif isinstance(selection, np.ndarray) and selection.dtype == bool:
        return flex.bool(selection).iselection()
    return flex.size_t(selection)


class BaseStructure(ABC):
    _COMPARISON_DICT = {"==": eq, "!=": eq, ">": gt, ">=": ge, "<=": le, "<": lt}

    def __init__(self,
                 atoms,
                 pdb_hierarchy,
                 selection=None,
                 parent=None,
                 hierarchy_objects=None,
                 **kwargs):
        self._atoms = atoms
        self._pdb_hierarchy = pdb_hierarchy
        self._hierarchy_objects = hierarchy_objects
        self._selection_cache = pdb_hierarchy.atom_selection_cache()
        if selection is not None:
            selection = _as_size_t(selection)
        self._selection = selection
        self.parent = parent
        # Save extra kwargs for general extraction and duplication methods.
        self._kwargs = kwargs
        self.link_data = kwargs.get("link_data", {})
        self.crystal_symmetry = kwargs.get("crystal_symmetry", None)
        self.unit_cell = kwargs.get("unit_cell", None)
        self.file_format = kwargs.get("file_format", None)
        self.total_length = atoms.size()
        if selection is None:
            self.natoms = self.total_length
        else:
            self.natoms = self._selection.size()
        self._active_flag = np.ones(self.total_length, dtype=bool)
        if self.crystal_symmetry:
            spg = self.crystal_symmetry.space_group_info()
            uc = self.crystal_symmetry.unit_cell()
            values = list(uc.parameters()) + [spg.type().lookup_symbol()]
            self.unit_cell = UnitCell(*values)

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

    def get_selected_atoms(self, selection=None):
        if selection is None:
            selection = self._selection
        if selection is None:
            return self._atoms
        else:
            return self._atoms.select(_as_size_t(selection))

    def get_selected_hierarchy(self, selection=None):
        if selection is None:
            selection = self._selection
        if selection is None:
            return self._pdb_hierarchy
        else:
            return self._pdb_hierarchy.select(_as_size_t(selection))

    @property
    def atoms(self):
        return self.get_selected_atoms()

    def get_selected_structure(self, selection):
        """
        Return a copy of the same type, containing only the selected atoms.
        'selection' should be a numpy boolean or uint array.
        """
        new_pdb = load_combined_atoms(self.get_selected_atoms(selection))
        hierarchy = get_pdb_hierarchy(new_pdb)
        return self.__class__(hierarchy.atoms(),
                              hierarchy,
                              **self._kwargs)

    def get_atom_selection(self, selection_string):
        selection = self._selection_cache.selection(selection_string)
        if self._selection is not None:
            base_sel = flex.bool(selection.size(), False)
            base_sel.set_selected(self._selection, True)
            selection = selection & base_sel
        return selection

    def with_symmetry(self, crystal_symmetry):
        kwargs = dict(self._kwargs)
        kwargs["crystal_symmetry"] = crystal_symmetry
        return self.__class__(self._atoms, self._pdb_hierarchy, **kwargs)

    def combine(self, other):
        """
        Combines two structures into one and returns the result (of the same
        type as 'self').
        """
        pdb_in = load_combined_atoms(self.atoms, other.atoms)
        pdb_hierarchy = get_pdb_hierarchy(pdb_in)
        return self.__class__(pdb_hierarchy.atoms(),
                              pdb_hierarchy,
                              crystal_symmetry=self.crystal_symmetry)

    @property
    def covalent_radius(self):
        return self._get_element_property("covrad")

    @property
    def vdw_radius(self):
        return self._get_element_property("vdwrad")

    def _copy(self, class_def=None):
        if class_def is None:
            class_def = self.__class__
        new_hierarchy = self.get_selected_hierarchy().deep_copy()
        atoms = new_hierarchy.atoms()
        return class_def(atoms, new_hierarchy, parent=None, selection=None, **self._kwargs)

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
            self._atoms,
            self._pdb_hierarchy,
            selection=selection,
            parent=self,
            hierarchy_objects=self._hierarchy_objects,
            **self._kwargs
        )

    def rotate(self, R):
        """Rotate structure in place"""
        coor = np.dot(self.coor, R.T)  # pylint: disable=access-member-before-definition
        self.coor = coor

    # FIXME this is not used anywhere, but the logic to handle ring flips
    # is something we should incorporate in the overall program
    def rmsd(self, other):
        coor1 = self.coor
        coor2 = other.coor
        if coor1.shape != coor2.shape:
            raise ValueError("Coordinate shapes are not equivalent")
        resnames = set(self.resn)
        if "TYR" in resnames:
            idx_cd1 = other.name.tolist().index("CD1")
            idx_cd2 = other.name.tolist().index("CD2")
            idx_ce1 = other.name.tolist().index("CE1")
            idx_ce2 = other.name.tolist().index("CE2")
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
            idx_cd1 = other.name.tolist().index("CD1")
            idx_cd2 = other.name.tolist().index("CD2")
            idx_ce1 = other.name.tolist().index("CE1")
            idx_ce2 = other.name.tolist().index("CE2")
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
            selection = self.get_atom_selection(string)
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
            selection = self._selection.select(flex.bool(mask))
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

    def to_xray_structure(self, active_only=False):
        xrs = self._pdb_hierarchy.extract_xray_structure(
            crystal_symmetry=self.crystal_symmetry)
        if self._selection is not None:
            xrs = xrs.select(self._selection)
        if active_only:
            xrs = xrs.select(_as_size_t(self.active))
        return xrs

    def translate(self, translation):
        """Translate atoms"""
        self.coor += translation

    def set_occupancies(self, values, selection=None):
        atoms = self.get_selected_atoms(selection)
        if isinstance(values, (int, float)):
            values = flex.double([values for x in range(len(atoms))])
        else:
            values = flex.double(values)
        atoms.set_occ(values)

    def set_xyz(self, values, selection=None):
        self.get_selected_atoms(selection).set_xyz(flex.vec3_double(values))

    def get_xyz(self, selection):
        return self.get_selected_atoms(selection).extract_xyz().as_numpy_array()

    def get_atom_xyz(self, i_seq):
        """Return the coordinates of a single atom as a numpy array"""
        return np.array(self._atoms[i_seq].xyz)

    def set_altloc(self, value, selection=None):
        for atom in self.get_selected_atoms(selection):
            atom.parent().altloc = value

    def get_name(self, selection):
        return np.array([a.name.strip() for a in self.get_selected_atoms(selection)])

    @property
    def name(self):
        return np.array([a.name.strip() for a in self.atoms])

    @name.setter
    def name(self, value):
        for atom in self.atoms:
            atom.name = value

    @property
    def b(self):
        return self.atoms.extract_b().as_numpy_array()

    @b.setter
    def b(self, value):
        if isinstance(value, (int, float)):
            for atom in self.atoms:
                atom.b = value
        else:
            return self.atoms.set_b(flex.double(value))

    @property
    def q(self):
        return self.atoms.extract_occ().as_numpy_array()

    @q.setter
    def q(self, value):
        if isinstance(value, (int, float)):
            for atom in self.atoms:
                atom.occ = value
        else:
            return self.set_occupancies(value)

    @property
    def coor(self):
        return self.atoms.extract_xyz().as_numpy_array()

    @coor.setter
    def coor(self, value):
        return self.atoms.set_xyz(flex.vec3_double(value))

    @property
    def resn(self):
        return np.array([a.parent().resname.strip() for a in self.atoms])

    @property
    def resi(self):
        return np.array([a.parent().parent().resseq_as_int() for a in self.atoms])

    @property
    def icode(self):
        return np.array([a.parent().parent().icode.strip() for a in self.atoms])

    @property
    def e(self):
        return np.array(self.atoms.extract_element().strip())

    @property
    def charge(self):
        return np.array(["" for a in self.atoms])  # FIXME

    @property
    def chain(self):
        return np.array([a.chain().id.strip() for a in self.atoms])

    @property
    def altloc(self):
        return np.array([a.parent().altloc.strip() for a in self.atoms])

    @altloc.setter
    def altloc(self, value):
        if isinstance(value, str):
            value = [value for x in range(self.natoms)]
        for i, atom in enumerate(self.atoms):
            atom.parent().altloc = value[i]

    @property
    def atomid(self):
        return np.array(self.atoms.extract_serial())

    # FIXME this needs to be handled differently
    @property
    def record(self):
        def _rec_type(atom):
            return "HETATM" if atom.hetero else "ATOM"
        return np.array([_rec_type(a) for a in self.atoms])

    @property
    def active(self):
        if self._selection is None:
            return self._active_flag.copy()
        else:
            return self._active_flag[self._selection].copy()

    @active.setter
    def active(self, value):
        self.set_active(selection=self._selection, value=value)

    def clear_active(self):
        self._active_flag[:] = False

    def set_active(self, selection=None, value=True):
        if selection is None:
            self._active_flag[:] = value
        elif isinstance(value, bool):
            self._active_flag[selection] = value
        else:  # assume a list or array
            self._active_flag[selection] = value

    def extract_anisous(self):
        scaled_uij = np.array(self.atoms.extract_uij()) * ANISOU_SCALE
        return np.array([(
            (uij[0], uij[3], uij[4]),
            (uij[3], uij[1], uij[5]),
            (uij[4], uij[5], uij[2])
        ) for uij in scaled_uij])

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


class BaseMonomer(BaseStructure):
    """
    Base class for any single 'residue', of any chemical type (amino acid,
    nucleic acid, or ligand).  May be multi-conformer as long as the
    composition is homogenous.
    """

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self._type = kwds.get("monomer_type", None)

    @property
    def type(self):
        return self._type

    @property
    def resname(self):
        return self.atoms[0].parent().resname.strip()

    def _is_next_polymer_residue(self, other):
        return False

    def is_next_residue(self, other):
        """Return True if 'other' is the next residue in a polymer chain."""
        if self.type == other.type:
            return self._is_next_polymer_residue(other)
        return False
