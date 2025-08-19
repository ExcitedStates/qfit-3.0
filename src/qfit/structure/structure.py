from collections import defaultdict
import itertools
import os

from scitbx.array_family import flex
import numpy as np

from .base_structure import BaseStructure, BaseMonomer
from .pdbfile import (read_pdb_or_mmcif,
                      get_pdb_hierarchy,
                      load_atoms_from_labels)
from .ligand import Ligand
from .residue import Residue, RotamerResidue, residue_type
from .rotamers import ROTAMERS
from qfit.utils.normalize_to_precision import normalize_to_precision


class Structure(BaseStructure):
    """Class with access to underlying PDB hierarchy."""

    def __init__(self, pdb_hierarchy, **kwargs):
        self._dist2_matrix = None
        self._chains = []
        super().__init__(pdb_hierarchy, **kwargs)

    @staticmethod
    def fromfile(fname):
        pdb_inp, link_data, file_format = read_pdb_or_mmcif(fname)
        pdb_hierarchy = get_pdb_hierarchy(pdb_inp)
        return Structure(
            pdb_hierarchy=pdb_hierarchy,
            link_data=link_data,
            crystal_symmetry=pdb_inp.crystal_symmetry(),
            file_format=file_format,
        )

    @staticmethod
    def fromstructurelike(structure_like):
        return structure_like.copy_as(Structure)

    def __getitem__(self, key):
        if not self._chains:
            self._build_hierarchy()
        if isinstance(key, int):
            nchains = len(self._chains)
            if key < 0:
                key = key + nchains
            if key >= nchains or key < 0:
                raise IndexError("Selection out of range.")
            else:
                return self._chains[key]
        elif isinstance(key, str):
            for chain in self._chains:
                if key == chain.id:
                    return chain
            raise KeyError
        else:
            raise TypeError

    def __repr__(self):
        if not self._chains:
            self._build_hierarchy()
        return f"Structure: {self.natoms} atoms"

    @property
    def chains(self):
        if not self._chains:
            self._build_hierarchy()
        return self._chains

    @property
    def residue_groups(self):
        for chain in self.chains:
            for rg in chain.residue_groups:
                yield rg

    @property
    def residues(self):
        for chain in self.chains:
            for conformer in chain.conformers:
                for residue in conformer.residues:
                    yield residue

    @property
    def single_conformer_residues(self):
        residues_d = {}
        for chain in self.chains:
            if chain.chain[0] not in residues_d:
                residues_d[chain.chain[0]] = {}
            for conformer in chain.conformers:
                for residue in conformer.residues:
                    if residue.resi[0] not in residues_d[chain.chain[0]]:
                        yield residue
                        residues_d[chain.chain[0]][residue.resi[0]] = 1

    @property
    def segments(self):
        for chain in self.chains:
            for conformer in chain.conformers:
                for segment in conformer.segments:
                    yield segment

    def _build_hierarchy(self):
        self._chains = []
        self._atoms.reset_i_seq()
        hierarchy = self.get_selected_hierarchy()
        # FIXME this extra bit of logic is to preserve the pre-CCTBX behavior,
        # where qfit groups all atoms by chain ID, so that ligands and other
        # heteroatoms become part of the same _Chain object as the protein
        # they are associated with.  since qfit actually handles protein and
        # non-protein atoms separately, the internal grouping should be changed
        # to reflect that as well
        chains_d = defaultdict(list)
        for iotbx_chain in hierarchy.only_model().chains():
            chains_d[iotbx_chain.id].append(iotbx_chain)
        sel_cache = hierarchy.atom_selection_cache()
        for chain_id, chains in chains_d.items():
            selection = sel_cache.selection(f"chain '{chain_id}'").iselection()
            chain = _Chain(self._pdb_hierarchy,
                           selection=selection,
                           parent=self,
                           hierarchy_objects=chains,
                           atoms=self._atoms,
                           crystal_symmetry=self.crystal_symmetry)
            self._chains.append(chain)

    def split_models(self):
        """
        Split a multi-model Structure into a list of single-model Structures
        using the existing iotbx hierarchy.
        """
        hierarchy = self._pdb_hierarchy
        model_count = hierarchy.models_size()

        if model_count <= 1:
            return [self]

        structures = []
        for model in hierarchy.models():
            single_hierarchy = iotbx.pdb.hierarchy.root()
            single_hierarchy.append_model(model.detached_copy())

            # Create a Structure from this model's hierarchy
            new_struct = Structure(single_hierarchy, **self._kwargs)
            structures.append(new_struct)

        return structures
     
   def collapse_backbone(self, resid, chainid):
        """
        Collapse the backbone atoms of a given residue and return a new copy
        """
        # determine altloc to keep
        sel_str = f"resi {resid} and chain {chainid}"
        conformers = self.extract(sel_str)
        altlocs = sorted(list(set(conformers.altloc)))
        altloc_keep = altlocs[0]
        altlocs_remove = altlocs[1:]
        if len(altlocs_remove) == 0:
            return self.copy()
        altlocs_sel_str = " or ".join([f"altloc {a}" for a in altlocs_remove])
        remove_sel_str = f"{sel_str} and (name CA or name C or name O or name N or name H or name HA) and ({altlocs_sel_str})"
        remove_mask = self._get_base_atom_selection(remove_sel_str)
        keep_sel_str = f"{sel_str} and (name CA or name C or name O or name N) and altloc '{altloc_keep}'"
        keep_mask = self._get_base_atom_selection(keep_sel_str)
        keep_sel = keep_mask.select(~remove_mask).iselection()
        atoms = self.get_selected_atoms((~remove_mask).iselection())
        # XXX This should ideally manipulate the hierarchy directly instead of
        # using the atom label objects
        atom_labels = [a.fetch_labels() for a in atoms]
        for i_seq in keep_sel:
            atom_labels[i_seq].set_occ(1.0)
            atom_labels[i_seq].altloc = ""
        new_hierarchy = get_pdb_hierarchy(load_atoms_from_labels(atom_labels))
        return Structure(new_hierarchy, selection=None, **self._kwargs)

    def set_backbone_occ(self):
        """
        Return a copy of the structure with the "backbone" occupancy set to 0
        and the occupancy of other atoms to set 1.0
        """
        BACKBONE_ATOMS = ["CA", "C", "N", "O", "H", "HA"]
        BACKBONE_ATOMS_GLY = ["CA", "C", "N", "H", "HA"]
        if self.resn[0] == "GLY":
            atom_names = BACKBONE_ATOMS_GLY
        else:
            atom_names = BACKBONE_ATOMS
        backbone_sel_str = " or ".join([f"name {a}" for a in atom_names])
        mask_backbone = self.get_atom_selection(backbone_sel_str)
        mask_nonbackbone = ~mask_backbone
        if self._selection is not None:
            mask_backbone = mask_backbone.select(self._selection)
            mask_nonbackbone = mask_nonbackbone.select(self._selection)
        new_structure = self.copy()
        new_structure.set_occupancies(0.0, mask_backbone.iselection())
        new_structure.set_occupancies(1.0, mask_nonbackbone.iselection())
        return new_structure

    def reorder(self):
        """
        Sort the atoms within each residue group, first by altloc, then by
        the atom ordering defined in the rotamer library, with hydrogens at
        the end of the list.
        Returns a new copy of the structure with the reordered atoms.
        """
        assert self._selection is None, "Can't call reorder() on a sub-structure"
        ordering = []
        for chain in self.chains:
            for rg in chain.residue_groups:
                if rg.resn[0] in ROTAMERS:
                    ordering.append(_get_residue_group_atom_order(rg))
                    continue
                for ag in rg.atom_groups:
                    ordering.append(ag.selection)
        return self.get_selected_structure(np.concatenate(ordering))

    def normalize_occupancy(self):
        """This function will scale the occupancy of protein residues to make the sum(occ) equal to 1 for all.
        The goal of this function is to determine if the sum(occ) of each residue coming out of qFit residue or segement is < 1 (as it can be due to QP),
        and scale each alt conf occupancy for the residue to sum to one.
        To accomplish this, if the sum(occ) is less than 1, then we will divide each conformer occupancy by the sum(occ).

        Returns a copy of the structure with normalized occupancies.
        """
        assert self._selection is None
        multiconformer = self.copy()
        for chain in multiconformer:
            for residue in chain:
                altlocs = list(set(residue.altloc))
                mask = None
                base_sel_str = f"resi {residue.resi[0]} and chain {residue.chain[0]}"
                if len(altlocs) == 1:  # confirm occupancy = 1
                    mask = self.get_atom_selection(base_sel_str)
                    multiconformer.set_altloc("", mask.iselection())
                    multiconformer.set_occupancies(1.0, mask.iselection())
                else:
                    new_occ = []
                    if "" in altlocs and len(altlocs) > 1:
                        sel_str = f"{base_sel_str} and altloc ''"
                        mask = self.get_atom_selection(sel_str)
                        multiconformer.set_altloc("", mask.iselection())
                        multiconformer.set_occupancies(1.0, mask.iselection())
                        altlocs.remove("")
                    conformers = []
                    for altloc in altlocs:
                        sel_str = f"{base_sel_str} and altloc {altloc}"
                        conformers.append(self.extract(sel_str))
                    alt_sum = 0
                    for i in range(0, len(conformers)):
                        alt_sum += np.round(conformers[i].q[0], 2)
                        new_occ = np.append(new_occ, (np.round(conformers[i].q[0], 2)))
                    if alt_sum != 1.0:  # we need to normalize
                        new_occ = []
                        for i in range(0, len(altlocs)):
                            new_occ = np.append(
                                new_occ, ((np.round(conformers[i].q[0], 2)) / alt_sum)
                            )
                        new_occ = normalize_to_precision(
                            np.array(new_occ), 2
                        )  # deal with imprecision
                    for i in range(0, len(new_occ)):
                        sel_str = f"{base_sel_str} and altloc {altlocs[i]}"
                        mask = self.get_atom_selection(sel_str)
                        multiconformer.set_occupancies(new_occ[i], mask.iselection())
        return multiconformer

    def _remove_conformer(self, resi, chain, altloc_keep, altloc_remove):
        keep_sel_str = f"resi {resi} and chain {chain} and altloc {altloc_keep}"
        remove_sel_str = f"resi {resi} and chain {chain} and altloc {altloc_remove}"
        keep_mask = self.get_atom_selection(keep_sel_str)
        remove_mask = self.get_atom_selection(remove_sel_str)
        new_structure = self.copy()
        occ = new_structure.q
        new_structure.set_occupancies(occ[keep_mask] + occ[remove_mask],
                                      keep_mask.iselection())
        final_sel = (~remove_mask).iselection()
        return new_structure.get_selected_structure(final_sel)

    def remove_identical_conformers(self, rmsd_cutoff=0.01):
        multiconformer = self.copy()
        for chain in multiconformer:
            for residue in chain:
                altlocs = list(set(residue.altloc))
                try:
                    altlocs.remove("")
                except ValueError:
                    pass
                sel_str = (
                    f"resi {residue.resi[0]} and chain {residue.chain[0]} and altloc "
                )
                conformers = [multiconformer.extract(sel_str + x) for x in altlocs]
                if len(set(altlocs)) == 1:
                    continue
                else:
                    removed_conformers = []  # list of all conformer that are removed
                    for conf_a, conf_b in itertools.combinations(conformers, 2):
                        if conf_a.altloc[0] in removed_conformers:
                            continue  # we have already removed this conformer
                        else:
                            rmsd = np.sqrt(np.mean((conf_a.coor - conf_b.coor) ** 2))
                            if rmsd > rmsd_cutoff:
                                continue
                            else:
                                multiconformer = multiconformer._remove_conformer(  # pylint: disable=protected-access
                                    residue.resi[0],
                                    residue.chain[0],
                                    conf_a.altloc[0],
                                    conf_b.altloc[0],
                                )
                                removed_conformers.append(conf_b.altloc[0])
        return multiconformer

    def _get_non_hetero_structure(self):
        atom_sel = self._get_base_atom_selection("not hetero")
        if self._selection is not None:
            sel_mask = flex.bool(atom_sel.size(), False)
            sel_mask.set_selected(self._selection, True)
            atom_sel = atom_sel & sel_mask
        return self.get_selected_structure(atom_sel.iselection())

    def _n_residue_conformers(self):
        total_altlocs = 0
        for rg in self._get_non_hetero_structure().residue_groups:
            altlocs = set(rg.altloc)
            if "" in altlocs and len(altlocs) > 1:
                altlocs.remove("")
            total_altlocs += len(altlocs)
        return total_altlocs

    def n_residues(self):
        """Number of residues in the structure (exclude heteroatoms).

        Required because Structure.residue_groups is a generator.

        Returns:
            int
        """
        # XXX this is very inefficient, but since this method is called
        # infrequently it's probably not worth the effort to optimize
        residue_groups = self._get_non_hetero_structure().residue_groups
        return sum(1 for _ in residue_groups)

    def average_conformers(self):
        """Average number of conformers over the structure (exclude heteroatoms).

        Returns:
            float
        """
        return self._n_residue_conformers() / self.n_residues()

    def _init_clash_detection(self):
        # Setup the condensed distance based arrays for clash detection and
        # fill them
        self._ndistances = self.natoms * (self.natoms - 1) // 2
        self._clash_mask = np.ones(self._ndistances, bool)
        self._clash_radius2 = np.zeros(self._ndistances, float)
        radii = self.covalent_radius
        offset = self.natoms * (self.natoms - 1) // 2
        for i in range(self.natoms - 1):
            rotamers = ROTAMERS[self.resn[i]]
            bonds = rotamers["bonds"]
            natoms_left = self.natoms - i
            starting_index = offset - natoms_left * (natoms_left - 1) // 2 - i - 1
            name1 = self.name[i]
            covrad1 = radii[i]
            for j in range(i + 1, self.natoms):
                bond1 = [name1, self.name[j]]
                bond2 = bond1[::-1]
                covrad2 = radii[j]
                index = starting_index + j
                self._clash_radius2[index] = covrad1 + covrad2 + 0.4
                if self.resi[i] == self.resi[j] and self.chain[i] == self.chain[j]:
                    if bond1 in bonds or bond2 in bonds:
                        self._clash_mask[index] = False
                elif (
                    self.resi[i] + 1 == self.resi[j] and self.chain[i] == self.chain[j]
                ):
                    if name1 == "C" and self.name[j] == "N":
                        self._clash_mask[index] = False
                elif self.e[i] == "S" and self.e[j] == "S":
                    self._clash_mask[index] = False
        self._clash_radius2 *= self._clash_radius2
        self._clashing = np.zeros(self._ndistances, bool)
        self._dist2_matrix = np.empty(self._ndistances, float)

        # All atoms are active from the start
        self.set_active()
        self._active_mask = np.ones(self._ndistances, bool)

    def clashes(self):
        """Checks if there are any internal clashes.
        Deactivated atoms are not taken into account.
        """

        if self._dist2_matrix is None:
            self._init_clash_detection()
        dm = self._dist2_matrix
        coor = self.coor
        dot = np.dot
        k = 0
        for i in range(self.natoms - 1):
            u = coor[i]
            for j in range(i + 1, self.natoms):
                u_v = u - coor[j]
                dm[k] = dot(u_v, u_v)
                k += 1
        np.less_equal(dm, self._clash_radius2, self._clashing)
        self._clashing &= self._clash_mask
        self._clashing &= self._active_mask
        nclashes = self._clashing.sum()
        return nclashes

    def extract_neighbors(self, residue, distance=5.0):
        # Remove the residue of interest from the structure:
        sel_str = f"resi {residue.resi[0]} and chain {residue.chain[0]}"
        sel_str = f"not ({sel_str})"

        # Identify all atoms within distance of the residue of interest
        neighbors = self.extract(sel_str)
        mask = neighbors.coor[:, 1] < -np.inf
        for coor in residue.coor:
            diffs = neighbors.coor - np.array(coor)
            dists = np.linalg.norm(diffs, axis=1)
            mask = np.logical_or(mask, dists < distance)
        sel = flex.bool(mask).iselection()
        return neighbors.copy().get_selected_structure(sel).with_symmetry(
            self.crystal_symmetry)


class _Chain(BaseStructure):
    def __init__(self, pdb_hierarchy, **kwargs):
        super().__init__(pdb_hierarchy, **kwargs)
        self._residue_groups = []
        self._conformers = []
        self.conformer_ids = np.unique(self.altloc)

    @property
    def id(self):
        return self._hierarchy_objects[0].id

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            if not self._residue_groups:
                self._build_hierarchy()
        nresidues = len(self._residue_groups)
        if isinstance(key, int):
            if key < 0:
                key += nresidues
            if key >= nresidues or key < 0:
                raise IndexError
            else:
                return self._residue_groups[key]
        elif isinstance(key, slice):
            start = key.start
            end = key.end
            if start < 0:
                start += nresidues
            if end < 0:
                end += nresidues
            return self._residue_groups[start:end]
        elif isinstance(key, str):
            if not self._conformers:
                self._build_conformers()
            for conformer in self._conformers:
                if conformer.id == key:
                    return conformer
            raise KeyError
        else:
            raise TypeError

    def __repr__(self):
        return "Chain: {chainid}".format(chainid=self.id)

    @property
    def conformers(self):
        if not self._conformers:
            self._build_conformers()
        return self._conformers

    @property
    def residue_groups(self):
        if not self._residue_groups:
            self._build_hierarchy()
        return self._residue_groups

    def _build_hierarchy(self):
        for iotbx_chain in self._hierarchy_objects:
            for iotbx_residue_group in iotbx_chain.residue_groups():
                selection = iotbx_residue_group.atoms().extract_i_seq()
                residue_group = ResidueGroup(
                    self._pdb_hierarchy,
                    selection=selection,
                    parent=self,
                    hierarchy_objects=(iotbx_residue_group,),
                    atoms=self._atoms,
                    crystal_symmetry=self.crystal_symmetry
                )
                self._residue_groups.append(residue_group)

    def _build_conformers(self):
        self._conformers = []
        # FIXME like the _Chain extraction, we need some extra logic to
        # reproduce the pre-CCTBX behavior.  in the future we should just
        # use IOTBX objects directly and keep non-protein atoms separate
        confs_d = defaultdict(list)
        for iotbx_chain in self._hierarchy_objects:
            for iotbx_conformer in iotbx_chain.conformers():
                confs_d[iotbx_conformer.altloc.strip()].append(iotbx_conformer)
        blank_altconf = []
        if len(confs_d) > 1 and "" in confs_d:
            blank_altconf = confs_d[""]
        for altloc, conformers in confs_d.items():
            if altloc == "" and blank_altconf:
                continue
            conformers = conformers + blank_altconf
            selection = conformers[0].atoms().extract_i_seq()
            for conf in conformers[1:]:
                selection.extend(conf.atoms().extract_i_seq())
            conformer = _Conformer(
                self._pdb_hierarchy,
                selection=selection,
                parent=self,
                hierarchy_objects=conformers,
                atoms=self._atoms,
                crystal_symmetry=self.crystal_symmetry
            )
            self._conformers.append(conformer)


class ResidueGroup(BaseMonomer):

    """Guarantees a group with similar resi and icode."""

    def __init__(self, pdb_hierarchy, **kwargs):
        super().__init__(pdb_hierarchy, **kwargs)
        self._iotbx_residue_group = self._hierarchy_objects[0]
        self._atom_groups = []

    @property
    def id(self):
        return (self._iotbx_residue_group.resseq_as_int(),
                self._iotbx_residue_group.icode.strip())

    @property
    def atom_groups(self):
        if not self._atom_groups:
            self._build_hierarchy()
        return self._atom_groups

    def _build_hierarchy(self):
        # An atom group is a collection of entries that have a unique
        # chain, resi, icode, resn and altloc
        for iotbx_atom_group in self._iotbx_residue_group.atom_groups():
            selection = iotbx_atom_group.atoms().extract_i_seq()
            atom_group = _AtomGroup(
                self._pdb_hierarchy,
                selection=selection,
                parent=self,
                hierarchy_objects=(iotbx_atom_group,),
                atoms=self._atoms,
                crystal_symmetry=self.crystal_symmetry
            )
            self._atom_groups.append(atom_group)

    def __repr__(self):
        resi, icode = self.id
        string = "ResidueGroup: resi {}".format(resi)
        if icode:
            string += ":{}".format(icode)
        return string


class _AtomGroup(BaseStructure):
    def __init__(self, pdb_hierarchy, **kwargs):
        super().__init__(pdb_hierarchy, **kwargs)
        self._iotbx_atom_group = self._hierarchy_objects[0]

    @property
    def id(self):
        return (self._iotbx_atom_group.resname, self._iotbx_atom_group.altloc)

    def __repr__(self):
        string = "AtomGroup: {} {}".format(*self.id)
        return string


class _Conformer(BaseStructure):

    """Guarantees a single consistent conformer."""

    def __init__(self, pdb_hierarchy, **kwargs):
        super().__init__(pdb_hierarchy, **kwargs)
        self._residues = []
        self._segments = []

    @property
    def id(self):
        return self._hierarchy_objects[0].altloc

    def __setstate__(self, d):
        super().__setstate__(d)
        self._hierarchy_objects = None

    def __getitem__(self, arg):
        if not self._residues:
            self._build_residues()
        key = None
        if isinstance(arg, int):
            key = (arg, "")
        elif isinstance(arg, tuple):
            key = arg
        for residue in self._residues:
            if residue.id == key:
                return residue
        raise ValueError("Residue id not found.")

    def __repr__(self):
        return "Conformer {}".format(self.id)

    @property
    def ligands(self):
        if not self._residues:
            self._build_residues()
        _ligands = [l for l in self._residues if isinstance(l, Ligand) and l.natoms > 3]
        return _ligands

    @property
    def residues(self):
        if not self._residues:
            self._build_residues()
        return self._residues

    @property
    def segments(self):
        if not self._segments:
            self._build_segments()
        return self._segments

    def _build_residues(self):
        for iotbx_conformer in self._hierarchy_objects:
            for iotbx_residue in iotbx_conformer.residues():
                selection = iotbx_residue.atoms().extract_i_seq()
                residue = self.extract(selection)
                rtype = residue_type(residue)
                if rtype == "rotamer-residue":
                    monomer_class = RotamerResidue
                elif rtype in ["aa-residue", "residue"]:
                    monomer_class = Residue
                elif rtype == "ligand":
                    monomer_class = Ligand
                else:
                    continue
                residue = monomer_class(
                    self._pdb_hierarchy,
                    selection=selection,
                    parent=self,
                    atoms=self._atoms,
                    resi=iotbx_residue.resseq_as_int(),
                    icode=iotbx_residue.icode.strip(),
                    monomer_type=rtype,
                    crystal_symmetry=self.crystal_symmetry
                )
                self._residues.append(residue)

    def _build_segments(self):
        if not self._residues:
            self._build_residues()

        segments = []
        segment = []
        for res in self._residues:
            if not segment:
                segment.append(res)
            else:
                prev = segment[-1]
                if prev.is_next_residue(res):
                    segment.append(res)
                else:
                    segments.append(segment)
                    segment = [res]
        segments.append(segment)

        for segment in segments:
            if len(segment) > 1:
                selections = [residue.selection for residue in segment]
                selection = np.concatenate(selections)
                segment = Segment(
                    self._pdb_hierarchy,
                    selection=selection,
                    parent=self,
                    atoms=self._atoms,
                    residues=segment,
                    crystal_symmetry=self.crystal_symmetry
                )
                self._segments.append(segment)


class Segment(BaseStructure):

    """Class that guarantees connected residues and allows
    backbone rotations."""

    def __init__(self, pdb_hierarchy, **kwargs):
        super().__init__(pdb_hierarchy, **kwargs)
        self.residues = kwargs["residues"]

    def __contains__(self, residue):
        for res in self.residues:
            if res.id == residue.id and res.altloc[0] == residue.altloc[0]:
                return True
        return False

    def __getitem__(self, arg):
        if isinstance(arg, int):
            return self.residues[arg]
        elif isinstance(arg, slice):
            residues = self.residues[arg]
            selections = []
            for residue in residues:
                selections.append(residue.selection)
            selection = np.concatenate(selections)
            return Segment(
                self._pdb_hierarchy,
                selection=selection,
                parent=self.parent,
                atoms=self._atoms,
                residues=residues,
                crystal_symmetry=self.crystal_symmetry
            )
        else:
            raise TypeError

    def __len__(self):
        return len(self.residues)

    def __repr__(self):
        return "Segment: length {}".format(len(self.residues))

    def find(self, residue_id):
        """
        Return the index in self.residues of the residue with the specified
        ID (either resseq or resseq+icode)
        """
        if isinstance(residue_id, int):
            residue_id = (residue_id, "")
        for i_res, residue in enumerate(self.residues):
            if residue.id == residue_id:
                return i_res
        raise ValueError("Residue is not part of segment.")

    @staticmethod
    def from_structure(structure, selection, residues):
        return Segment(structure._pdb_hierarchy,  # pylint: disable=protected-access
                       selection=selection,
                       parent=structure,
                       atoms=structure._atoms,  # pylint: disable=protected-access
                       residues=residues)

    def get_psi_phi_angles(self):
        """
        Iterate over residues in reverse order and extract the selection,
        axis, and origin for psi and phi backbone angles.
        Used in qfit.samplers.BackboneRotator
        """
        for n, residue in enumerate(self.residues[::-1]):
            psi_sel = residue.select("name", ("O", "OXT"))
            if n > 0:
                psi_sel = np.concatenate((psi_sel, self.residues[-n].selection))
            phi_sel = residue.select("name", ("N", "CA", "O", "OXT", "H", "HA"), "!=")
            N = residue.extract("name", "N")
            CA = residue.extract("name", "CA")
            C = residue.extract("name", "C")
            psi_axis = C.coor[0] - CA.coor[0]
            phi_axis = CA.coor[0] - N.coor[0]
            psi_origin = C.coor[0]
            phi_origin = CA.coor[0]
            yield (psi_sel, psi_axis, psi_origin,
                   phi_sel, phi_axis, phi_origin)


def _get_residue_group_atom_order(rg):
    # There are rare cases in the PDB (for example crambin) that have
    # heterogeneous amino acids in the same residue group
    if len(set(rg.resn)) > 1:
        resi = rg.resi[0]
        icode = rg.icode[0]
        raise RuntimeError(
            f"Heterogeneous residue group detected: {resi}{icode}")
    rotamer = ROTAMERS[rg.resname]
    atom_order = rotamer["atoms"] + rotamer["hydrogens"]
    atomnames = list(rg.name)
    # Check if all atomnames are standard. We don't touch the
    # ordering if it isnt recognized.
    for atom in atomnames:
        if atom not in atom_order:
            break
    # Check if the residue has alternate conformers. If the
    # number of altlocs is equal to the whole residue, sort
    # it after one another. If its just a few, sort it like a
    # zipper.
    altlocs = sorted(list(set(rg.altloc)))
    try:
        altlocs.remove("")
    except ValueError:
        pass
    naltlocs = len(altlocs)
    if naltlocs < 2:
        residue_ordering = []
        for atom in atom_order:
            try:
                index = atomnames.index(atom)
                residue_ordering.append(index)
            except ValueError:
                continue
        return rg.selection.select(flex.size_t(residue_ordering))
    else:
        atoms_per_altloc = []
        zip_atoms = True
        if rg.select("altloc", "").size == 0:
            for altloc in altlocs:
                nsel = rg.select("altloc", altloc).size
                atoms_per_altloc.append(nsel)
            zip_atoms = not all(
                a == atoms_per_altloc[0] for a in atoms_per_altloc
            )
        residue_orderings = []
        for altloc in altlocs:
            altconf = rg.extract("altloc", ("", altloc))
            residue_ordering = []
            atomnames = list(altconf.name)
            for atom in atom_order:
                try:
                    index = atomnames.index(atom)
                    residue_ordering.append(index)
                except ValueError:
                    continue
            residue_ordering = altconf.selection.select(flex.size_t(residue_ordering))
            residue_orderings.append(residue_ordering)
        if (
            zip_atoms
            and len(list(set([len(x) for x in residue_orderings]))) == 1
        ):
            residue_ordering = list(zip(*residue_orderings))
            residue_ordering = np.concatenate(residue_ordering)
        else:
            residue_ordering = np.concatenate(residue_orderings)
        # Now remove duplicates while keeping order
        seen = set()
        return [
            x
            for x in residue_ordering
            if not (x in seen or seen.add(x))
        ]
