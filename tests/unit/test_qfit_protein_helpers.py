"""
Tests for qfit_protein / qfit_segment helpers that were rewritten for speed.

Each new implementation is compared against a verbatim copy of the previous
implementation (the `_reference_*` functions).
"""

import itertools
import os
import tempfile

import numpy as np

from qfit.command_line.qfit_protein import (
    QFitProtein,
    QFitOptions,
    _estimated_sampling_cost,
    _select_residues_to_run,
)
from qfit.qfit import _build_unique_fragments
from qfit.structure import Structure
from qfit.structure.math import calc_rmsd

from .base_test_case import UnitBase


def _reference_backbone_coor_dict(structure, residues):
    """Previous inline construction in QFitProtein._run_qfit_residue_parallel."""
    backbone_coor_dict = {}
    for residue in residues:
        grouped_coords = {}
        grouped_u_matrices = {}
        residue_chain_key = (
            residue.chain[0].replace('"', ""),
            residue.resi[0],
            residue.id[1],
        )
        if residue_chain_key not in backbone_coor_dict:
            backbone_coor_dict[residue_chain_key] = {}
        for altloc in np.unique(
            structure.extract("chain", residue.chain[0], "==")
            .extract("resi", residue.resi[0], "==")
            .extract("icode", residue.id[1], "==")
            .altloc
        ):
            grouped_coords[altloc] = (
                structure.extract("chain", residue.chain[0], "==")
                .extract("resi", residue.resi[0], "==")
                .extract("icode", residue.id[1], "==")
                .extract("altloc", altloc)
                .coor
            )
            atom_name = "O" if residue.resn[0] == "GLY" else "CB"
            atom = (
                structure.extract("chain", residue.chain[0], "==")
                .extract("resi", residue.resi[0], "==")
                .extract("icode", residue.id[1], "==")
                .extract("name", atom_name)
                .extract("altloc", altloc)
            )
            try:
                u_matrix = atom.extract_anisous()[0]
            except Exception:
                u_matrix = None
            grouped_u_matrices[altloc] = u_matrix
        backbone_coor_dict[residue_chain_key]["coords"] = grouped_coords
        backbone_coor_dict[residue_chain_key]["u_matrices"] = grouped_u_matrices
    return backbone_coor_dict


def _reference_unique_fragments(fragment_multiconformer):
    """Previous fragment construction loop in QFitSegment.find_paths."""
    fragments = []
    for fragment_conformer in itertools.product(*fragment_multiconformer):
        fragment = fragment_conformer[0].set_backbone_occ()
        for element in fragment_conformer[1:]:
            fragment = fragment.combine(element.set_backbone_occ())
        combine = True
        for fragment2 in fragments:
            if calc_rmsd(fragment.coor, fragment2.coor) < 0.05:
                combine = False
                break
        if combine:
            fragments.append(fragment)
    return fragments


def _residue_conformers(structure):
    """Per residue group, the list of alternate conformers (as in
    QFitSegment.__call__)."""
    out = []
    for rg in structure.residue_groups:
        altlocs = np.unique(rg.altloc)
        confs = [Structure.fromstructurelike(rg.extract("altloc", (altloc, "")))
                 for altloc in altlocs if not (altloc == "" and len(altlocs) > 1)]
        out.append(confs)
    return out


def _with_anisou(pdb_in, pdb_out, altloc="A"):
    """Copy a PDB, adding ANISOU records for atoms with the given altloc."""
    with open(pdb_in) as f_in, open(pdb_out, "w") as f_out:
        for n, line in enumerate(f_in):
            f_out.write(line)
            if line.startswith("ATOM") and line[16] == altloc:
                u = [100 + n, 200 + n, 300 + n, 10 + n, 20 + n, 30 + n]
                f_out.write("ANISOU" + line[6:28] +
                            "".join(f"{x:7d}" for x in u) + line[70:])


def _assert_same_atoms(a, b):
    assert a.natoms == b.natoms
    assert np.allclose(a.coor, b.coor)
    for attr in ("name", "altloc", "resi", "chain", "resn"):
        assert np.array_equal(getattr(a, attr), getattr(b, attr)), attr
    assert np.allclose(a.q, b.q)
    assert np.allclose(a.b, b.b)


class TestBackboneCoorDict(UnitBase):

    def _compare(self, pdb_file):
        structure = Structure.fromfile(pdb_file).reorder()
        residues = _select_residues_to_run(structure)
        qfit = QFitProtein(structure, None, QFitOptions())
        new = qfit._build_backbone_coor_dict(residues)
        ref = _reference_backbone_coor_dict(structure, residues)
        assert new.keys() == ref.keys()
        for key in ref:
            assert new[key]["coords"].keys() == ref[key]["coords"].keys()
            for altloc in ref[key]["coords"]:
                assert np.allclose(new[key]["coords"][altloc],
                                   ref[key]["coords"][altloc])
                u_new = new[key]["u_matrices"][altloc]
                u_ref = ref[key]["u_matrices"][altloc]
                if u_ref is None:
                    assert u_new is None
                else:
                    assert np.allclose(u_new, u_ref)
        return new

    def test_multiconformer_structure(self):
        result = self._compare(self._get_file_path("SKH_multiconf.pdb"))
        assert any(len(v["coords"]) > 1 for v in result.values())

    def test_anisotropic_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdb = os.path.join(tmp, "aniso.pdb")
            _with_anisou(self._get_file_path("SKH_multiconf.pdb"), pdb)
            result = self._compare(pdb)
        u_matrices = [u for v in result.values()
                      for u in v["u_matrices"].values()]
        assert any(u is not None for u in u_matrices)
        assert any(u is None for u in u_matrices)


class TestCombine(UnitBase):

    def test_combine_many_matches_chained_combine(self):
        structure = Structure.fromfile(
            self._get_file_path("GNNAFNS_multiconf.pdb")).reorder()
        parts = list(structure.residue_groups)
        # As in QFitSegment: an (empty) Structure, combined with residue groups
        start = Structure.fromstructurelike(structure.extract("altloc", "Z"))
        chained = start
        for part in parts:
            chained = chained.combine(part)
        _assert_same_atoms(start.combine(*parts), chained)

    def test_combine_single_argument_unchanged(self):
        structure = Structure.fromfile(
            self._get_file_path("GNNAFNS_multiconf.pdb")).reorder()
        a, b = [Structure.fromstructurelike(rg)
                for rg in list(structure.residue_groups)[:2]]
        combined = a.combine(b)
        assert combined.natoms == a.natoms + b.natoms


class TestUniqueFragments(UnitBase):

    def test_matches_reference(self):
        structure = Structure.fromfile(
            self._get_file_path("SKH_multiconf.pdb")).reorder()
        segment = [c for c in _residue_conformers(structure) if len(c) > 1]
        assert len(segment) == 3
        for length in (1, 2, 3):
            fragment_multiconformer = segment[:length]
            new = _build_unique_fragments(fragment_multiconformer)
            ref = _reference_unique_fragments(fragment_multiconformer)
            assert len(new) == len(ref) > 0
            for a, b in zip(new, ref):
                _assert_same_atoms(a, b)

    def test_duplicates_are_removed(self):
        structure = Structure.fromfile(
            self._get_file_path("SKH_multiconf.pdb")).reorder()
        segment = [c for c in _residue_conformers(structure) if len(c) > 1]
        # Repeat every conformer of the first residue: no new fragments.
        doubled = [segment[0] + segment[0]] + segment[1:]
        assert (len(_build_unique_fragments(doubled)) ==
                len(_build_unique_fragments(segment)))

    def test_input_conformers_not_modified(self):
        structure = Structure.fromfile(
            self._get_file_path("SKH_multiconf.pdb")).reorder()
        segment = [c for c in _residue_conformers(structure) if len(c) > 1]
        q_before = [[c.q.copy() for c in confs] for confs in segment]
        _build_unique_fragments(segment)
        for confs, qs in zip(segment, q_before):
            for c, q in zip(confs, qs):
                assert np.array_equal(c.q, q)


class TestResidueSelection(UnitBase):

    def test_all_residues(self):
        structure = Structure.fromfile(
            self._get_file_path("GNNAFNS_multiconf.pdb")).reorder()
        residues = _select_residues_to_run(structure)
        assert [r.id[0] for r in residues] == [1, 2, 3, 4, 5, 6, 7]

    def test_single_residue(self):
        structure = Structure.fromfile(
            self._get_file_path("GNNAFNS_multiconf.pdb")).reorder()
        chain = structure.chain[0]
        for option in (f"{chain},5", f"{chain},5:", f"{chain},5_"):
            residues = _select_residues_to_run(structure, option)
            assert [r.id for r in residues] == [(5, "")]

    def test_cost_ordering_puts_large_residues_first(self):
        structure = Structure.fromfile(
            self._get_file_path("GNNAFNS_multiconf.pdb")).reorder()
        residues = _select_residues_to_run(structure)
        ordered = sorted(residues, key=_estimated_sampling_cost, reverse=True)
        assert ordered[0].resn[0] == "PHE"
        assert ordered[-1].resn[0] in ("GLY", "ALA")


class TestUniqueCoordinateSet(UnitBase):

    def test_matches_list_based_check(self):
        from qfit.qfit import (_UniqueCoordinateSet, _is_new_conformer_unique,
                               DEFAULT_RMSD_CUTOFF)
        rng = np.random.default_rng(0)
        base = rng.random((300, 12, 3)) * 5
        candidates = []
        for coor in base:
            candidates.append(coor)
            # near-duplicates either side of the 0.01 A cutoff
            candidates.append(coor + 0.005 / np.sqrt(3))
            candidates.append(coor + 0.02 / np.sqrt(3))
        rng.shuffle(candidates)

        reference = []
        decisions_ref = []
        for coor in candidates:
            keep = _is_new_conformer_unique(coor, reference, DEFAULT_RMSD_CUTOFF)
            decisions_ref.append(keep)
            if keep:
                reference.append(coor)

        unique = _UniqueCoordinateSet(DEFAULT_RMSD_CUTOFF)
        decisions = [unique.add(coor) for coor in candidates]
        assert decisions == decisions_ref
        assert len(unique) == len(reference) < len(candidates)
        for a, b in zip(unique.coor_list, reference):
            assert a is b
