"""
Tests for RotamerResidue clash detection and chi-angle helpers.

The vectorized implementations are compared against verbatim copies of the
previous pure-Python implementations (the `_reference_*` functions).
"""

import numpy as np
import pytest

from qfit.structure import Structure
from qfit.structure.math import Rz, dihedral_angle, gram_schmidt_orthonormal_zx
from qfit.structure.rotamers import ROTAMERS

from .base_test_case import UnitBase


def _reference_clash_arrays(residue, scaling_factor=0.75):
    """Previous RotamerResidue._init_clash_detection, without side effects."""
    natoms = residue.natoms
    ndistances = natoms * (natoms - 1) // 2
    clash_mask = np.ones(ndistances, bool)
    clash_radius2 = np.zeros(ndistances, float)
    radii = residue.covalent_radius
    bonds = residue.get_residue_info("bonds")
    offset = natoms * (natoms - 1) // 2
    for i in range(natoms - 1):
        natoms_left = natoms - i
        starting_index = offset - natoms_left * (natoms_left - 1) // 2 - i - 1
        name1 = residue.name[i]
        covrad1 = radii[i]
        for j in range(i + 1, natoms):
            bond1 = [name1, residue.name[j]]
            bond2 = bond1[::-1]
            covrad2 = radii[j]
            index = starting_index + j
            clash_radius2[index] = covrad1 + covrad2 + 0.4
            if bond1 in bonds or bond2 in bonds:
                clash_mask[index] = False
    clash_radius2 *= clash_radius2
    clash_radius2 *= scaling_factor
    return clash_radius2, clash_mask


def _reference_clashes(residue):
    """Previous RotamerResidue.clashes (pure-Python distance loop)."""
    coor = residue.coor
    natoms = residue.natoms
    dm = np.empty(natoms * (natoms - 1) // 2)
    k = 0
    for i in range(natoms - 1):
        u = coor[i]
        for j in range(i + 1, natoms):
            u_v = u - coor[j]
            dm[k] = np.dot(u_v, u_v)
            k += 1
    clashing = dm <= residue._clash_radius2
    clashing &= residue._clash_mask
    clashing &= residue._active_mask
    return clashing.sum()


def _reference_get_chi(residue, chi_index):
    """Previous RotamerResidue.get_chi (reads every atom of the structure)."""
    atoms = residue.get_residue_info("chi")[chi_index]
    selection = residue.select("name", atoms)
    ordered_sel = []
    for atom in atoms:
        for i_seq in selection:
            if atom == residue._atoms[int(i_seq)].name.strip():
                ordered_sel.append(i_seq)
                break
    coor = residue._atoms.extract_xyz().as_numpy_array()[ordered_sel]
    return dihedral_angle(coor)


def _reference_set_chi(residue, chi_index, value):
    """Previous RotamerResidue.set_chi (reads every atom of the structure)."""
    atoms = residue.get_residue_info("chi")[chi_index]
    selection = residue.select("name", atoms)
    coor = residue._atoms.extract_xyz().as_numpy_array()[selection]
    origin = coor[1].copy()
    coor -= origin
    backward = gram_schmidt_orthonormal_zx(coor)
    forward = backward.T
    atoms_to_rotate = residue.get_residue_info("chi-rotate")[chi_index]
    selection = residue.select("name", atoms_to_rotate)
    angle = np.deg2rad(value - _reference_get_chi(residue, chi_index))
    R = forward @ Rz(angle) @ backward
    coor_to_rotate = residue._atoms.extract_xyz().as_numpy_array()[selection]
    coor_to_rotate -= origin
    coor_to_rotate = np.dot(coor_to_rotate, R.T)
    coor_to_rotate += origin
    residue.set_xyz(coor_to_rotate, selection)


class TestResidueClashDetection(UnitBase):

    def _load_residues(self):
        structure = Structure.fromfile(
            self._get_file_path("GNNAFNS_single.pdb")).reorder()
        residues = [r for r in structure.single_conformer_residues
                    if r.resn[0] in ROTAMERS and r.natoms > 1]
        assert len(residues) > 3
        return structure, residues

    def test_rotamer_residue_does_not_build_clash_arrays_eagerly(self):
        _, residues = self._load_residues()
        for residue in residues:
            assert residue._ndistances is None

    def test_clash_arrays_match_reference(self):
        _, residues = self._load_residues()
        for scaling in (0.75, 0.85):
            for residue in residues:
                residue._init_clash_detection(scaling)
                ref_radius2, ref_mask = _reference_clash_arrays(residue, scaling)
                assert np.allclose(residue._clash_radius2, ref_radius2)
                assert np.array_equal(residue._clash_mask, ref_mask)

    def test_clashes_match_reference(self):
        _, residues = self._load_residues()
        rng = np.random.default_rng(0)
        for residue in residues:
            start = residue.coor
            for scale in (0.0, 0.3, 1.0, 3.0):
                residue.coor = start + rng.normal(scale=scale, size=start.shape)
                assert residue.clashes() == _reference_clashes(residue)
            # Collapse everything onto one point: every non-bonded pair clashes
            residue.coor = np.zeros_like(start)
            assert residue.clashes() == _reference_clashes(residue) > 0
            residue.coor = start

    def test_clashes_respect_active_atoms(self):
        _, residues = self._load_residues()
        residue = max(residues, key=lambda r: r.natoms)
        residue.coor = np.zeros_like(residue.coor)
        n_all = residue.clashes()
        residue.clear_active()
        residue.set_active(residue.select("name", ("N", "CA", "C")))
        residue.update_clash_mask()
        assert residue.clashes() == _reference_clashes(residue) < n_all

    def test_clash_mask_checks_only_active_pairs(self):
        _, residues = self._load_residues()
        rng = np.random.default_rng(1)
        for residue in residues:
            n = residue.natoms
            for _ in range(5):
                active = rng.random(n) < 0.6
                residue.clear_active()
                residue.set_active(residue._numpy_selection[active], True)
                residue.update_clash_mask()
                i, j = np.triu_indices(n, k=1)
                assert np.array_equal(residue._active_mask, active[i] & active[j])
            residue.active = True
            residue.update_clash_mask()
            assert residue._active_mask.all()

    def test_inactive_atoms_do_not_clash(self):
        # Move one inactive atom onto an active one: no clash. Then move an
        # active atom onto another active one: clash.
        _, residues = self._load_residues()
        residue = max(residues, key=lambda r: r.natoms)
        names = list(residue.name)
        start = residue.coor
        residue.clear_active()
        residue.set_active(residue.select("name", ("N", "CA", "C", "O", "CB")))
        residue.update_clash_mask()
        assert residue.clashes() == 0

        inactive = next(k for k, a in enumerate(residue.active) if not a)
        coor = start.copy()
        coor[inactive] = coor[names.index("N")]
        residue.coor = coor
        assert residue.clashes() == 0

        coor = start.copy()
        coor[names.index("CB")] = coor[names.index("O")]  # not bonded
        residue.coor = coor
        assert residue.clashes() > 0
        residue.coor = start
        residue.active = True
        residue.update_clash_mask()

    def test_lazy_init_preserves_active_flags(self):
        # Atoms deactivated before the first clash query must stay inactive.
        _, residues = self._load_residues()
        residue = max(residues, key=lambda r: r.natoms)
        residue.clear_active()
        residue.set_active(residue.select("name", ("N", "CA")))
        residue.update_clash_mask()  # triggers lazy initialization
        assert residue.active.sum() == 2

    def test_rotamer_library_is_not_shared(self):
        _, residues = self._load_residues()
        residue = residues[0]
        resname = residue.resname
        n_library = len(ROTAMERS[resname]["rotamers"])
        residue.rotamers.append([0.0] * residue.nchi)
        assert len(ROTAMERS[resname]["rotamers"]) == n_library


class TestChiAngles(UnitBase):

    def _residues_with_chis(self, structure):
        return [r for r in structure.single_conformer_residues
                if r.resn[0] in ROTAMERS and r.nchi > 0]

    def test_get_chi_matches_reference(self):
        structure = Structure.fromfile(
            self._get_file_path("GNNAFNS_single.pdb")).reorder()
        for residue in self._residues_with_chis(structure):
            for chi_index in range(1, residue.nchi + 1):
                assert residue.get_chi(chi_index) == pytest.approx(
                    _reference_get_chi(residue, chi_index))

    def test_set_chi_matches_reference(self):
        new = Structure.fromfile(
            self._get_file_path("GNNAFNS_single.pdb")).reorder()
        ref = new.copy()
        for res_new, res_ref in zip(self._residues_with_chis(new),
                                    self._residues_with_chis(ref)):
            for chi_index in range(1, res_new.nchi + 1):
                for value in (-170.0, -60.0, 65.0, 180.0):
                    res_new.set_chi(chi_index, value)
                    _reference_set_chi(res_ref, chi_index, value)
                    assert np.allclose(res_new.coor, res_ref.coor, atol=1e-6)
                    assert res_new.get_chi(chi_index) == pytest.approx(
                        value, abs=1e-3) or res_new.get_chi(
                            chi_index) == pytest.approx(value - 360, abs=1e-3)
