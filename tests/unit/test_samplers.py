"""
Unit tests for qfit.samplers
"""

import os.path as op

import numpy as np
import pytest

from qfit.samplers import (BackboneRotator,
                           BisectingAngleRotator,
                           BondRotator,
                           CBAngleRotator,
                           ChiRotator,
                           GlobalRotator,
                           Translator)
from qfit.structure import Structure
from qfit.structure.ligand import Ligand

from .base_test_case import UnitBase


class TestSamplers(UnitBase):
    def setUp(self):
        super().setUp()
        asa_single_file = op.join(self.DATA, "ASA_single.pdb")
        self._STRUCTURE_ASA_SINGLE = Structure.fromfile(asa_single_file)
        aka_single_file = op.join(self.DATA, "AKA_single.pdb")
        self._STRUCTURE_AKA_SINGLE = Structure.fromfile(aka_single_file)
        ppi_single_file = op.join(self.DATA, "PPI_single.pdb")
        self._STRUCTURE_PPI_SINGLE = Structure.fromfile(ppi_single_file)

    def test_samplers_backbone_rotator(self):
        s = self._STRUCTURE_ASA_SINGLE.copy()
        s_orig = s.copy()
        seg = s.chains[0].conformers[0].segments[0]
        bbr = BackboneRotator(seg)
        assert bbr.ndofs == 6
        assert len(bbr._origins) == len(bbr._aligners) == 6  # pylint: disable=protected-access
        torsions = np.zeros(bbr.ndofs, float)
        bbr(torsions)
        assert np.all(s.coor == s_orig.coor)
        torsions += 1
        bbr(torsions)
        # first two atoms are unchanged
        assert np.sum(s.coor == s_orig.coor) == 6
        assert np.all((s.coor == s_orig.coor)[0:2])
        assert np.max(s.coor - s_orig.coor) == pytest.approx(0.13925, abs=0.00001)
        assert s.rmsd(s_orig) == pytest.approx(0.080397, abs=0.00001)
        # reset angles
        torsions = np.zeros(bbr.ndofs, float)
        bbr(torsions)
        assert np.all(s.coor == s_orig.coor)

    def test_samplers_cbangle_rotator(self):
        s = self._STRUCTURE_AKA_SINGLE.copy()
        s_orig = s.copy()
        seg = s.chains[0].conformers[0].segments[0]
        cbr = CBAngleRotator(seg.residues[1])
        cbr(60)
        assert s.rmsd(s_orig) == pytest.approx(1.59877, abs=0.00001)
        cbr(120)
        assert s.rmsd(s_orig) == pytest.approx(2.76916, abs=0.00001)
        cbr(0)
        assert s.rmsd(s_orig) == pytest.approx(0, abs=1e-12)
        cbr(360)
        assert s.rmsd(s_orig) == pytest.approx(0, abs=1e-12)

    def test_samplers_chi_rotator(self):
        s = self._STRUCTURE_AKA_SINGLE.copy()
        s_orig = s.copy()
        seg = s.chains[0].conformers[0].segments[0]
        cr = ChiRotator(seg.residues[1], 1)
        cr(60)
        assert s.rmsd(s_orig) == pytest.approx(0.972117, abs=0.00001)
        # four atoms moved
        assert np.sum(np.abs(s.coor - s_orig.coor) > 0.01) == 12
        cr(0)
        assert s.rmsd(s_orig) == pytest.approx(0, abs=1e-12)
        cr(360)
        assert s.rmsd(s_orig) == pytest.approx(0, abs=1e-12)
        cr(-60)
        assert s.rmsd(s_orig) == pytest.approx(0.972117, abs=0.00001)
        cr(0)
        cr = ChiRotator(seg.residues[1], 4)
        cr(180)
        atom_sel = s.name == "NZ"
        assert np.sum(np.abs(s.coor - s_orig.coor) > 0.01) == 3
        assert np.all((np.abs(s.coor - s_orig.coor) > 0.01)[atom_sel])

    def test_samplers_translator(self):
        s = self._STRUCTURE_PPI_SINGLE.copy()
        s_orig = s.copy()
        lig = Ligand.from_structure(s)
        tr = Translator(lig)
        tr([1, 1, 1])
        assert s.rmsd(s_orig) == pytest.approx(np.sqrt(3), abs=1e-8)
        tr([0, 0, 0])
        assert s.rmsd(s_orig) == 0

    def test_samplers_global_rotator(self):
        s = self._STRUCTURE_PPI_SINGLE.copy()
        s_orig = s.copy()
        lig = Ligand.from_structure(s)
        gr = GlobalRotator(lig)
        identity = [[1,0,0],[0,1,0],[0,0,1]]
        gr(identity)
        assert s.rmsd(s_orig) == 0
        gr([[1,0,0],[0,1,0],[0,1,0]])
        assert s.rmsd(s_orig) == pytest.approx(1.253416, abs=0.00001)
        gr(identity)
        assert s.rmsd(s_orig) == 0
        gr([[-1,0,0],[0,-1,0],[0,0,-1]])
        assert s.rmsd(s_orig) == pytest.approx(2.91190, abs=0.00001)
        gr(identity)
        assert s.rmsd(s_orig) == 0
        gr = GlobalRotator(lig, center=(4, 4, 1))
        gr([[-1,0,0],[0,-1,0],[0,0,-1]])
        assert s.rmsd(s_orig) == pytest.approx(3.073002, abs=0.00001)
        gr(identity)
        assert s.rmsd(s_orig) == 0
        gr = GlobalRotator(lig, (40, 40, 40))
        gr([[-1,0,0],[0,-1,0],[0,0,-1]])
        assert s.rmsd(s_orig) == pytest.approx(128.97431, abs=0.00001)

    def test_samplers_bond_rotator(self):
        s = self._STRUCTURE_PPI_SINGLE.copy()
        s_orig = s.copy()
        lig = Ligand.from_structure(s)
        br = BondRotator(lig, "C1", "C2")
        lig.coor = br(60)
        assert s.rmsd(s_orig) == pytest.approx(1.22705, abs=0.00001)
        assert np.sum(np.abs(s.coor - s_orig.coor) > 0.01) == 3
        assert np.all((np.abs(s.coor - s_orig.coor) > 0.01)[2])  # C3 atom
        lig.coor = br(0)
        assert s.rmsd(s_orig) == pytest.approx(0, abs=1e-15)

    def test_samplers_bisecting_angle_rotator(self):
        s = self._STRUCTURE_AKA_SINGLE.copy()
        s_orig = s.copy()
        res = s.chains[0].conformers[0].segments[0].residues[1]
        bar = BisectingAngleRotator(res)
        bar(10)
        res_orig = s_orig.chains[0].conformers[0].segments[0].residues[1]
        assert res_orig.rmsd(res) == pytest.approx(0.39955, abs=0.00001)


def _reference_backbone_rotate(rotator, torsions):
    """Previous BackboneRotator.__call__: read/write the structure per torsion."""
    from qfit.structure.math import Rz
    torsions = np.deg2rad(np.asarray(torsions)[::-1])
    segment = rotator.segment
    segment.coor = rotator._starting_coor
    iterator = zip(torsions, rotator._origins, rotator._aligners,
                   rotator._atoms_to_rotate)
    for torsion, origin, aligner, atoms_to_rotate in iterator:
        if torsion == 0.0:
            continue
        coor = segment.get_xyz(atoms_to_rotate)
        coor -= origin
        R = aligner.forward_rotation @ Rz(torsion) @ aligner.backward_rotation
        coor = np.dot(coor, R.T)
        coor += origin
        segment.set_xyz(coor, atoms_to_rotate)
    return segment.coor


class TestBackboneRotatorEquivalence(UnitBase):

    def test_matches_reference(self):
        for pdb_name in ("GNNAFNS_single.pdb", "AKA_single.pdb"):
            s = Structure.fromfile(op.join(self.DATA, pdb_name)).reorder()
            seg = s.chains[0].conformers[0].segments[0]
            # A sub-segment, so that segment positions differ from i_seq
            sub = seg[1:len(seg)] if len(seg) > 3 else seg
            rotator = BackboneRotator(sub)
            rng = np.random.default_rng(0)
            for _ in range(5):
                torsions = rng.uniform(-30, 30, rotator.ndofs)
                torsions[rng.integers(rotator.ndofs)] = 0.0
                rotator(torsions)
                new = sub.coor
                ref = _reference_backbone_rotate(rotator, torsions)
                assert np.array_equal(new, ref)
            # Atoms outside the segment are never moved
            rest = s.extract(np.setdiff1d(np.arange(s.natoms),
                                          sub._numpy_selection))
            original = Structure.fromfile(op.join(self.DATA, pdb_name)).reorder()
            rest_orig = original.extract(np.setdiff1d(np.arange(original.natoms),
                                                      sub._numpy_selection))
            assert np.array_equal(rest.coor, rest_orig.coor)
