"""
Unit tests for structure I/O and basic manipulation
"""

import tempfile
import unittest
import os.path as op

from qfit.structure import Structure
from qfit.structure.pdbfile import read_pdb

from .base_test_case import UnitBase


class TestStructure(UnitBase):
    def test_pdbfile(self):
        pdb = read_pdb(self.TINY_PDB)
        assert pdb.crystal_symmetry is not None
        assert pdb.resolution == 1.39
        assert len(pdb.coor["record"]) == 40
        assert pdb.coor["record"][-1] == "ATOM"
        assert pdb.coor["name"][-1] == "HZ"
        assert pdb.coor["charge"][-1] == ""
        assert pdb.coor["resn"][-1] == "PHE"
        assert pdb.coor["resi"][-1] == 113
        assert pdb.coor["chain"][-1] == "A"
        assert pdb.coor["altloc"][-1] == "B"
        assert pdb.coor["x"][-1] == 1.431
        assert pdb.coor["b"][-1] == 29.87
        assert pdb.coor["q"][-1] == 0.37
        assert pdb.coor["e"][-1] == "H"
        assert len(pdb.link) == 0
        assert len(pdb.anisou["atomid"]) == 22
        assert pdb.anisou["u00"][0] == 962
        assert pdb.anisou["u11"][0] == 910
        assert pdb.anisou["u22"][0] == 1702
        assert pdb.anisou["u01"][0] == -88
        assert pdb.anisou["u02"][0] == 638
        assert pdb.anisou["u12"][0] == -737

    def test_structure_fromfile(self):
        def _check_structure(s):
            assert s.unit_cell is not None
            assert s.n_residues == 1
            assert len(list(s.chains)) == 1
            assert len(list(s.residue_groups)) == 1
            assert len(list(s.single_conformer_residues)) == 1
            assert len(list(s.atoms)) == 40
            assert s.average_conformers() == 2
            assert s.clashes() == 24  # XXX is this correct?
            assert s.data["record"][-1] == "ATOM"
            assert s.data["name"][-1] == "HZ"
            assert s.data["charge"][-1] == ""
            assert s.data["resn"][-1] == "PHE"
            assert s.data["resi"][-1] == 113
            assert s.data["chain"][-1] == "A"
            assert s.data["altloc"][-1] == "B"
            assert s.data["b"][-1] == 29.87
            assert s.data["q"][-1] == 0.37
            assert s.data["e"][-1] == "H"
            assert (
                str(s.unit_cell)
                == "UnitCell(a=8.000000, b=12.000000, c=15.000000, alpha=90.000000, beta=90.000000, gamma=90.000000)"
            )

        s1 = Structure.fromfile(self.TINY_PDB)
        _check_structure(s1)
        # XXX these are read-only
        assert s1.data["u00"][0] == 962
        assert s1.data["u11"][0] == 910
        assert s1.data["u22"][0] == 1702
        assert s1.data["u01"][0] == -88
        assert s1.data["u02"][0] == 638
        assert s1.data["u12"][0] == -737
        # I/O recycling
        pdb_tmp_out = tempfile.NamedTemporaryFile(suffix=".pdb").name
        print(pdb_tmp_out)
        s1.tofile(pdb_tmp_out)
        s2 = Structure.fromfile(pdb_tmp_out)
        assert str(s2.unit_cell) == str(s1.unit_cell)
        _check_structure(s2)
