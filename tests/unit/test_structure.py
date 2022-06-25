"""
Unit tests for structure I/O and basic manipulation
"""

import tempfile
import unittest
import os.path as op

import numpy as np

from qfit.structure import Structure
from qfit.structure.pdbfile import read_pdb, write_pdb

from .base_test_case import UnitBase

class TestStructure(UnitBase):

    def _validate_tiny_pdb(self, pdb):
        assert pdb.crystal_symmetry is not None
        assert pdb.resolution == 1.39
        assert len(pdb.coor["record"]) == 40
        assert pdb.coor["record"][-1] == "ATOM"
        assert pdb.coor["name"][-1] == "HZ"
        assert pdb.coor["charge"][-1] == ""
        assert pdb.coor["resn"][-1] == "PHE"
        assert pdb.coor["resi"][-1] == 113
        assert pdb.coor["chain"][-1] == "A"
        assert pdb.coor["icode"][-1] == ""
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

    def test_read_pdb(self):
        pdb = read_pdb(self.TINY_PDB)
        assert pdb.file_format == "pdb"
        self._validate_tiny_pdb(pdb)

    def test_read_mmcif(self):
        pdb = read_pdb(self.TINY_CIF)
        assert pdb.file_format == "cif"
        self._validate_tiny_pdb(pdb)

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
            assert s.data["icode"][-1] == ""
            assert s.data["b"][-1] == 29.87
            assert s.data["q"][-1] == 0.37
            assert s.data["e"][-1] == "H"
            assert str(s.unit_cell) == "UnitCell(a=8.000000, b=12.000000, c=15.000000, alpha=90.000000, beta=90.000000, gamma=90.000000)"
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
        write_pdb(pdb_tmp_out, s2)
        s3 = Structure.fromfile(pdb_tmp_out)
        assert str(s3.unit_cell) == str(s1.unit_cell)
        _check_structure(s3)
        # test gzip I/O support
        s3.tofile(pdb_tmp_out + ".gz")
        s4 = Structure.fromfile(pdb_tmp_out + ".gz")
        _check_structure(s4)
        # mmCIF input
        s5 = Structure.fromfile(self.TINY_CIF)
        assert s5.file_format == "cif"
        _check_structure(s5)
        # recycling to PDB should work here too
        s5.tofile(pdb_tmp_out + ".gz")
        s6 = Structure.fromfile(pdb_tmp_out + ".gz")
        _check_structure(s6)

    def test_structure_with_links(self):
        def _check_structure(s):
            assert s.data["icode"][0] == ""
            assert s.data["altloc"][0] == ""
            assert len(s.link_data["record"]) == 5
            assert s.link_data["name1"][0] == "ZN"
            assert s.link_data["altloc1"][0] == ""
            assert s.link_data["resn1"][0] == "ZN"
            assert s.link_data["chain1"][0] == "A"
            assert s.link_data["resi1"][0] == 701
            assert s.link_data["name2"][0] == "O"
            assert s.link_data["altloc2"][0] == "A"
            assert s.link_data["chain2"][0] == "A"
            assert s.link_data["resi2"][0] == 702
            assert s.link_data["resn2"][0] == "28T"
            assert np.round(s.link_data["length"][0], decimals=2) == 1.84
            assert s.link_data["resn1"][-1] == "HIS"
            assert s.link_data["resn2"][-1] == "ZN"
            assert np.round(s.link_data["length"][-1], decimals=2) == 2.09
            assert len([r for r in s.data["resn"] if r == "ZN"]) == 1
            assert len([r for r in s.data["record"] if r == "HETATM"]) == 40
        PDB = op.join(self.DATA, "4ms6_tiny.pdb.gz")
        CIF = op.join(self.DATA, "4ms6_tiny.cif.gz")
        for fname, ftype in zip([PDB, CIF], ["pdb", "cif"]):
            s1 = Structure.fromfile(fname)
            assert s1.file_format == ftype
            _check_structure(s1)
            pdb_tmp_out = tempfile.NamedTemporaryFile(suffix=".pdb.gz").name
            s1.tofile(pdb_tmp_out)
            s2 = Structure.fromfile(pdb_tmp_out)
            assert s2.file_format == "pdb"
            _check_structure(s2)
