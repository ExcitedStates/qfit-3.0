"""
Unit tests for structure I/O and basic manipulation
"""

import tempfile
import os.path as op

import numpy as np
import pytest

from qfit.structure import Structure
from qfit.structure.residue import RotamerResidue
from qfit.structure.pdbfile import write_pdb

from .base_test_case import UnitBase


class TestStructureIO(UnitBase):
    def _validate_tiny_structure(self, s):
        assert s.unit_cell is not None
        assert s.n_residues() == 1
        assert len(list(s.chains)) == 1
        assert len(list(s.residue_groups)) == 1
        assert len(list(s.single_conformer_residues)) == 1
        assert len(list(s.atoms)) == 40
        assert s.average_conformers() == 2
        assert s.clashes() == 24  # XXX is this correct?
        assert s.name[-1] == "HZ"
        assert s.charge[-1] == ""
        assert s.resn[-1] == "PHE"
        assert s.resi[-1] == 113
        assert s.chain[-1] == "A"
        assert s.altloc[-1] == "B"
        assert s.icode[-1] == ""
        assert s.b[-1] == 29.87
        assert s.q[-1] == 0.37
        assert s.e[-1] == "H"
        assert (
            str(s.unit_cell)
            == "UnitCell(a=8.000000, b=12.000000, c=15.000000, alpha=90.000000, beta=90.000000, gamma=90.000000)"
        )

    def test_structure_file_io_pdb(self):
        s1 = Structure.fromfile(self.TINY_PDB)
        self._validate_tiny_structure(s1)
        # I/O recycling
        pdb_tmp_out = tempfile.NamedTemporaryFile(suffix=".pdb").name
        s1.tofile(pdb_tmp_out)
        s2 = Structure.fromfile(pdb_tmp_out)
        assert str(s2.unit_cell) == str(s1.unit_cell)
        self._validate_tiny_structure(s2)
        write_pdb(pdb_tmp_out, s2)
        s3 = Structure.fromfile(pdb_tmp_out)
        assert str(s3.unit_cell) == str(s1.unit_cell)
        self._validate_tiny_structure(s3)

    def test_structure_file_io_pdb_gzip(self):
        pdb_tmp_out = tempfile.NamedTemporaryFile(suffix=".pdb").name
        s1 = Structure.fromfile(self.TINY_PDB)
        s1.tofile(pdb_tmp_out + ".gz")
        s2 = Structure.fromfile(pdb_tmp_out + ".gz")
        self._validate_tiny_structure(s2)

    def test_structure_file_io_mmcif(self):
        # mmCIF input
        s5 = Structure.fromfile(self.TINY_CIF)
        assert s5.file_format == "cif"
        self._validate_tiny_structure(s5)
        # recycle to mmCIF
        pdb_tmp_out = tempfile.NamedTemporaryFile(suffix=".pdb").name
        cif_tmp_out = tempfile.NamedTemporaryFile(suffix=".cif").name
        s5.tofile(cif_tmp_out + ".gz")
        s6 = Structure.fromfile(cif_tmp_out + ".gz")
        assert s6.file_format == "cif"
        self._validate_tiny_structure(s6)
        # recycling to PDB should work here too
        s5.tofile(pdb_tmp_out + ".gz")
        s7 = Structure.fromfile(pdb_tmp_out + ".gz")
        assert s7.file_format == "pdb"
        self._validate_tiny_structure(s7)

    def test_structure_with_links(self):
        def _check_structure(s):
            assert s.icode[0] == ""
            assert s.altloc[0] == ""
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
            assert len([r for r in s.resn if r == "ZN"]) == 1

        PDB = op.join(self.DATA, "4ms6_tiny.pdb.gz")
        CIF = op.join(self.DATA, "4ms6_tiny.cif.gz")
        pdb_tmp = tempfile.NamedTemporaryFile(suffix=".pdb.gz").name
        cif_tmp = tempfile.NamedTemporaryFile(suffix=".cif.gz").name
        for fname, ftype in zip([PDB, CIF], ["pdb", "cif"]):
            s1 = Structure.fromfile(fname)
            assert s1.file_format == ftype
            _check_structure(s1)
            # recycle IO in both formats
            for ftype2, tmp_out in zip(["pdb", "cif"], [pdb_tmp, cif_tmp]):
                s1.tofile(tmp_out)
                s2 = Structure.fromfile(tmp_out)
                assert s2.file_format == ftype2
                _check_structure(s2)

    # XXX see also tests/test_qfit_model_io.py
    def test_structure_extract_io_recycling(self):
        PDB = op.join(self.DATA, "4ms6_tiny.pdb.gz")
        pdb_tmp = tempfile.NamedTemporaryFile(suffix=".pdb.gz").name
        s1 = Structure.fromfile(PDB)
        assert len(list(s1.single_conformer_residues)) == 5
        s2 = s1.extract("resi", (295, 299), "==")
        assert len(list(s2.single_conformer_residues)) == 2
        s2.tofile(pdb_tmp)
        s3 = Structure.fromfile(pdb_tmp)
        assert len(list(s3.single_conformer_residues)) == 2



class StructureUnitBase(UnitBase):
    def setUp(self):
        super().setUp()
        awa_3conf_file = op.join(self.DATA, "AWA_3conf.pdb")
        self._STRUCTURE_AWA_3CONF = Structure.fromfile(awa_3conf_file)
        awa_single_file = op.join(self.DATA, "AWA_single.pdb")
        self._STRUCTURE_AWA_SINGLE = Structure.fromfile(awa_single_file)
        gnnafns_multi_file = op.join(self.DATA, "GNNAFNS_multiconf.pdb")
        self._STRUCTURE_7MER_MULTI = Structure.fromfile(gnnafns_multi_file)
        self._TYR_PDB_IN = """\
ATOM      1  N   TYR    22       3.603   0.782   5.912  1.00  0.00           N
ATOM      2  CA  TYR    22       5.028   0.899   5.616  1.00  0.00           C
ATOM      3  C   TYR    22       5.857   0.191   6.660  1.00  0.00           C
ATOM      4  O   TYR    22       5.343  -0.394   7.617  1.00  0.00           O
ATOM      5  CB  TYR    22       5.468   2.392   5.585  1.00  0.00           C
ATOM      6  CG  TYR    22       4.806   3.283   4.527  1.00  0.00           C
ATOM      7  CD1 TYR    22       5.223   3.235   3.193  1.00  0.00           C
ATOM      8  CD2 TYR    22       3.769   4.145   4.895  1.00  0.00           C
ATOM      9  CE1 TYR    22       4.604   4.038   2.239  1.00  0.00           C
ATOM     10  CE2 TYR    22       3.152   4.947   3.940  1.00  0.00           C
ATOM     11  CZ  TYR    22       3.570   4.893   2.612  1.00  0.00           C
ATOM     12  OH  TYR    22       2.964   5.677   1.672  1.00  0.00           O
TER
END"""
        pdb_tmp = self._write_tmp_pdb(self._TYR_PDB_IN)
        self._STRUCTURE_TYROSINE = Structure.fromfile(pdb_tmp)


def _simple_rmsd(s1, s2):
    return np.sqrt(np.sum((s2.coor - s1.coor)**2)/s1.total_length)


class TestBaseStructureMethods(StructureUnitBase):
    """
    Unit tests for qfit.structure.base_structure.BaseStructure
    """

    def test_base_structure_copy(self):
        s = self._STRUCTURE_AWA_3CONF
        q_initial = s.q.copy()
        ss = s.copy()
        ss.set_occupancies(0.5)
        assert np.all(ss.q == 0.5)
        assert np.all(s.q == q_initial)

    def test_base_structure_translate(self):
        s = self._STRUCTURE_AWA_SINGLE
        s_initial = s.copy()
        t = np.array([0.5, 0.5, 0.5])
        s.translate(t)
        assert s.rmsd(s_initial) == np.sqrt(np.sum(t**2))
        # test extract behavior - translate() should only affect selected xyz
        s = s_initial.copy()
        ss = s.extract("name", ("CA","C","CB","O"))
        ss.translate([10,10,10])
        assert s.rmsd(s_initial) == pytest.approx(12.247, abs=0.01)

    def test_base_structure_rmsd(self):
        def _flip_ring(s):
            # FIXME would be better to use CCTBX methods for this
            for base_name in ["CD", "CE"]:
                name1, name2 = f" {base_name}1", f" {base_name}2"
                atom_names = s.name
                for atom in s.atoms:
                    if atom.name == name1:
                        atom.name = name2
                    elif atom.name == name2:
                        atom.name = name1
            return s.reorder()

        def test_ring_flip(s, simple_rmsd):
            ss = s.copy()
            assert s.rmsd(ss) == 0.0
            ss.translate(np.array([0.1, 0.0, 0.0]))
            assert s.rmsd(ss) == pytest.approx(0.1, abs=0.0000000001)
            ss = _flip_ring(ss)
            assert _simple_rmsd(ss, s) == pytest.approx(simple_rmsd, abs=0.001)
            assert s.rmsd(ss) == pytest.approx(0.1, abs=0.0000000001)

        test_ring_flip(self._STRUCTURE_TYROSINE, 1.398)
        phe_pdb = "\n".join(
            [l for l in self._TYR_PDB_IN.split("\n") if not "OH" in l]
        ).replace("TYR", "PHE")
        phe_structure = Structure.fromfile(self._write_tmp_pdb(phe_pdb))
        test_ring_flip(phe_structure, 1.4599)

    def test_base_structure_rotate(self):
        s = self._STRUCTURE_AWA_SINGLE
        ss = s.copy()
        r_unity = np.array([[1,0,0],[0,1,0],[0,0,1]])
        ss.rotate(r_unity)
        assert _simple_rmsd(ss, s) == 0
        r_inv = np.array([[-1,0,0],[0,-1,0],[0,0,-1]])
        ss.rotate(r_inv)
        assert _simple_rmsd(ss, s) == pytest.approx(16.61454, abs=0.00001)

    def test_base_structure_select(self):
        s = self._STRUCTURE_AWA_3CONF
        sel = s.select("altloc", "A")
        assert len(sel) == 14
        sel = s.select("altloc", ("B","C"), "==")
        assert len(sel) == 28
        sel = s.select("altloc", ("B","C"), "!=")
        assert len(sel) == 24
        sel = s.select("name", ("CA","C","N","O"), "!=")
        assert len(sel) == 32
        sel = s.select("resn", "TRP")
        assert len(sel) == 42
        assert all([x == "TRP" for x in s.resn[sel]])

    def test_base_structure_active_flag(self):
        s = self._STRUCTURE_AWA_SINGLE.copy()
        assert np.all(s.active)
        assert len(s.active) == s.total_length == s.natoms
        assert len(s.active) == len(s.q)
        s.active = False
        assert np.all(s.active == False)
        s.active = True
        assert np.all(s.active)
        s.clear_active()
        assert np.all(s.active == False)
        s.set_active(1, True)
        assert np.sum(s.active) == 1
        s.set_active()
        assert np.all(s.active)
        ss = s.extract("name", ("CA",))
        assert len(ss.active) == 3
        assert np.sum(ss.active) == 3
        ss.active = np.array([False, True, True])
        assert np.sum(ss.active) == 2

    def test_base_structure_structure_properties(self):
        s = self._STRUCTURE_AWA_SINGLE.copy()  # make copy first
        assert np.all(s.charge == '')
        assert np.all(s.chain == 'A')
        assert set(s.e) == {'C','N','O'}
        assert len(s.q) == 24
        assert set(s.resi) == {1,2,3}
        assert np.sum(s.resn == "ALA") == 10
        assert s.coor.shape == (24, 3)
        # test copy-on-get behavior
        q = s.q
        q[:] = 0
        assert np.all(s.q > 0)
        # test setters
        s.q = 0
        assert np.all(s.q == 0)
        s.q = np.full(s.q.size, 1.0)
        assert np.all(s.q == 1.0)
        s.b = 10
        assert np.all(s.b == 10)
        ss = s.copy()
        ss.coor = ss.coor - [1.0,1.0,1.0]
        rmsd = _simple_rmsd(ss, s)
        assert rmsd == pytest.approx(np.sqrt(3), abs=0.00000001)
        with pytest.raises(RuntimeError):
            ss.coor = s.coor[:-1]
        ss.name = "CB"
        assert set(ss.name) == {"CB"}
        altlocs = s.altloc
        s.altloc = "A"
        ss = s.copy().extract("altloc", "A", "!=")
        assert ss.q.size == 0
        # extract structure
        ss = s.copy().extract("name", "CA")
        assert len(ss.q) == 3
        assert len(ss.coor) == 3
        assert set(ss.name) == {"CA"}
        assert set(ss.e) == {"C"}
        assert np.all(ss.get_atom_xyz(2) == [3.023, 2.662, 2.928])

    def test_get_adp_ellipsoid_axes(self):
        s = Structure.fromfile(op.join(self.DATA, "phe113_fake_uc.pdb"))
        ca_atom = s.extract("name", "CA").extract("altloc", "A")
        assert len(ca_atom.name) == 1
        anisous = ca_atom.extract_anisous()
        assert len(anisous) == 1
        np.testing.assert_array_almost_equal(
            anisous[0],
            ((1465.0, -684.0, 354.0),
             (-684.0, 1150.0, -298.0),
             (354.0, -298.0, 1600.0)))
        np.testing.assert_array_almost_equal(
            ca_atom.get_adp_ellipsoid_axes(),
            [[0.61876973, 0.78542739, 0.01509452],
             [-0.43646415,  0.32774904,  0.83790191],
             [-0.65316389,  0.52505655, -0.54561209]])

    def test_structure_record(self):
        s = self._STRUCTURE_AWA_3CONF
        assert np.all(s.record == "ATOM")
        s = Structure.fromfile(op.join(self.DATA, "PPI_single.pdb"))
        assert np.all(s.record == "HETATM")


class TestProteinStructure(StructureUnitBase):
    """
    unit tests for qfit.structure.structure.Structure
    """

    def test_structure_hierarchy(self):
        s = self._STRUCTURE_AWA_3CONF
        assert s.total_length == 52
        assert s.selection is None
        assert len(list(s.atoms)) == 52
        chains = list(s.chains)
        assert len(chains) == 1
        assert chains[0].id == "A"
        assert s["A"] is chains[0]
        assert len(list(s.residue_groups)) == 3
        assert len(list(s.single_conformer_residues)) == 3
        single_residues = list(s.single_conformer_residues)
        assert len(single_residues) == 3
        assert [c.id for c in chains[0].conformers] == ["A","B","C"]
        assert [len(c.residues) for c in chains[0].conformers] == [3,3,3]
        assert len(list(s.residues)) == 9
        r = chains[0].conformers[0].residues[0]
        assert len(r.active) == 5
        assert r.resname == "ALA"
        assert chains[0].conformers[0].residues[1].resname == "TRP"
        # these are associated with conformers
        segments = list(s.segments)
        assert len(segments) == 3
        assert len(segments[0].residues) == 3
        assert segments[0].find(2) == 1
        assert segments[0][1].id == (2, '')
        with pytest.raises(ValueError):
            x = segments[0].find(5)
        s = self._STRUCTURE_AWA_SINGLE
        segments = list(s.segments)
        assert len(segments) == 1

    def test_structure_average_conformers(self):
        s = self._STRUCTURE_AWA_SINGLE
        assert s.average_conformers() == 1
        s = self._STRUCTURE_AWA_3CONF
        assert s.average_conformers() == pytest.approx(1.6666666, abs=0.000001)
        ss = s.extract("altloc", ("B","C"), "!=")
        assert ss.average_conformers() == 1

    def test_structure_extract_and_combine(self):
        s = self._STRUCTURE_AWA_3CONF
        single = s.extract("altloc", ("B","C"), "!=")
        multi = s.extract("altloc", ("B","C"), "==")
        assert len(multi.active) == 28
        assert len(single.active) == 24
        combined = single.combine(multi)
        assert len(combined.active) == 52

    def test_structure_collapse_backbone(self):
        s = self._STRUCTURE_AWA_3CONF
        # this will remove 4 atoms each from B and C conformers
        ss = s.collapse_backbone(2, "A")
        assert ss.total_length == 44
        base_sel = ss.altloc == ""
        assert np.sum(base_sel) == 14
        for x in ["A", "B", "C"]:
            assert np.sum(ss.altloc == x) == 10
        assert np.sum((ss.altloc != "") & (ss.name == "CA")) == 0
        assert np.all(ss.q[base_sel] == 1.0)
        assert np.all(ss.q[~base_sel] < 1.0)
        # sanity test with single-conformer residue
        ss = s.collapse_backbone(1, "A")
        assert ss.total_length == s.total_length

    def test_structure_set_backbone_occ(self):
        s = self._STRUCTURE_AWA_3CONF
        assert np.sum(s.q > 0) == 52
        ss = s.set_backbone_occ()
        assert np.sum(ss.q == 0) == 20
        assert np.sum(ss.q == 1) == 32
        assert np.sum(s.q > 0) == 52
        s = self._STRUCTURE_AWA_SINGLE
        ss = s.set_backbone_occ()
        assert len(ss.active) == len(s.active) == 24
        assert np.sum(s.q) == len(s.q)
        assert np.sum(ss.q == 0) == 12
        for name in ["N","CA","C","O","H","HA"]:
            assert np.sum(ss.q[ss.name == name]) == 0
        # 7-mer with initial glycine residue, which is treated differently
        s = self._STRUCTURE_7MER_MULTI
        ss = s.set_backbone_occ()
        assert np.sum((ss.resn == "GLY") & (ss.q == 1)) == 1
        s = self._STRUCTURE_7MER_MULTI.extract("resn", ("GLY",), "==")
        ss = s.set_backbone_occ()
        assert np.sum((ss.resn == "GLY") & (ss.q == 1)) == 1

    def test_structure_internal_remove_conformer(self):
        s = self._STRUCTURE_AWA_3CONF
        ss = s._remove_conformer(2, "A", "A", "B")  # pylint: disable=protected-access
        assert ss.total_length == 38
        assert np.sum(ss.altloc == "B") == 0
        assert np.sum((ss.altloc == "A") & (ss.q == 0.7)) == 14
        sss = ss._remove_conformer(2, "A", "A", "C")  # pylint: disable=protected-access
        assert sss.total_length == 24
        assert np.sum(sss.altloc == "C") == 0
        assert np.sum((sss.altloc == "A") & (sss.q == 1.0)) == 14

    def test_structure_clashes(self):
        s = self._STRUCTURE_AWA_SINGLE
        assert s.clashes() == 0
        # apparently the clash detection doesn't distinguish between alt confs,
        # it just treats overlapping conformations as clashes
        s = self._STRUCTURE_AWA_3CONF
        assert s.clashes() == 31
        ss = s.extract("altloc", ("B",), "!=")
        assert ss.clashes() == 7
        sss = ss.extract("altloc", ("C",), "!=")
        assert sss.clashes() == 0

    def test_structure_extract_neighbors(self):
        s = self._STRUCTURE_AWA_SINGLE
        r = list(s.residues)[1]
        assert np.all(r.resn == "TRP")
        n = s.extract_neighbors(r)
        assert np.all(n.resi == [1, 1, 1, 1, 1, 3, 3, 3, 3, 3])
        assert np.all(n.resn == "ALA")
        n = s.extract_neighbors(r, 2.0)
        assert np.all((n.name == ["C", "N"]) & (n.resi == [1, 3]))

    def test_structure_remove_identical_conformers(self):
        s1 = self._STRUCTURE_AWA_SINGLE.copy()
        s2 = self._STRUCTURE_AWA_SINGLE.copy()
        s1.altloc = "A"
        s2.altloc = "B"
        s_multi = s1.combine(s2)
        assert s_multi.total_length == 48
        single = s_multi.remove_identical_conformers()
        assert single.total_length == 24
        s2.translate(np.array([0.005, 0.005, 0.005]))
        s_multi = s1.combine(s2)
        single = s_multi.remove_identical_conformers()
        assert single.total_length == 24
        s2.translate(np.array([0.01, 0.01, 0.01]))
        s_multi = s1.combine(s2)
        single = s_multi.remove_identical_conformers()
        assert single.total_length == 48
        single = s_multi.remove_identical_conformers(rmsd_cutoff=0.05)
        assert single.total_length == 24

    def test_structure_reorder(self):
        # glycine with hydrogens
        GLY = """\
ATOM      1 N    GLY A   1       1.931   0.090  -0.034  1.00 20.00           N
ATOM      2 CA   GLY A   1       0.761  -0.799  -0.008  1.00 20.00           C
ATOM      3 C    GLY A   1      -0.498   0.029  -0.005  1.00 20.00           C
ATOM      4 O    GLY A   1      -0.429   1.235  -0.023  1.00 20.00           O
ATOM      5 H    GLY A   1       1.910   0.738   0.738  1.00 20.00           H
ATOM      6 HA2  GLY A   1       0.772  -1.440  -0.889  1.00 20.00           H
ATOM      7 HA3  GLY A   1       0.793  -1.415   0.891  1.00 20.00           H
END"""
        gly_tmp = self._write_tmp_pdb(GLY)
        s = Structure.fromfile(gly_tmp)
        s2 = s.reorder()
        assert np.all(s2.name == s.name)
        lines = GLY.split("\n")
        # scramble the order
        GLY2 = "\n".join([lines[i] for i in [3,5,4,0,2,6,1]])
        gly_tmp2 = self._write_tmp_pdb(GLY2)
        s3 = Structure.fromfile(gly_tmp2)
        # this gets handled automatically by the pdb hierarchy now
        assert np.all(s3.name == s.name)
        # glycine with hydrogens
        GLY3 = """\
ATOM      1  N  AGLY A   1       1.931   0.090  -0.034  0.50 20.00           N
ATOM      2  N  BGLY A   1       1.931   0.090  -0.034  0.50 20.00           N
ATOM      3  CA AGLY A   1       0.761  -0.799  -0.008  0.50 20.00           C
ATOM      4  CA BGLY A   1       0.761  -0.799  -0.008  0.50 20.00           C
ATOM      5  C  AGLY A   1      -0.498   0.029  -0.005  0.50 20.00           C
ATOM      6  C  BGLY A   1      -0.498   0.029  -0.005  0.50 20.00           C
ATOM      7  O  AGLY A   1      -0.429   1.235  -0.023  0.50 20.00           O
ATOM      8  O  BGLY A   1      -0.429   1.235  -0.023  0.50 20.00           O
ATOM      9  H  AGLY A   1       1.910   0.738   0.738  0.50 20.00           H
ATOM     10  H  BGLY A   1       1.910   0.738   0.738  0.50 20.00           H
ATOM     11  HA2AGLY A   1       0.772  -1.440  -0.889  0.50 20.00           H
ATOM     12  HA2BGLY A   1       0.772  -1.440  -0.889  0.50 20.00           H
ATOM     13  HA3AGLY A   1       0.793  -1.415   0.891  0.50 20.00           H
ATOM     14  HA3BGLY A   1       0.793  -1.415   0.891  0.50 20.00           H
END"""
        gly_tmp3 = self._write_tmp_pdb(GLY3)
        print(gly_tmp3)
        s = Structure.fromfile(gly_tmp3)
        s2 = s.reorder()
        assert np.all(s2.altloc[0:7] == "A")
        assert np.all(s2.altloc[7:] == "B")

    def test_structure_normalize_occupancy(self):
        s = self._STRUCTURE_AWA_3CONF.copy()
        s2 = s.normalize_occupancy()
        assert np.all(s2.q == s.q)
        sel1 = s.q == 1.0
        selA = s.altloc == "A"
        selB = s.altloc == "B"
        selC = s.altloc == "C"
        ss = s.copy()
        ss.q = s.q * 0.9
        ss.set_occupancies(0.8, sel1)  # further scaling of single conf
        assert np.all(ss.q != s.q)
        ss2 = ss.normalize_occupancy()
        assert np.all(ss2.q == s.q)
        assert np.all(ss.q != s.q)  # modified structure is unchanged
        assert np.all(ss2.q[sel1] == 1.0)
        assert np.all(ss2.q[selA] == 0.4)
        assert np.all(ss2.q[selB] == 0.3)
        assert np.all(ss2.q[selC] == 0.3)
        # sanity checks on single-conf structure
        s = self._STRUCTURE_AWA_SINGLE.copy()
        assert np.all(s.q == 1.0)
        s2 = s.normalize_occupancy()
        assert np.all(s2.q == 1.0)
        ss = s.copy()
        sel1 = s.resi == 1
        sel2 = s.resi == 2
        sel3 = s.resi == 3
        q = ss.q
        q[sel1] = 0.9
        q[sel2] = 0.8
        q[sel3] = 0.7
        ss.q = q
        assert np.all(ss.q != s.q)
        ss2 = ss.normalize_occupancy()
        assert np.all(ss2.q == 1.0)

    def test_structure_four_digit_residues(self):
        """Check if 4 digit residues are read correctly"""
        pdb_dir = op.join(op.dirname(op.dirname(__file__)), "qfit_io_test")
        # FIXME replace with something much smaller
        pdb_file = op.join(pdb_dir, "4e3y.pdb")
        structure = Structure.fromfile(pdb_file)
        structure = structure.extract("resi", (1024, 1025), "==")
        assert len(list(structure.single_conformer_residues)) == 2

    def test_structure_negative_residue_id(self):
        """Read and write a structure with negative resseq"""
        pdb_dir = op.join(op.dirname(op.dirname(__file__)), "qfit_io_test")
        # FIXME replace with something much smaller
        pdb_file = op.join(pdb_dir, "1fnt.pdb")
        structure = Structure.fromfile(pdb_file)
        structure = structure.extract("chain", "J", "==")
        structure = structure.extract("resi", (-7, -5, -3, -2), "==")
        assert len(list(structure.single_conformer_residues)) == 4
        pdb_tmp = tempfile.NamedTemporaryFile(suffix=".pdb").name
        structure.tofile(pdb_tmp)
        structure2 = Structure.fromfile(pdb_tmp)
        structure2 = structure2.extract("resi", (-3, -2), "==")
        assert len(list(structure2.single_conformer_residues)) == 2

    # XXX unclear what this is actually testing
    def test_structure_hetatm_in_between_atoms(self):
        """Read PDB file with HETATM record appearing in between ATOM records"""
        pdb_dir = op.join(op.dirname(op.dirname(__file__)), "qfit_io_test")
        # FIXME replace with something much smaller
        pdb_file = op.join(pdb_dir, "5orl.pdb")
        structure = Structure.fromfile(pdb_file)
        # Residue 288 is marked as HETATM in the PDB file since it is a
        # modified residue
        structure = structure.extract("resi", (287, 288, 289), "==")
        assert np.unique(structure.resi)[1] == 288


class TestStructureResidue(StructureUnitBase):

    def setUp(self):
        super().setUp()
        ara_single_file = op.join(self.DATA, "ARA_single.pdb")
        self._STRUCTURE_ARA_SINGLE = Structure.fromfile(ara_single_file)

    def test_structure_residue_calc_coordinates(self):
        i = np.array([1.829, 3.56,  3.636])
        j = np.array([3.202, 3.355, 3.193])
        k = np.array([3.316, 3.59,  1.694])
        L = 1.52
        sigL = 0.03
        theta = 1.9914
        sig_theta = 0.035
        chi = 1.082
        xyz = RotamerResidue.calc_coordinates(i, j, k, L, sigL, theta, sig_theta, chi)
        np.testing.assert_array_almost_equal(xyz, [2.96822105, 5.00438571, 1.25930142])

    def test_structure_residue_tyrosine_monomer(self):
        s = self._STRUCTURE_TYROSINE.copy()
        assert s.clashes() == 0
        s2 = s.copy()
        r = s.chains[0].conformers[0].residues[0]
        assert r.get_chi(1) == pytest.approx(-61.755163, abs=0.000001)
        assert r.get_chi(2) == pytest.approx(-78.113452, abs=0.000001)
        r.set_chi(1, -65)
        assert r.get_chi(1) == -65.0
        assert r.get_chi(2) == pytest.approx(-78.113452, abs=0.000001)
        assert s2.rmsd(s) == pytest.approx(0.144069, abs=0.000001)
        r.set_chi(2, -71)
        assert r.get_chi(1) == -65.0
        assert r.get_chi(2) == -71.0
        assert s2.rmsd(s) == pytest.approx(0.157135, abs=0.000001)
        r.set_chi(1, 65)
        assert s2.rmsd(s) == pytest.approx(4.429105, abs=0.000001)
        assert s.clashes() == 0
        mc = s.extract("name", ("C", "CA", "N", "O")).copy()
        mc2 = s2.extract("name", ("C", "CA", "N", "O")).copy()
        assert _simple_rmsd(mc, mc2) == 0
        with pytest.raises(KeyError):
            chi = r.get_chi(3)
        with pytest.raises(KeyError):
            r.set_chi(3, 120)
        r.set_chi(1, -61.755163)
        r.set_chi(2, -78.113452)
        assert s2.rmsd(s) == pytest.approx(0, abs=1e-12)

    def test_structure_residue_alanine(self):
        s = self._STRUCTURE_ARA_SINGLE.copy()
        r = s.chains[0].conformers[0].residues[0]
        with pytest.raises(KeyError):
            chi = r.get_chi(1)
        assert r.clashes() == 0
        sel = r.get_named_atom_selection(["C", "CA", "CB"])
        assert list(sel) == [2, 1, 4]

    def test_structure_residue_internal_add_atom(self):
        pdb_file = op.join(self.DATA, "AKA_single.pdb")
        s = Structure.fromfile(pdb_file)
        r_orig = s.chains[0].conformers[0].residues[1]
        assert list(r_orig.selection) == [5, 6, 7, 8, 9, 10, 11, 12, 13]
        ss = s.extract("name", ("C", "CA", "N", "O", "CB")).copy()
        r = ss.chains[0].conformers[0].residues[1]
        assert list(r.selection) == [5, 6, 7, 8, 9]
        assert r.resname == "LYS"
        assert np.all(r.name == ['N', 'CA', 'C', 'O', 'CB'])
        new_names = ['CG', 'CD', 'CE', 'NZ']
        for name, coor in zip(new_names, r_orig.coor[5:]):
            r._add_atom(name, name[0], coor)  # pylint: disable=protected-access
        assert list(r.selection) == [5, 6, 7, 8, 9, 10, 11, 12, 13]
        assert r.natoms == r_orig.natoms
        assert np.all(r.name == r_orig.name)
        assert r.rmsd(r_orig) == 0

    def test_structure_residue_arginine(self):
        s = self._STRUCTURE_ARA_SINGLE.copy()
        s2 = s.copy()
        r = s.chains[0].conformers[0].residues[1]
        assert r.get_chi(1) == pytest.approx(-67.148831, abs=0.000001)
        assert r.get_chi(2) == pytest.approx(-177.977424, abs=0.000001)
        assert r.get_chi(3) == pytest.approx(-176.992967, abs=0.000001)
        assert r.get_chi(4) == pytest.approx(173.942592, abs=0.000001)
        assert r.clashes() == 0
        with pytest.raises(KeyError):
            chi = r.get_chi(5)
        r.set_chi(1, 120)
        r.set_chi(2, 0)
        r.set_chi(3, -90)
        assert r.clashes() == 1
        assert s2.rmsd(s) == pytest.approx(4.183182, abs=0.000001)
        r.set_chi(4, 60)
        assert r.clashes() == 2
        mc = s.extract("name", ("C", "CA", "N", "O")).copy()
        mc2 = s2.extract("name", ("C", "CA", "N", "O")).copy()
        assert _simple_rmsd(mc, mc2) == 0
        r.set_chi(1, -67.148831)
        r.set_chi(2, -177.977424)
        r.set_chi(3, -176.992967)
        r.set_chi(4, 173.942592)
        # XXX we lose some precision here relative to Tyr with 2 chi angles
        assert s2.rmsd(s) == pytest.approx(0, abs=1e-7)
        # test sidechain completion
        MC_NAMES = ['N', 'CA', 'C', 'O', 'CB']
        SC_NAMES = ['CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2']
        s2 = s2.extract("name", MC_NAMES).copy()
        s2.tofile("ara_backbone.pdb")
        s2 = Structure.fromfile("ara_backbone.pdb")
        assert s2.total_length == 15
        r = s2.chains[0].conformers[0].residues[1]
        assert np.all(r.name == MC_NAMES)
        r.complete_residue()
        assert np.all(r.name == MC_NAMES + SC_NAMES)
        # this is how qfit_protein calls it
        s3 = r.get_rebuilt_structure()
        r3 = s3.chains[0].conformers[0].residues[1]
        r3c = r3.copy()
        r3r = r3c.extract("resi 2 and chain A")
        s3.tofile("all_rebuilt.pdb")
        rebuilt2 = Structure.fromfile("all_rebuilt.pdb")
        assert np.all(rebuilt2.name == s.name)
        # check behavior when combined with existing selection
        s2 = Structure.fromfile("ara_backbone.pdb")
        rs = s2.extract("chain A and resi 2").copy()
        r = rs.chains[0].conformers[0].residues[0]
        r.complete_residue()
        assert np.all(r.name == MC_NAMES + SC_NAMES)
        # save, reload, set chi angles, and compare to original sidechain
        # XXX note that the original parent Structure is not automatically
        # updated and will never see the new atoms - this is ok in the context
        # of how the rebuilding is actually used but might be confusing
        # behavior
        r.tofile("arg_rebuilt.pdb")
        s.extract("chain A and resi 2").copy().tofile("arg_start.pdb")
        arg_rebuilt = Structure.fromfile("arg_rebuilt.pdb")
        arg_start = Structure.fromfile("arg_start.pdb")
        assert arg_rebuilt.total_length == arg_start.total_length
        r = arg_rebuilt.chains[0].conformers[0].residues[0]
        r.set_chi(1, -67.148831)
        r.set_chi(2, -177.977424)
        r.set_chi(3, -176.992967)
        r.set_chi(4, 173.942592)
        # need to allow for slight differences in geometry here
        assert arg_rebuilt.rmsd(arg_start) == pytest.approx(0, abs=0.002)

    def test_structure_residue_complete_residue_sidechains(self):
        s_in = self._STRUCTURE_ARA_SINGLE.extract("resi", 1).copy()
        s_in.tofile("ala_tmp.pdb")
        residue_natoms = {
            "ARG": 11,
            "ASN": 8,
            "ASP": 8,
            "CYS": 6,
            "GLU": 9,
            "GLN": 9,
            "HIS": 10,
            "ILE": 8,
            "LEU": 8,
            "LYS": 9,
            "MET": 8,
            "PHE": 11,
            "SER": 6,
            "THR": 7,
            "TRP": 14,
            "TYR": 12,
            "VAL": 7
        }
        ala_pdb = open("ala_tmp.pdb", "rt", encoding="ascii").read()
        for resname, natoms in residue_natoms.items():
            suffix = f"{resname}_backbone.pdb"
            pdb_str = ala_pdb.replace("ALA", resname)
            pdb_tmp = self._write_tmp_pdb(pdb_str, suffix)
            s = Structure.fromfile(pdb_tmp)
            r = s.chains[0].conformers[0].residues[0]
            r.complete_residue()
            assert len(r.name) == natoms
            # TODO some validation of geometry would be helpful
