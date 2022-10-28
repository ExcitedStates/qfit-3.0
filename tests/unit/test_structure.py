"""
Unit tests for structure I/O and basic manipulation
"""

import tempfile
import unittest

from qfit.structure import Structure
from qfit.structure.pdbfile import read_pdb


class TestStructure(unittest.TestCase):
    PDB = """\
REMARK   2 RESOLUTION.    1.39 ANGSTROMS.                                       
CRYST1   43.096   52.592   89.249  90.00  90.00  90.00 P 21 21 21    4          
ORIGX1      1.000000  0.000000  0.000000        0.00000                         
ORIGX2      0.000000  1.000000  0.000000        0.00000                         
ORIGX3      0.000000  0.000000  1.000000        0.00000                         
SCALE1      0.023204  0.000000  0.000000        0.00000                         
SCALE2      0.000000  0.019014  0.000000        0.00000                         
SCALE3      0.000000  0.000000  0.011205        0.00000    
ATOM   2098  N  APHE A 113       3.050  10.553  10.300  0.63  9.41           N
ANISOU 2098  N  APHE A 113      962    910   1702    -88    638   -737       N
ATOM   2099  N  BPHE A 113       3.114  10.573   9.945  0.37 22.48           N
ANISOU 2099  N  BPHE A 113     5220   1712   1610   -675  -1633    540       N
ATOM   2100  CA APHE A 113       1.808   9.838  10.309  0.63 11.09           C
ANISOU 2100  CA APHE A 113     1465   1150   1600   -684    354   -298       C
ATOM   2101  CA BPHE A 113       1.930   9.754   9.751  0.37 14.35           C
ANISOU 2101  CA BPHE A 113     3095   1060   1297   1211    639   -297       C
ATOM   2102  C  APHE A 113       2.041   8.376  10.478  0.63 11.82           C
ANISOU 2102  C  APHE A 113     2521    570   1402   -319    862    125       C
ATOM   2103  C  BPHE A 113       2.130   8.398  10.395  0.37 14.01           C
ANISOU 2103  C  BPHE A 113     2314    729   2280   -565   -933   -687       C
ATOM   2104  O  APHE A 113       3.080   7.811  10.121  0.63 16.73           O
ANISOU 2104  O  APHE A 113     3335   1125   1896    878    357    393       O
ATOM   2105  O  BPHE A 113       3.210   7.818  10.321  0.37 13.49           O
ANISOU 2105  O  BPHE A 113      681   2360   2085     12    612   -545       O
ATOM   2106  CB APHE A 113       0.944  10.142   9.075  0.63 12.51           C
ANISOU 2106  CB APHE A 113     1890   1458   1405     22    144     18       C
ATOM   2107  CB BPHE A 113       1.674   9.433   8.273  0.37 10.84           C
ANISOU 2107  CB BPHE A 113     1116    851   2152    113    193   -723       C
ATOM   2108  CG APHE A 113       1.639   9.916   7.759  0.63 14.16           C
ANISOU 2108  CG APHE A 113     2543   1293   1546  -1028    -62    309       C
ATOM   2109  CG BPHE A 113       1.607  10.614   7.353  0.37 22.10           C
ANISOU 2109  CG BPHE A 113     4156   1061   3180  -2006     47    -61       C
ATOM   2110  CD1APHE A 113       2.401  10.903   7.191  0.63 13.06           C
ANISOU 2110  CD1APHE A 113     2486    875   1601  -1062   -918     52       C
ATOM   2111  CD1BPHE A 113       0.953  11.788   7.697  0.37 19.63           C
ANISOU 2111  CD1BPHE A 113     4450   1592   1418   1532    -61   -100       C
ATOM   2112  CD2APHE A 113       1.486   8.716   7.066  0.63 14.36           C
ANISOU 2112  CD2APHE A 113     2374   1406   1676   -407     45      7       C
ATOM   2113  CD2BPHE A 113       2.157  10.507   6.090  0.37 15.99           C
ANISOU 2113  CD2BPHE A 113      947   2477   2651    -58  -1203  -1405       C
ATOM   2114  CE1APHE A 113       3.016  10.713   5.961  0.63 14.46           C
ANISOU 2114  CE1APHE A 113     1372   2530   1592  -1334   -261    -94       C
ATOM   2115  CE1BPHE A 113       0.901  12.840   6.805  0.37 22.28           C
ANISOU 2115  CE1BPHE A 113     1398   2456   4612    618    186   -554       C
ATOM   2116  CE2APHE A 113       2.099   8.521   5.837  0.63 17.54           C
ANISOU 2116  CE2APHE A 113     3209   1746   1709    -57    313    219       C
ATOM   2117  CE2BPHE A 113       2.107  11.556   5.201  0.37 26.31           C
ANISOU 2117  CE2BPHE A 113     2550   6024   1424   2383   -550   -441       C
ATOM   2118  CZ APHE A 113       2.873   9.515   5.299  0.63 14.68           C
ANISOU 2118  CZ APHE A 113     1691   2253   1633   -535    -29     94       C
ATOM   2119  CZ BPHE A 113       1.464  12.720   5.554  0.37 24.99           C
ANISOU 2119  CZ BPHE A 113     4789   3138   1569  -2177   -226    759       C
ATOM   2120  H  APHE A 113       3.683  10.197   9.839  0.63 11.17           H
ATOM   2121  H  BPHE A 113       3.796  10.297   9.501  0.37 26.86           H
ATOM   2122  HA APHE A 113       1.297  10.136  11.091  0.63 13.19           H
ATOM   2123  HA BPHE A 113       1.140  10.188  10.136  0.37 17.10           H
ATOM   2124  HB2APHE A 113       0.161   9.571   9.096  0.63 14.89           H
ATOM   2125  HB2BPHE A 113       2.390   8.859   7.957  0.37 12.89           H
ATOM   2126  HB3APHE A 113       0.672  11.072   9.108  0.63 14.89           H
ATOM   2127  HB3BPHE A 113       0.829   8.962   8.204  0.37 12.89           H
ATOM   2128  HD1APHE A 113       2.504  11.714   7.634  0.63 15.55           H
ATOM   2129  HD1BPHE A 113       0.567  11.874   8.539  0.37 23.44           H
ATOM   2130  HD2APHE A 113       0.964   8.039   7.431  0.63 17.11           H
ATOM   2131  HD2BPHE A 113       2.589   9.721   5.844  0.37 19.07           H
ATOM   2132  HE1APHE A 113       3.542  11.387   5.595  0.63 17.23           H
ATOM   2133  HE1BPHE A 113       0.471  13.629   7.045  0.37 26.62           H
ATOM   2134  HE2APHE A 113       2.002   7.711   5.391  0.63 20.93           H
ATOM   2135  HE2BPHE A 113       2.492  11.473   4.358  0.37 31.46           H
ATOM   2136  HZ APHE A 113       3.289   9.386   4.477  0.63 17.50           H
ATOM   2137  HZ BPHE A 113       1.431  13.433   4.957  0.37 29.87           H
"""

    def _make_tmp_pdb(self):
        pdb_tmp = tempfile.NamedTemporaryFile(suffix=".pdb").name
        with open(pdb_tmp, "wt") as pdb_out:
            pdb_out.write(self.PDB)
        return pdb_tmp

    def test_pdbfile(self):
        pdb_tmp = self._make_tmp_pdb()
        pdb = read_pdb(pdb_tmp)
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
                == "UnitCell(a=43.096000, b=52.592000, c=89.249000, alpha=90.000000, beta=90.000000, gamma=90.000000)"
            )

        pdb_tmp = self._make_tmp_pdb()
        s1 = Structure.fromfile(pdb_tmp)
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
        print(pdb_tmp)
        s1.tofile(pdb_tmp_out)
        s2 = Structure.fromfile(pdb_tmp_out)
        assert str(s2.unit_cell) == str(s1.unit_cell)
        _check_structure(s2)
