"""
Utilities for writing unit and integration tests, especially related to
synthetic map generation.
"""

from collections import defaultdict
import unittest
import tempfile
import logging
import os.path as op
import os

from mmtbx.command_line import fmodel
from iotbx.file_reader import any_file
from cctbx.crystal import symmetry
from scitbx.array_family import flex
from libtbx.utils import null_out

from qfit.structure import Structure

WATER = """\
REMARK 1 WATER MOLECULE (NO HYDROGEN) IN P1 BOX
REMARK 1 FOR TESTING BASIC CRYSTALLOGRAPHIC METHODS
CRYST1    4.000    5.000    6.000  90.00  90.00  90.00 P 1           1
HETATM    1  O   HOH A   1       2.000   2.500   3.000  1.00 10.00           O
END"""

SERINE_MULTICONF = """\
CRYST1    6.000    6.000    6.000  90.00  90.00  90.00 P 1
ATOM      1  N  ASER A   1       2.264   2.024   3.928  0.50  1.00           N
ATOM      2  N  BSER A   1       2.267   2.022   3.930  0.50  1.00           N
ATOM      3  CA ASER A   1       3.355   2.711   4.609  0.50  1.00           C
ATOM      4  CA BSER A   1       3.356   2.716   4.606  0.50  1.00           C
ATOM      5  C  ASER A   1       4.708   2.300   4.035  0.50  1.00           C
ATOM      6  C  BSER A   1       4.710   2.299   4.039  0.50  1.00           C
ATOM      7  O  ASER A   1       5.027   1.114   3.962  0.50  1.00           O
ATOM      8  O  BSER A   1       5.023   1.111   3.963  0.50  1.00           O
ATOM      9  CB ASER A   1       3.313   2.425   6.111  0.50  1.00           C
ATOM     10  CB BSER A   1       3.312   2.447   6.111  0.50  1.00           C
ATOM     11  OG ASER A   1       2.144   2.967   6.701  0.50  1.00           O
ATOM     12  OG BSER A   1       4.392   3.079   6.775  0.50  1.00           O
TER
END"""

SERINE_SINGLE = """\
CRYST1    6.000    6.000    6.000  90.00  90.00  90.00 P 1
ATOM      1  N   SER A   1       2.264   2.024   3.928  1.00  1.00           N
ATOM      2  CA  SER A   1       3.355   2.711   4.609  1.00  1.00           C
ATOM      3  C   SER A   1       4.708   2.300   4.035  1.00  1.00           C
ATOM      4  O   SER A   1       5.027   1.114   3.962  1.00  1.00           O
ATOM      5  CB  SER A   1       3.313   2.425   6.111  1.00  1.00           C
ATOM      6  OG  SER A   1       2.144   2.967   6.701  1.00  2.00           O
TER
END"""

# These are space group:unit cell combos that can be used to replace the
# CRYST1 records in the serine models above.  They have been manually
# inspected in PyMol to confirm lack of atomic clashes with symmetry mates.
# The packing isn't very realistic or optimal but it's sufficient for testing
# sensitivity to map calculation differences.
SERINE_ALTERNATE_SYMMETRY = {
    "P1": (6.5, 7.5, 7.0, 85, 95, 90.5),
    "P21": (6, 10, 9, 90, 100, 90),
    "P4212": (18, 18, 18, 90, 90, 90),
    "P6322": (16, 16, 36, 90, 90, 120),
    "C2221": (15, 12, 15, 90, 90, 90),
    "I212121": (12, 12, 18, 90, 90, 90),
    "I422": (24, 24, 20, 90, 90, 90)
}


class TemporaryDirectoryManager:
    """
    Context manager for running a function/command in a separate temporary
    directory.
    """
    def __init__(self, dirname=None):
        if dirname is None:
            dirname = tempfile.mkdtemp("qfit-test")
        else:
            os.makedirs(dirname, exist_ok=False)
        self._dirname = dirname
        self._cwd = os.getcwd()

    def __enter__(self):
        os.chdir(self._dirname)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        os.chdir(self._cwd)
        return


class BaseTestRunner(unittest.TestCase):
    # allowed difference from rotameric chi*
    CHI_RADIUS = 10

    def setUp(self):
        tmp_dir = tempfile.mkdtemp("qfit_protein")
        print(f"TMP={tmp_dir}")
        self._cwd = os.getcwd()
        os.chdir(tmp_dir)

    def tearDown(self):
        os.chdir(self._cwd)

    def _run_in_tmpdir(self, dirname=None):
        return TemporaryDirectoryManager(dirname)

    def _create_fmodel(self,
                       pdb_file_name,
                       high_resolution,
                       output_file=None):
        if output_file is None:
            output_file = tempfile.NamedTemporaryFile(suffix="-fmodel.mtz").name
        fmodel_args = [
            pdb_file_name,
            f"high_resolution={high_resolution}",
            "r_free_flags_fraction=0.1",
            "output.label=FWT",
            f"output.file_name={output_file}",
        ]
        fmodel.run(args=fmodel_args, log=null_out())
        return output_file

    def _replace_symmetry(self,
                          new_symmetry,
                          pdb_file,
                          output_pdb_file=None):
        if isinstance(new_symmetry, tuple):
            (space_group_symbol, unit_cell) = new_symmetry
            new_symmetry = symmetry(space_group_symbol=space_group_symbol,
                                    unit_cell=unit_cell)
        suffix = "_newsymm.pdb"
        if not output_pdb_file:
            output_pdb_file = op.splitext(op.basename(pdb_file))[0] + suffix
        pdb_in = Structure.fromfile(pdb_file)
        pdb_in.crystal_symmetry = new_symmetry
        pdb_in.tofile(output_pdb_file)
        return output_pdb_file

    def _iterate_symmetry_mate_models(self, pdb_file):
        pdb_in = any_file(pdb_file)
        base = op.splitext(op.basename(pdb_file))[0]
        pdbh = pdb_in.file_object.hierarchy
        xrs = pdb_in.file_object.xray_structure_simple()
        sites_frac = xrs.sites_frac()
        for i_op, rt_mx in enumerate(xrs.space_group().smx()):
            new_sites = flex.vec3_double([rt_mx * xyz for xyz in sites_frac])
            xrs.set_sites_frac(new_sites)
            pdbh.atoms().set_xyz(xrs.sites_cart())
            pdb_new = tempfile.NamedTemporaryFile(
                suffix=f"-{base}_op{i_op}.pdb").name
            pdbh.write_pdb_file(pdb_new, crystal_symmetry=xrs)
            yield op.abspath(pdb_new)

    def _compare_maps(self, mtz_file_1, mtz_file_2, expected_correlation):
        fmodel_1 = any_file(mtz_file_1)
        fmodel_2 = any_file(mtz_file_2)
        array1 = fmodel_1.file_object.as_miller_arrays()[0].data()
        array2 = fmodel_2.file_object.as_miller_arrays()[0].data()
        lc = flex.linear_correlation(flex.abs(array1), flex.abs(array2))
        assert lc.coefficient() >= expected_correlation

    def _get_rotamer(self, residue, chi_radius=CHI_RADIUS):
        # FIXME this is awful, we should replace it with something like
        # mmtbx.rotalyze but I don't have the necessary library data
        if len(residue.rotamers) == 0:
            return None
        chis = [residue.get_chi(i + 1) for i in range(len(residue.rotamers[0]))]
        for rotamer in residue.rotamers:
            delta_chi = [abs(a - b) for a, b in zip(chis, rotamer)]
            if all([x < chi_radius or x > 360 - chi_radius for x in delta_chi]):
                return tuple(rotamer)
        raise ValueError(f"Can't find a rotamer for residue {residue}")

    def _get_model_rotamers(self, file_name, chi_radius=CHI_RADIUS):
        s = Structure.fromfile(file_name)
        rotamers = defaultdict(set)
        for residue in s.residues:
            try:
                rot = self._get_rotamer(residue, chi_radius=chi_radius)
                rotamers[residue.resi[0]].add(rot)
            except (IndexError, ValueError) as e:
                print(e)
        return rotamers

    def _get_water_pdb(self):
        """
        Write a temporary PDB file containing water in a P1 box.
        """
        pdb_tmp = tempfile.NamedTemporaryFile(suffix="-water.pdb").name
        with open(pdb_tmp, "wt") as pdb_out:
            pdb_out.write(WATER)
        return pdb_tmp

    def _get_serine_monomer_inputs(self, crystal_symmetry=None):
        """
        Get a pair of models (two-conformer and single-conformer) for a single
        Ser residue in a P1 box.  Optionally, apply a different symmetry
        (assumed to be compatible with existing model coordinates).

        This is probably the smallest/simplest protein structure that qFit
        can do something relatively interesting with, and the full qfit_protein
        tool should find the second conformer easily and relatively quickly
        at 1.5 Angstrom resolution.
        """
        pdb_multi = tempfile.NamedTemporaryFile(suffix="-ser-multi.pdb").name
        pdb_single = tempfile.NamedTemporaryFile(suffix="-ser-single.pdb").name
        final_pdbs = []
        for file_name, content in zip([pdb_multi, pdb_single],
                                      [SERINE_MULTICONF, SERINE_SINGLE]):
            with open(file_name, "wt") as pdb_out:
                pdb_out.write(content)
                logging.info(f"Wrote {file_name}")
            if crystal_symmetry:
                final_pdbs.append(self._replace_symmetry(
                    crystal_symmetry,
                    file_name))
            else:
                final_pdbs.append(file_name)
        return (final_pdbs[0], final_pdbs[1])

    def _get_serine_monomer_with_symmetry(self, space_group_symbol):
        """
        Create a pair of Ser monomer input files with the specific space
        group, using a precalculated lookup table of unit cell dimensions.
        Only a handful of space groups are supported.  These models are
        used to test the sensitivity of the algorithm to map gridding and
        symmetry operations.
        """
        unit_cell_params = SERINE_ALTERNATE_SYMMETRY.get(space_group_symbol)
        if unit_cell_params is None:
            supported = ", ".join(sorted(list(SERINE_ALTERNATE_SYMMETRY.keys())))
            raise ValueError(f"Can't find pre-calculated unit cell params for space group {space_group_symbol}.  Supported space groups: {supported}")
        symm = (space_group_symbol, unit_cell_params)
        return self._get_serine_monomer_inputs(symm)

    def _get_all_serine_monomer_crystals(self):
        for space_group_symbol in SERINE_ALTERNATE_SYMMETRY.keys():
            yield self._get_serine_monomer_with_symmetry(space_group_symbol)
