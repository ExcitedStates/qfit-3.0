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

from qfit.structure import Structure, calc_rmsd

WATER = """\
REMARK 1 WATER MOLECULE (NO HYDROGEN) IN P1 BOX
REMARK 1 FOR TESTING BASIC CRYSTALLOGRAPHIC METHODS
CRYST1    4.000    5.000    6.000  90.00  90.00  90.00 P 1           1
HETATM    1  O   HOH A   1       2.000   2.500   3.000  1.00 10.00           O
END"""

SERINE_MULTICONF = """\
CRYST1    6.000    6.000    6.000  90.00  90.00  90.00 P 1
ATOM      1  N  ASER A   1       2.264   2.024   3.928  0.50  8.00           N
ATOM      2  N  BSER A   1       2.267   2.022   3.930  0.50  8.00           N
ATOM      3  CA ASER A   1       3.355   2.711   4.609  0.50  8.00           C
ATOM      4  CA BSER A   1       3.356   2.716   4.606  0.50  8.00           C
ATOM      5  C  ASER A   1       4.708   2.300   4.035  0.50  8.00           C
ATOM      6  C  BSER A   1       4.710   2.299   4.039  0.50  8.00           C
ATOM      7  O  ASER A   1       5.027   1.114   3.962  0.50  8.00           O
ATOM      8  O  BSER A   1       5.023   1.111   3.963  0.50  8.00           O
ATOM      9  CB ASER A   1       3.313   2.425   6.111  0.50  8.00           C
ATOM     10  CB BSER A   1       3.312   2.447   6.111  0.50  8.00           C
ATOM     11  OG ASER A   1       2.144   2.967   6.701  0.50  8.00           O
ATOM     12  OG BSER A   1       4.392   3.079   6.775  0.50  8.00           O
TER
END"""

SERINE_SINGLE = """\
CRYST1    6.000    6.000    6.000  90.00  90.00  90.00 P 1
ATOM      1  N   SER A   1       2.264   2.024   3.928  1.00  8.00           N
ATOM      2  CA  SER A   1       3.355   2.711   4.609  1.00  8.00           C
ATOM      3  C   SER A   1       4.708   2.300   4.035  1.00  8.00           C
ATOM      4  O   SER A   1       5.027   1.114   3.962  1.00  8.00           O
ATOM      5  CB  SER A   1       3.313   2.425   6.111  1.00  8.00           C
ATOM      6  OG  SER A   1       2.144   2.967   6.701  1.00 12.00           O
TER
END"""

TRIMER_TEMPLATE = """\
ATOM      1  N   ALA A   1       0.661   1.882   2.716  1.00 20.00           N
ATOM      2  CA  ALA A   1       2.002   1.825   2.146  1.00 20.00           C
ATOM      3  C   ALA A   1       2.974   2.688   2.942  1.00 20.00           C
ATOM      4  O   ALA A   1       2.902   3.916   2.906  1.00 20.00           O
ATOM      5  CB  ALA A   1       1.976   2.260   0.689  1.00 20.00           C
ATOM      6  N   ALA A   2       3.885   2.037   3.661  1.00 20.00           N
ATOM      7  CA  ALA A   2       4.884   2.713   4.476  1.00 20.00           C
ATOM      8  C   ALA A   2       6.272   2.339   3.980  1.00 20.00           C
ATOM      9  O   ALA A   2       6.604   1.152   3.888  1.00 20.00           O
ATOM     10  CB  ALA A   2       4.731   2.347   5.955  1.00 20.00           C
ATOM     12  N   ALA A   3       7.076   3.348   3.662  1.00 20.00           N
ATOM     13  CA  ALA A   3       8.432   3.124   3.174  1.00 20.00           C
ATOM     14  C   ALA A   3       9.458   3.813   4.067  1.00 20.00           C
ATOM     15  O   ALA A   3       9.520   5.041   4.125  1.00 20.00           O
ATOM     16  CB  ALA A   3       8.566   3.610   1.739  1.00 20.00           C
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


def is_github_pull_request():
    """
    Attempt to detect the branch being run in GitHub CI actions; this allows
    us to mark tests for running post-merge only.
    https://docs.github.com/en/actions/learn-github-actions/variables
    """
    branch_name = os.environ.get("GITHUB_REF", "main").split("/")[-1]
    return branch_name in {"merge"}


def create_fmodel(pdb_file_name, high_resolution, output_file=None,
                  em=False, reference_file=None):
    if output_file is None:
        output_file = tempfile.NamedTemporaryFile(suffix="-fmodel.mtz").name
    fmodel_args = [
        pdb_file_name,
        "r_free_flags_fraction=0.1",
        "output.label=FWT",
        f"output.file_name={output_file}",
    ]
    if reference_file:
        fmodel_args.append(reference_file)
    else:
        fmodel_args.append(f"high_resolution={high_resolution}")
    if em:
        fmodel_args.append("scattering_table=electron")
    # XXX the CLI implementation for mmtbx tools has changed - the old 'run'
    # method no longer exists in the current repo
    if hasattr(fmodel, "run"):
        fmodel.run(args=fmodel_args, log=null_out())
    else:
        from iotbx.cli_parser import run_program
        from mmtbx.programs import fmodel as fmodel_program
        run_program(program_class=fmodel_program.Program,
                    args=fmodel_args,
                    logger=null_out())
    return output_file


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
        print(f"PWD={self._dirname}")
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

    def _create_fmodel(self, *args, **kwds):
        return create_fmodel(*args, **kwds)

    def _replace_symmetry(self,
                          new_symmetry,
                          pdb_file,
                          output_pdb_file=None):
        if isinstance(new_symmetry, tuple):
            (space_group_symbol, unit_cell) = new_symmetry
            new_symmetry = symmetry(space_group_symbol=space_group_symbol,
                                    unit_cell=unit_cell)
        if not output_pdb_file:
            suffix = "_newsymm.pdb"
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

    def _compare_maps(self, mtz_file_1, mtz_file_2, cc_min):
        fmodel_1 = any_file(mtz_file_1)
        fmodel_2 = any_file(mtz_file_2)
        array1 = fmodel_1.file_object.as_miller_arrays()[0].data()
        array2 = fmodel_2.file_object.as_miller_arrays()[0].data()
        lc = flex.linear_correlation(flex.abs(array1), flex.abs(array2))
        cc = lc.coefficient()
        assert cc >= cc_min, f"Bad CC: {cc} < {cc_min}"

    def _get_rotamer(self, residue, chi_radius=CHI_RADIUS):
        # TODO this is awful, we should replace it with something like
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

    def _write_tmp_pdb(self, pdb_str, suffix=""):
        pdb_tmp = tempfile.NamedTemporaryFile(suffix=f"{suffix}.pdb").name
        with open(pdb_tmp, "wt", encoding="ascii") as pdb_out:
            pdb_out.write(pdb_str)
        return pdb_tmp

    def _get_water_pdb(self):
        """
        Write a temporary PDB file containing water in a P1 box.
        """
        return self._write_tmp_pdb(WATER, "-water")

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
            with open(file_name, "wt", encoding="ascii") as pdb_out:
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

    def _get_axa_tripeptide_pdb(self, middle_resname="ALA"):
        return TRIMER_TEMPLATE.replace("ALA A   2", f"{middle_resname} A   2")

    def _write_axa_tripeptide_pdb(self, resname):
        # this is the truncated-sidechain version
        pdb_str = self._get_axa_tripeptide_pdb(resname)
        return self._write_tmp_pdb(pdb_str, f"-{resname}-single")

    def _create_mock_multi_conf_3mer(self, resname, set_b_iso=10):
        """
        Create a tripeptide model AXA where the central residue is rebuilt
        to have two conformations with the most distant rotamers possible,
        as well as the sidechain-free single-conformer starting model.
        """
        # this is the truncated-sidechain version; only the multi-conf pdb
        # has complete sidechains
        pdb_single = self._write_axa_tripeptide_pdb(resname)
        s_single = Structure.fromfile(pdb_single)
        res = s_single.copy().chains[0].conformers[0].residues[1]
        res.complete_residue()
        s = res.get_rebuilt_structure()
        res = s.chains[0].conformers[0].residues[1]
        best_rmsd = 0
        best_pair = []
        for i, angles1 in enumerate(res.rotamers[:-1]):
            res1 = res.copy()
            for k, chi in enumerate(angles1, start=1):
                res1.set_chi(k, chi)
            for _, angles2 in enumerate(res.rotamers[i+1:]):
                res2 = res.copy()
                for k, chi in enumerate(angles2, start=1):
                    res2.set_chi(k, chi)
                rmsd = res1.rmsd(res2)
                if rmsd > best_rmsd:
                    best_rmsd = rmsd
                    best_pair = (res1, res2)
        (res1, res2) = best_pair
        res1.q = 0.5
        res2.q = 0.5
        res1.atoms[0].parent().altloc = "A"
        res2.atoms[0].parent().altloc = "B"
        first_res = s_single.extract("resi 1").copy()
        last_res = s_single.extract("resi 3").copy()
        s_multi = first_res.combine(res1).combine(res2).combine(last_res)
        # this will automatically create a P1 box around the atoms
        xrs = s_multi._pdb_hierarchy.extract_xray_structure()  # pylint: disable=protected-access
        s_multi = s_multi.with_symmetry(xrs.crystal_symmetry())
        s_single = s_single.with_symmetry(xrs.crystal_symmetry())
        # NOTE it is very important that these be the same initial values!
        # the qFit algorithm is very sensitive to initial B-factors
        s_multi.b = set_b_iso
        s_single.b = set_b_iso
        assert s_multi.natoms == 10 + 2 * len(res.name)
        pdb_multi = pdb_single.replace(f"-{resname}-single.pdb",
                                       f"-{resname}-multi.pdb")
        s_multi.tofile(pdb_multi)
        s_single.tofile(pdb_single)
        print(f"RMSD is {best_rmsd}")
        return (pdb_multi, pdb_single)

    def _check_max_rmsd(self,
                        conformers,
                        expected_global_rmsd=0,
                        expected_atom_rmsds=()):
        """
        Check that the maximum pairwise RMSD between conformers (and optionally
        individual atoms) is at least the expected value.
        """
        max_rmsd = 0
        max_rmsd_by_atom = defaultdict(int)
        for i in range(len(conformers) - 1):
            for j in range(i, len(conformers)):
                xyzA = conformers[i].coor
                xyzB = conformers[j].coor
                max_rmsd = max(calc_rmsd(xyzA, xyzB), max_rmsd)
                for name, _ in expected_atom_rmsds:
                    sel = conformers[0].name == name
                    rmsd = calc_rmsd(xyzA[sel], xyzB[sel])
                    max_rmsd_by_atom[name] = max(max_rmsd_by_atom[name], rmsd)
        assert max_rmsd > expected_global_rmsd
        for name, expected_rmsd in expected_atom_rmsds:
            assert max_rmsd_by_atom[name] >= expected_rmsd
