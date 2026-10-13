"""
Performance regression tests for qfit_protein hot paths.

Each test times the current implementation against a copy of the previous
implementation (from the equivalence tests) in the same process, so the
asserted speedups are relative and robust to machine speed. Absolute budgets
are deliberately generous. Run with:

    pytest -m perf tests/test_performance.py -s --durations=0

(-s prints the measured timings.)
"""

import os.path as op
import time

import numpy as np
import pytest

from qfit.qfit import (
    _UniqueCoordinateSet,
    _build_unique_fragments,
    _is_new_conformer_unique,
    DEFAULT_RMSD_CUTOFF,
)
from qfit.samplers import BackboneRotator
from qfit.solvers import available_miqp_solvers
from qfit.structure import Structure
from qfit.structure.rotamers import ROTAMERS

from tests.test_solvers import _models_with_duplicates, _reference_find_redundant
from tests.unit.test_qfit_protein_helpers import (
    _reference_unique_fragments,
    _residue_conformers,
)
from tests.unit.test_residue_clash_detection import _reference_clashes
from tests.unit.test_samplers import _reference_backbone_rotate

pytestmark = pytest.mark.perf

TESTS_DIR = op.dirname(__file__)
DATA = op.join(TESTS_DIR, "data")
DATA_BIG = op.join(TESTS_DIR, "basic_qfit_protein_test")


def _best_time(func, repeat=5, number=1):
    """Best-of-`repeat` wall time (seconds) for `number` calls of func."""
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        for _ in range(number):
            func()
        best = min(best, (time.perf_counter() - start) / number)
    return best


def _report(name, new, old=None):
    if old is None:
        print(f"\n[perf] {name}: {new * 1e6:.1f} us")
    else:
        print(f"\n[perf] {name}: {new * 1e6:.1f} us (previous {old * 1e6:.1f} us, "
              f"{old / new:.1f}x faster)")


@pytest.fixture(scope="module")
def protein():
    return Structure.fromfile(op.join(DATA_BIG, "1G8A_refined.pdb")).reorder()


@pytest.fixture(scope="module")
def lys67(protein):
    residues = protein.extract("record", "HETATM", "!=").single_conformer_residues
    return next(r for r in residues if r.id[0] == 67)


def test_residue_self_clash_speed(lys67):
    lys67._init_clash_detection()
    new = _best_time(lys67.clashes, number=200)
    old = _best_time(lambda: _reference_clashes(lys67), number=200)
    _report("RotamerResidue.clashes", new, old)
    assert new < 200e-6
    assert old / new > 3


def test_structure_accessors_speed(lys67):
    # Accessors used to iterate cctbx flex arrays directly, costing
    # 0.04-0.6 ms per call regardless of size.
    for name in ("name", "resn", "altloc", "active"):
        t = _best_time(lambda: getattr(lys67, name), number=200)
        _report(f"residue.{name}", t)
        assert t < 100e-6, name


@pytest.mark.parametrize("solver_class", available_miqp_solvers.values())
def test_redundant_conformer_detection_speed(solver_class):
    models = _models_with_duplicates(n=400, nvoxels=5000)

    def run_new():
        solver_class(np.zeros(models.shape[1]), models).find_redundant_conformers()

    new = _best_time(run_new, repeat=3)
    old = _best_time(lambda: _reference_find_redundant(models), repeat=1)
    _report("find_redundant_conformers (400 x 5000)", new, old)
    assert old / new > 5


def test_backbone_rotator_speed(protein):
    structure = protein.copy()
    segment = max(structure.segments, key=len)
    middle = len(segment) // 2
    window = segment[middle - 3:middle + 4]  # 7 residues, as in backbone sampling
    rotator = BackboneRotator(window)
    torsions = np.random.default_rng(0).uniform(-5, 5, rotator.ndofs)
    new = _best_time(lambda: rotator(torsions), number=100)
    old = _best_time(lambda: _reference_backbone_rotate(rotator, torsions),
                     number=100)
    _report("BackboneRotator call (7 residues)", new, old)
    # ~1.9x in a test process; larger inside qfit, where cctbx calls cost more
    assert old / new > 1.3


def test_unique_coordinate_set_speed():
    rng = np.random.default_rng(0)
    candidates = list(rng.random((3000, 15, 3)) * 5)

    def run_new():
        unique = _UniqueCoordinateSet(DEFAULT_RMSD_CUTOFF)
        for coor in candidates:
            unique.add(coor)

    def run_old():
        accepted = []
        for coor in candidates:
            if _is_new_conformer_unique(coor, accepted, DEFAULT_RMSD_CUTOFF):
                accepted.append(coor)

    new = _best_time(run_new, repeat=3)
    old = _best_time(run_old, repeat=1)
    _report("conformer uniqueness, 3000 conformers", new, old)
    # ~4x at 3000 conformers; the gap grows with the number of conformers
    assert old / new > 2


def test_unique_fragments_speed():
    structure = Structure.fromfile(op.join(DATA, "SKH_multiconf.pdb")).reorder()
    segment = [c for c in _residue_conformers(structure) if len(c) > 1]
    new = _best_time(lambda: _build_unique_fragments(segment), repeat=3)
    old = _best_time(lambda: _reference_unique_fragments(segment), repeat=3)
    _report("qfit_segment fragment building (3 x 3 conformers)", new, old)
    assert old / new > 1.5


@pytest.mark.slow
def test_single_residue_end_to_end_time(tmp_path):
    """Sample one residue (LYS67) of the full 1G8A structure.

    Took ~22 s before the performance work and ~9 s after on an Apple M5;
    the budget leaves plenty of room for slower machines.
    """
    from qfit.command_line.qfit_protein import (
        QFitOptions, build_argparser, prepare_qfit_protein, _run_qfit_residue)

    class _NullQueue:
        def put_nowait(self, record):
            pass

    args = build_argparser().parse_args([
        op.join(DATA_BIG, "composite_omit_map.mtz"),
        op.join(DATA_BIG, "1G8A_refined.pdb"),
        "-l", "2FOFCWT,PH2FOFCWT", "-d", str(tmp_path)])
    options = QFitOptions()
    options.apply_command_args(args)
    qfit = prepare_qfit_protein(options)
    residues = qfit.structure.extract("record", "HETATM", "!=") \
        .extract("resn", "HOH", "!=").single_conformer_residues
    residue = next(r for r in residues if r.id[0] == 67)
    assert residue.resn[0] in ROTAMERS
    xmap = qfit.get_map_around_substructure(residue)
    bb = qfit._build_backbone_coor_dict([residue])

    start = time.perf_counter()
    _run_qfit_residue(residue=residue, structure=qfit.structure, xmap=xmap,
                      options=qfit.options, logqueue=_NullQueue(),
                      backbone_coor_dict=bb)
    elapsed = time.perf_counter() - start
    print(f"\n[perf] qfit residue LYS67 (full 1G8A): {elapsed:.1f} s")
    assert op.exists(tmp_path / "A_67" / "multiconformer_residue.pdb")
    assert elapsed < 90
