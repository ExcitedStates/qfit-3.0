import logging
import os.path as op
import os

import pytest

from qfit.command_line.qfit_ligand import (
    QFitOptions,
    prepare_qfit_ligand,
    build_argparser,
)
from qfit.logtools import (
    setup_logging,
    log_run_info,
)
from qfit.utils.mock_utils import BaseTestRunner, is_github_pull_request

logger = logging.getLogger(__name__)


SKIP_ME = is_github_pull_request() or \
    os.environ.get("QFIT_DISABLE_LIGAND_TEST", "false").lower() == "true"

@pytest.mark.skipif(SKIP_ME, reason="Skipping post-merge-only ligand tests")
@pytest.mark.slow
class TestQFitLigand(BaseTestRunner):
    DATA = op.join(op.dirname(__file__), "qfit_ligand_test")

    def _mock_main(self, pdb_file_name, mtz_file_name, selection, smiles,
                   extra_args=()):
        # Prepare args
        args = [
            op.join(self.DATA, mtz_file_name),
            op.join(self.DATA, pdb_file_name),
            selection,
            "--smiles", smiles,
            "-l", "2FOFCWT,PH2FOFCWT",
            "--write_intermediate_conformers",
        ] + list(extra_args)
        # TODO: Add options to reduce computational load

        # Collect and act on arguments
        p = build_argparser()
        args = p.parse_args(args=args)
        os.makedirs(args.directory, exist_ok=True)

        # Apply the arguments to options
        options = QFitOptions()
        options.apply_command_args(args)
        options.debug = True  # For debugging in tests

        # Setup logger
        setup_logging(options=options, filename="qfit_ligand.log")
        log_run_info(options, logger)

        # Build a QFitLigand job
        qfit_ligand, chainid, resi, icode, receptor = prepare_qfit_ligand(
            options=options
        )
        return qfit_ligand

    # FIXME this is currently failing with:
    # 'Generated an exception: No matching found'
    def test_run_qfit_ligand_5agk(self):
        qfit_ligand = self._mock_main(
            pdb_file_name="5AGK.pdb",
            mtz_file_name="5AGK_composite_omit_map.mtz",
            selection="B, 801",
            smiles=r"[H]/N=C(\CS(=O)C)/NCCC[C@@H](C(=O)O)N")
        assert qfit_ligand.ligand.natoms == 15
        qfit_ligand.run()
        conformers = qfit_ligand.get_conformers()
        assert len(conformers) == 2

    def test_run_qfit_ligand_5c4o(self):
        qfit_ligand = self._mock_main(
            pdb_file_name="5C40.pdb",
            mtz_file_name="5C40_composite_omit_map.mtz",
            selection="A, 401",
            smiles=r"c1nc(c2c(n1)n(cn2)[C@H]3[C@@H]([C@@H]([C@H](O3)CO[P@@](=O)(O)O[P@](=O)(CP(=O)(O)O)O)O)O)N")
        assert qfit_ligand.ligand.natoms == 31
        qfit_ligand.run()
        conformers = qfit_ligand.get_conformers()
        assert len(conformers) >= 2

    def test_run_qfit_ligand_3nm0(self):
        qfit_ligand = self._mock_main(
            pdb_file_name="3NM0.pdb",
            mtz_file_name="3NM0_composite_omit_map.mtz",
            selection="A,800",
            smiles="C[C@H]1C[C@H](N=C(C1)N)C[C@@H]2CNC[C@@H]2OCCNCC(c3ccccc3)(F)F",
            extra_args=["-nc", "100"])
        assert qfit_ligand.ligand.natoms == 28
        qfit_ligand.run()
        conformers = qfit_ligand.get_conformers()
        assert len(conformers) >= 2
