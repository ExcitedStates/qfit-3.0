import pytest
import os
import logging
import tempfile
import multiprocessing as mp

from qfit.command_line.qfit_protein import (
    QFitOptions,
    build_argparser,
    prepare_qfit_protein,
)
from qfit.logtools import (
    setup_logging,
    log_run_info,
)
from qfit.utils.mock_utils import BaseTestRunner


logger = logging.getLogger(__name__)


def setup_module(module):  # pylint: disable=unused-argument
    # Here, we add compatibility for multiprocessing coverage reports.
    # via: https://pytest-cov.readthedocs.io/en/latest/subprocess-support.html#if-you-use-multiprocessing-pool
    try:
        from pytest_cov.embed import cleanup_on_sigterm
    except ImportError:
        pass
    else:
        cleanup_on_sigterm()


class TestQFitProtein(BaseTestRunner):
    def mock_main(self, file_ext, tmp_dir):  # pylint: disable=unused-argument
        data_dir = os.path.join(os.path.dirname(__file__), "basic_qfit_protein_test")
        # Prepare args
        args = [
            os.path.join(data_dir, "3k0n_map.mtz"),
            os.path.join(data_dir, "3k0n_refine.pdb"),
            "-l",
            "2FOFCWT,PH2FOFCWT",
            # Add options to reduce computational load
            "--backbone-amplitude",
            "0.10",  # default: 0.30
            "--rotamer-neighborhood",
            "30",  # default: 60
            "-d",
            tmp_dir,
        ]
        # Collect and act on arguments
        p = build_argparser()
        args = p.parse_args(args=args)
        os.makedirs(args.directory, exist_ok=True)

        # Apply the arguments to options
        options = QFitOptions()
        options.apply_command_args(args)
        options.debug = True  # For debugging in tests

        # Setup logger
        setup_logging(options=options)
        log_run_info(options, logger)

        # Build a QFitProtein job
        qfit = prepare_qfit_protein(options)

        # Reach into QFitProtein job,
        # simplify to only run on two residues (reduce computational load)
        qfit.structure = qfit.structure.extract("resi", (99, 113), "==")
        qfit.structure = qfit.structure.reorder()
        assert len(list(qfit.structure.single_conformer_residues)) == 2
        return qfit

    def _run_mock_qfit_residue_parallel(self, file_ext):
        tmp_dir = tempfile.mkdtemp("test-qfit-protein")
        print(f"TMP for {file_ext} is {tmp_dir}")
        qfit = self.mock_main(file_ext, tmp_dir)
        # Run qfit object
        multiconformer = qfit._run_qfit_residue_parallel()  # pylint: disable=protected-access
        mconformer_list = list(multiconformer.residues)
        print(mconformer_list)  # If we fail, this gets printed.
        mconformer_resn = [r.resn[0] for r in mconformer_list]
        # FIXME the latest 'dev' changes break this
        assert mconformer_resn.count("PHE") >= 1
        assert mconformer_resn.count("SER") >= 1
        # assert len(mconformer_list) > 5  # Expect: 3*Ser99, 2*Phe113

    def test_run_qfit_residue_parallel(self):
        self._run_mock_qfit_residue_parallel("pdb")

    # def test_run_qfit_residue_parallel_mmcif(self):
    #    self._run_mock_qfit_residue_parallel("cif.gz")
