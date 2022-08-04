import pytest
import os
import logging
import tempfile
import multiprocessing as mp

from qfit.qfit_protein import (
    QFitOptions,
    build_argparser,
    prepare_qfit_protein,
)
from qfit.logtools import (
    setup_logging,
    log_run_info,
)


logger = logging.getLogger(__name__)


def setup_module(module):
    # Here, we add compatibility for multiprocessing coverage reports.
    # via: https://pytest-cov.readthedocs.io/en/latest/subprocess-support.html#if-you-use-multiprocessing-pool
    try:
        from pytest_cov.embed import cleanup_on_sigterm
    except ImportError:
        pass
    else:
        cleanup_on_sigterm()


class TemporaryDirectoryRunner:

    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(self):
        tmp_dir = tempfile.mkdtemp("qfit_protein")
        print(f"TMP={tmp_dir}")
        cwd = os.getcwd()
        os.chdir(tmp_dir)
        yield
        os.chdir(cwd)


class TestQFitProtein(TemporaryDirectoryRunner):
    def mock_main(self, file_ext, tmp_dir):
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "example")
        # Prepare args
        args = [
            os.path.join(data_dir, "3K0N.mtz"),
            os.path.join(data_dir, f"3K0N.{file_ext}"),
            "-l", "2FOFCWT,PH2FOFCWT",
        ]

        # Add options to reduce computational load
        args.extend([
            "--backbone-amplitude", "0.10",  # default: 0.30
            "--rotamer-neighborhood", "30",  # default: 60
            "-d", tmp_dir,
            "--random-seed", "7"
        ])

        # Collect and act on arguments
        p = build_argparser()
        args = p.parse_args(args=args)
        try:
            os.mkdir(args.directory)
        except OSError:
            pass

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
        multiconformer = qfit._run_qfit_residue_parallel()
        mconformer_list = list(multiconformer.residues)
        print(mconformer_list)  # If we fail, this gets printed.
        mconformer_resn = [r.resn[0] for r in mconformer_list]
        assert mconformer_resn.count("PHE") == 2
        assert mconformer_resn.count("SER") >= 2
        #assert len(mconformer_list) > 5  # Expect: 3*Ser99, 2*Phe113

    def test_run_qfit_residue_parallel(self):
        self._run_mock_qfit_residue_parallel("pdb")

    def test_run_qfit_residue_parallel_mmcif(self):
        self._run_mock_qfit_residue_parallel("cif.gz")
