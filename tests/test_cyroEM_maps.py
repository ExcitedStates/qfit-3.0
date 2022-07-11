import pytest
import os
import logging
import multiprocessing as mp

from qfit.qfit_protein import (
    QFitProteinOptions,
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


class TestQFitProtein:
    def mock_main(self, structure, xmap, resolution):
        # Prepare args
        args = [
            xmap,  # mapfile, using relative directory from tests/
            structure,  # structurefile, using relative directory from tests/
            "-r", resolution,
        ]

        # Add options to reduce computational load
        args.extend([
            "--backbone-amplitude", "0.10",  # default: 0.30
            "--rotamer-neighborhood", "30",  # default: 60
        ])

        # Collect and act on arguments
        p = build_argparser()
        args = p.parse_args(args=args)

        # Apply the arguments to options
        options = QFitProteinOptions()
        options.apply_command_args(args)
        options.debug = True  # For debugging in tests

        # Setup logger
        setup_logging(options=options)
        log_run_info(options, logger)

        # Build a QFitProtein job
        qfit = prepare_qfit_protein(options)

        # Reach into QFitProtein job,
        # simplify to only run on two residues (reduce computational load)
        qfit.structure = qfit.structure.extract("resi", (11,12), "==")
        qfit.structure = qfit.structure.reorder()
        assert len(list(qfit.structure.single_conformer_residues)) == 2

        return qfit

    def test_run_qfit_residue_parallel(self):
        qfit = self.mock_main()

        # Run qfit object
        multiconformer = qfit._run_qfit_residue_parallel()
        mconformer_list = list(multiconformer.residues)
        print(mconformer_list)  # If we fail, this gets printed.
        return len(mconformer_list)   # Expect: one conformer per residue


assert TestQFitProtein('./example/apoF_chainA.pdb', './example/apoF_chainA.ccp4', 1.22) == 2
assert TestQFitProtein('./example/7a4m_modified_box.pdb', './example/7a4m_modified_box.xplor', 1.22) == 2

