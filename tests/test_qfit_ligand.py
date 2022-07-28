import os
import logging

from qfit.qfit_ligand import (
    QFitOptions,
    prepare_qfit_ligand,
    build_argparser,
)
from qfit.logtools import (
    setup_logging,
    log_run_info,
)

from .test_qfit_protein import TemporaryDirectoryRunner

logger = logging.getLogger(__name__)


class TestQFitLigand(TemporaryDirectoryRunner):
    def mock_main(self):
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "example")
        # Prepare args
        args = [
            os.path.join(data_dir, "composite_omit_map.mtz"),
            os.path.join(data_dir, "4ms6.pdb"),
            "-l",
            "2FOFCWT,PH2FOFCWT",
            "A, 702",  # selection
        ]

        # TODO: Add options to reduce computational load

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
        setup_logging(options=options, filename="qfit_ligand.log")
        log_run_info(options, logger)

        # Build a QFitLigand job
        qfit_ligand, chainid, resi, icode = prepare_qfit_ligand(options=options)

        assert qfit_ligand.ligand.natoms == 19

        return qfit_ligand

    def test_run_qfit_ligand(self):
        qfit_ligand = self.mock_main()

        # Run qfit object
        # NOTE: Currently, running this qfit_ligand job takes 20-something
        #       minutes on the GitHub Actions runner. This needs to be cut down.

        # output = qfit_ligand.run() # TODO: Determine if you can just run part of this
        # conformers = qfit_ligand.get_conformers()
        # assert len(conformers) == 3  # TODO: fix when qfit_ligand working
