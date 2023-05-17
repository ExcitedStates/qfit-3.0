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


logger = logging.getLogger(__name__)


class TestQFitLigand:
    def mock_main(self):
        # Prepare args
        args = [
            "./tests/qfit_ligand_test/5AGK_composite_omit_map.mtz",  # mapfile, using relative directory from tests/
            "./tests/qfit_ligand_test/5AGK.pdb",  # structurefile, using relative directory from tests/
            "-l",
            "2FOFCWT,PH2FOFCWT",
            "B, 801",  # selection
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
        qfit_ligand, chainid, resi, icode, receptor = prepare_qfit_ligand(
            options=options
        )
        assert qfit_ligand.ligand.natoms == 15

        return qfit_ligand

    def test_run_qfit_ligand(self):
        qfit_ligand = self.mock_main()

        # Run qfit object

        output = qfit_ligand.run()
        conformers = qfit_ligand.get_conformers()
        assert len(conformers) == 2
