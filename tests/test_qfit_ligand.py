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
from qfit.solvers import (
    available_qp_solvers,
    available_miqp_solvers,
)


logger = logging.getLogger(__name__)


class TestQFitLigand:
    def mock_main(self):
        # Prepare args
        args = [
            "./tests/qfit_ligand_test/5C40_composite_omit_map.mtz",  # mapfile, using relative directory from tests/
            "./tests/qfit_ligand_test/5C40.pdb",  # structurefile, using relative directory from tests/
            "-l",
            "2FOFCWT,PH2FOFCWT",
            "A, 401",  # selection
            "-sm",
            "c1nc(c2c(n1)n(cn2)C3C(C(C(O3)COP(=O)(O)OP(=O)(CP(=O)(O)O)O)O)O)N",
            "-nc",
            "10000"
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
        options.qp_solver = next(iter(available_qp_solvers.keys()))
        options.miqp_solver = next(iter(available_miqp_solvers.keys()))

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
