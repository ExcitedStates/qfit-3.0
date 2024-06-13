import os
import logging

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
from qfit.utils.mock_utils import BaseTestRunner
from qfit.solvers import (
    available_qp_solvers,
    available_miqp_solvers,
)


logger = logging.getLogger(__name__)


class TestQFitLigand(BaseTestRunner):
    def mock_main(self):
        data_dir = os.path.join(os.path.dirname(__file__), "qfit_ligand_test")
        # Prepare args
        args = [
            os.path.join(data_dir, "5C40_composite_omit_map.mtz"),
            os.path.join(data_dir, "5C40.pdb"),
            "-l",
            "2FOFCWT,PH2FOFCWT",
            "A, 401",  # selection
            "-sm",
            "c1nc(c2c(n1)n(cn2)C3C(C(C(O3)COP(=O)(O)OP(=O)(CP(=O)(O)O)O)O)O)N",
            "-nc",
            "5000"
        ]

        # TODO: Add options to reduce computational load

        # Collect and act on arguments
        p = build_argparser()
        args = p.parse_args(args=args)
        os.makedirs(args.directory, exist_ok=True)

        # Apply the arguments to options
        options = QFitOptions()
        options.apply_command_args(args)
        options.debug = True  # For debugging in tests
        options.qp_solver = next(iter(available_qp_solvers.keys()))
        options.miqp_solver = next(iter(available_miqp_solvers.keys()))
        #options.transformer = "cctbx"

        # Setup logger
        setup_logging(options=options, filename="qfit_ligand.log")
        log_run_info(options, logger)

        # Build a QFitLigand job
        qfit_ligand, chainid, resi, icode, receptor = prepare_qfit_ligand(
            options=options
        )
        assert qfit_ligand.ligand.natoms == 31

        return qfit_ligand

    @pytest.mark.slow
    def test_run_qfit_ligand(self):
        qfit_ligand = self.mock_main()
        # Run qfit object
        qfit_ligand.run()
        conformers = qfit_ligand.get_conformers()
        assert len(conformers) == 2
