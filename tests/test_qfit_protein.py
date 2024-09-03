import pytest
import os
import logging
import multiprocessing as mp
import numpy as np

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


class TestQFitProtein:
    def mock_main(self):
        # Prepare args
        args = [
            "./tests/basic_qfit_protein_test/composite_omit_map.mtz",  # mapfile, using relative directory from tests/
            "./tests/basic_qfit_protein_test/1G8A_refined.pdb",  # structurefile, using relative directory from tests/
            "-l",
            "2FOFCWT,PH2FOFCWT",
        ]

        # Add options to reduce computational load
        args.extend(
            [
                "--backbone-amplitude",
                "0.10",  # default: 0.30
                "--rotamer-neighborhood",
                "30",  # default: 60
            ]
        )

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
        qfit.structure = qfit.structure.extract("resi", (58, 69, 175), "==")  #leucine and phe and met 
        qfit.structure = qfit.structure.reorder()
        assert len(list(qfit.structure.single_conformer_residues)) == 3
        return qfit

    def test_run_qfit_residue_parallel(self):
        qfit = self.mock_main()

        # Run qfit object
        multiconformer = qfit._run_qfit_residue_parallel()
        mconformer_list = list(multiconformer.residues)
        print(mconformer_list)  # If we fail, this gets printed.
        for residue in mconformer_list:
            altlocs = residue.altloc
            if len(altlocs) > 1:
                print(f"RMSD calculations for {residue.resn[0]}{residue.resi[0]}:")
                for i, altloc1 in enumerate(altlocs):
                    for j, altloc2 in enumerate(altlocs[i+1:], start=i+1):
                        coords1 = residue.extract('altloc', altloc1).coor[0]
                        coords2 = residue.extract('altloc', altloc2).coor[0]
                        print(coords1)
                        print(coords2)
                        rmsd = np.sqrt(np.mean(np.sum((coords1 - coords2)**2, axis=1)))
                        print(f"  RMSD between altloc {altloc1} and {altloc2}: {rmsd:.3f} Å")
            else:
                print(f"{residue.resn[0]}{residue.resi[0]} has only one altloc.")
        assert len(mconformer_list) == 5  # Expect: 2*Leu58, 1*Phe69 2*Met175
