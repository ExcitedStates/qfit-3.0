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
            "./example/qfit_io_test/1fnt_phases.mtz",  # mapfile, using relative directory from tests/
            "./example/qfit_io_test/1fnt.pdb",  # structurefile, using relative directory from tests/
            "-l", "FWT,PHWT",
        ]

        # Add options to reduce computational load
        args.extend([
            "--backbone-amplitude", "0.10",  # default: 0.30
            "--rotamer-neighborhood", "30",  # default: 60
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
        # Test 1: chain J and j present in structure. Only J chain has negative residues
        chain="J"
        qfit.structure = qfit.structure.extract('chain', chain, '==')
        qfit.structure = qfit.structure.extract('resi',(-7,-5,-3,-2),"==")
        assert(len(list(qfit.structure.single_conformer_residues))) == 4
        #output the extracted residues to file
        qfit.structure.tofile("./example/qfit_io_test/j_neg_residues.pdb")

        #Read the written file again and check
        args = [
            "./example/qfit_io_test/1fnt_phases.mtz",  
            "./example/qfit_io_test/j_neg_residues.pdb",  
            "-l", "FWT,PHWT",
        ]

        args.extend([
            "--backbone-amplitude", "0.10",  # default: 0.30
            "--rotamer-neighborhood", "30",  # default: 60
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

        qfit = prepare_qfit_protein(options)
        qfit.structure = qfit.structure.extract('resi',(-3,-2),"==")
        assert(len(list(qfit.structure.single_conformer_residues))) == 2

        #Test 2: Check if 4 digit residues are read correctly
        args = [
            "./example/qfit_io_test/4e3y_phases.mtz",  # mapfile, using relative directory from tests/
            "./example/qfit_io_test/4e3y.pdb",  # structurefile, using relative directory from tests/
            "-l", "FWT,PHWT",
        ]

        # Add options to reduce computational load
        args.extend([
            "--backbone-amplitude", "0.10",  # default: 0.30
            "--rotamer-neighborhood", "30",  # default: 60
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

        qfit = prepare_qfit_protein(options)

        # Reach into QFitProtein job,
        # simplify to only run on two residues (reduce computational load)
        qfit.structure = qfit.structure.extract('resi',(1024,1025),"==")
        assert(len(list(qfit.structure.single_conformer_residues))) == 2

        # Test 3: Read PDB file with HETATM record appearing in between ATOM records

        args = [
            "./example/qfit_io_test/5orl_phases.mtz",  # mapfile, using relative directory from tests/
            "./example/qfit_io_test/5orl.pdb",  # structurefile, using relative directory from tests/
            "-l", "FWT,PHWT",
        ]

        # Add options to reduce computational load
        args.extend([
            "--backbone-amplitude", "0.10",  # default: 0.30
            "--rotamer-neighborhood", "30",  # default: 60
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

        qfit = prepare_qfit_protein(options)

        # Residue 288 is marked as HETATM in the PDB file since it is a modified residue
        qfit.structure = qfit.structure.extract('resi',(287,288,289),"==")
        assert(np.unique(qfit.structure.resi)[1]) == 288

        #Test 4: Read mmCIF file and map file of cryoEM structure. To be uncommented after cctbx integration is complete
        '''
        args = [
            "./example/qfit_io_test/7o9m.map",  # mapfile, using relative directory from tests/
            "./example/qfit_io_test/7o9m.pdb",  # structurefile, using relative directory from tests/
            #"-l", "FWT,PHWT",
        ]

        # Add options to reduce computational load
        args.extend([
            "--backbone-amplitude", "0.10",  # default: 0.30
            "--rotamer-neighborhood", "30",  # default: 60
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

        qfit = prepare_qfit_protein(options)
        chain='FA'
        # Reach into QFitProtein job,
        # simplify to only run on two residues (reduce computational load)
        qfit.structure = qfit.structure.extract('chain', chain, '==')
        qfit.structure = qfit.structure.extract('resi',(130,134),"==")
        assert(len(list(qfit.structure.single_conformer_residues))) == 2
        '''

        return qfit

    def test_run_qfit_model_io(self):
        qfit = self.mock_main()

