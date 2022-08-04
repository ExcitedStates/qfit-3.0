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

from .test_qfit_protein import TemporaryDirectoryRunner

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


class TestQFitModelIO(TemporaryDirectoryRunner):
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "example")
    CUSTOM_ARGS = [
        "--backbone-amplitude", "0.10",  # default: 0.30
        "--rotamer-neighborhood", "30",  # default: 60
    ]

    def _run_qfit(self, args):
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
        return prepare_qfit_protein(options)

    def test_qfit_model_io(self):
        # Prepare args
        args = [
            os.path.join(self.data_dir, "1fnt_phases.mtz"),
            os.path.join(self.data_dir, "1fnt.pdb"),
            "-l", "FWT,PHWT",
        ] + self.CUSTOM_ARGS
        qfit = self._run_qfit(args)

        # Reach into QFitProtein job,
        # Test 1: chain J and j present in structure. Only J chain has negative residues
        chain="J"
        qfit.structure = qfit.structure.extract('chain', chain, '==')
        qfit.structure = qfit.structure.extract('resi',(-7,-5,-3,-2),"==")
        assert(len(list(qfit.structure.single_conformer_residues))) == 4
        #output the extracted residues to file
        qfit.structure.tofile("j_neg_residues.pdb")

        #Read the written file again and check
        args = [
            os.path.join(self.data_dir, "1fnt_phases.mtz"),
            "j_neg_residues.pdb",
            "-l", "FWT,PHWT",
        ] + self.CUSTOM_ARGS

        qfit = self._run_qfit(args)
        qfit.structure = qfit.structure.extract('resi',(-3,-2),"==")
        assert(len(list(qfit.structure.single_conformer_residues))) == 2

    def test_four_digit_residues(self):
        """Check if 4 digit residues are read correctly"""
        args = [
            os.path.join(self.data_dir, "4e3y_phases.mtz"),
            os.path.join(self.data_dir, "4e3y.pdb"),
            "-l", "FWT,PHWT",
        ] + self.CUSTOM_ARGS

        qfit = self._run_qfit(args)
        # Reach into QFitProtein job,
        # simplify to only run on two residues (reduce computational load)
        qfit.structure = qfit.structure.extract('resi',(1024,1025),"==")
        assert(len(list(qfit.structure.single_conformer_residues))) == 2

    def test_hetatm_in_between_atoms(self):
        """Read PDB file with HETATM record appearing in between ATOM records"""
        args = [
            os.path.join(self.data_dir, "5orl_phases.mtz"),
            os.path.join(self.data_dir, "5orl.pdb"),
            "-l", "FWT,PHWT",
            "--backbone-amplitude", "0.10",  # default: 0.30
            "--rotamer-neighborhood", "30",  # default: 60
        ]

        qfit = self._run_qfit(args)

        # Residue 288 is marked as HETATM in the PDB file since it is a modified residue
        qfit.structure = qfit.structure.extract('resi',(287,288,289),"==")
        assert(np.unique(qfit.structure.resi)[1]) == 288

    @pytest.mark.skip("CCTBX appears to use the wrong CIF field for chain ID")
    def test_cryoem_mmcif(self):
        """Read mmCIF file and map file of cryoEM structure."""
        args = [
            os.path.join(self.data_dir, "7o9m.map"),
            os.path.join(self.data_dir, "7o9m.cif"),
            "--resolution", "2.6",
        ] + self.CUSTOM_ARGS

        qfit = self._run_qfit(args)
        chain='FA'
        # Reach into QFitProtein job,
        # simplify to only run on two residues (reduce computational load)
        qfit.structure = qfit.structure.extract('chain', chain, '==')
        qfit.structure = qfit.structure.extract('resi',(130,134),"==")
        assert(len(list(qfit.structure.single_conformer_residues))) == 2
