import logging
import os.path as op
import os

import numpy as np
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
from qfit.utils.mock_utils import BaseTestRunner, is_github_pull_request
from qfit.structure import Structure

from .test_qfit_protein_synth import POST_MERGE_ONLY

logger = logging.getLogger(__name__)


class QfitLigandRunner(BaseTestRunner):
    def _setup_qfit_ligand(self, pdb_in, mtz_in, selection, smiles,
                           transformer,
                           labels="2FOFCWT,PH2FOFCWT",
                           extra_args=()):
        # this allows us to write optional tests using bulky inputs that
        # aren't a part of the core qFit repository
        if not op.isfile(pdb_in):
            pytest.skip(f"{pdb_in} not available")
        if not op.isfile(mtz_in):
            pytest.skip(f"{mtz_in} not available")
        # Prepare args
        args = [
            mtz_in,
            pdb_in,
            selection,
            "--smiles", smiles,
            "-l", labels,
            "--write_intermediate_conformers",
            "--transformer", transformer,
        ] + list(extra_args)

        # Collect and act on arguments
        p = build_argparser()
        args = p.parse_args(args=args)
        os.makedirs(args.directory, exist_ok=True)

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
        return qfit_ligand


@pytest.mark.slow
class TestQFitLigand(QfitLigandRunner):
    DATA = op.join(op.dirname(__file__), "qfit_ligand_test")
    TRANSFORMER = os.environ.get("QFIT_TRANSFORMER", "qfit")

    def _mock_main(self, pdb_file_name, mtz_file_name, selection, smiles,
                   extra_args=(), transformer=None):
        pdb_in = op.join(self.DATA, pdb_file_name)
        mtz_in = op.join(self.DATA, mtz_file_name)
        transformer = self.TRANSFORMER if transformer is None else transformer
        return self._setup_qfit_ligand(
            pdb_in=pdb_in,
            mtz_in=mtz_in,
            selection=selection,
            smiles=smiles,
            transformer=transformer,
            extra_args=extra_args)

    # FIXME this is currently failing with:
    # 'Generated an exception: No matching found'
    @POST_MERGE_ONLY
    def test_run_qfit_ligand_5agk(self):
        qfit_ligand = self._mock_main(
            pdb_file_name="5AGK.pdb",
            mtz_file_name="5AGK_composite_omit_map.mtz",
            selection="B, 801",
            smiles=r"[H]/N=C(\CS(=O)C)/NCCC[C@@H](C(=O)O)N")
        assert qfit_ligand.ligand.natoms == 15
        qfit_ligand.run()
        conformers = qfit_ligand.get_conformers()
        assert len(conformers) == 2

    @POST_MERGE_ONLY
    def test_run_qfit_ligand_5c4o(self):
        qfit_ligand = self._mock_main(
            pdb_file_name="5C40.pdb",
            mtz_file_name="5C40_composite_omit_map.mtz",
            selection="A, 401",
            smiles=r"c1nc(c2c(n1)n(cn2)[C@H]3[C@@H]([C@@H]([C@H](O3)CO[P@@](=O)(O)O[P@](=O)(CP(=O)(O)O)O)O)O)N")
        assert qfit_ligand.ligand.natoms == 31
        qfit_ligand.run()
        conformers = qfit_ligand.get_conformers()
        assert len(conformers) >= 2

    def test_run_qfit_ligand_quick_3nm0(self):
        qfit_ligand = self._mock_main(
            pdb_file_name="3NM0.pdb",
            mtz_file_name="3NM0_composite_omit_map.mtz",
            selection="A,800",
            smiles="C[C@H]1C[C@H](N=C(C1)N)C[C@@H]2CNC[C@@H]2OCCNCC(c3ccccc3)(F)F",
            extra_args=["-nc", "100"])
        assert qfit_ligand.ligand.natoms == 28
        qfit_ligand.run()
        conformers = qfit_ligand.get_conformers()
        assert len(conformers) >= 2

    def _run_qfit_ligand_and_compare(self,
                                     pdb_file_name,
                                     mtz_file_name,
                                     selection,
                                     smiles,
                                     extra_args=(),
                                     n_atoms_ligand=2,
                                     expected_nconfs=2,
                                     expected_global_rmsd=0,
                                     expected_atom_rmsds=()):
        qfit_ligand = self._mock_main(
            pdb_file_name=pdb_file_name,
            mtz_file_name=mtz_file_name,
            selection=selection,
            smiles=smiles)
        assert qfit_ligand.ligand.natoms == n_atoms_ligand
        qfit_ligand.run()
        conformers = qfit_ligand.get_conformers()
        assert len(conformers) >= expected_nconfs
        self._check_max_rmsd(conformers,
                             expected_global_rmsd,
                             expected_atom_rmsds)

    #----------------------------------------------------------------
    # XXX OPTIONAL TESTS, SKIPPED AUTOMATICALLY IF INPUTS NOT PRESENT

    # this should have the double ring flipped
    def test_run_qfit_ligand_6hex(self):
        """6HEX: EphA2 kinase bound to derivative of NVP-BHG712 (G02)"""
        self._run_qfit_ligand_and_compare(
            pdb_file_name="6HEX_001.pdb",
            mtz_file_name="6HEX_composite_omit_map.mtz",
            selection="A,1000",
            smiles="Cc1ccc(cc1Nc2c3cnn(c3nc(n2)c4cnccn4)C)C(=O)Nc5cccc(c5)C(F)(F)F",
            extra_args=["-nc", "10000"],
            n_atoms_ligand=37,
            expected_nconfs=2,
            expected_global_rmsd=2.0,
            expected_atom_rmsds=[("CAF", 4.5)])

    # this should have an adenine ring flip
    def test_run_qfit_ligand_5o3r(self):
        """5O3R: Carbon regulatory protein SbtB bound to AMP"""
        self._run_qfit_ligand_and_compare(
            pdb_file_name="5O3R_001.pdb",
            mtz_file_name="5O3R_composite_omit_map.mtz",
            selection="C,200",
            smiles="c1nc(c2c(n1)n(cn2)C3C(C(C(O3)COP(=O)(O)O)O)O)N",
            extra_args=["-nc", "9000"],
            n_atoms_ligand=23,
            expected_nconfs=2,
            expected_global_rmsd=1.3,
            expected_atom_rmsds=[("C8", 2.5)])

    # XXX very slow, QP step takes >30m
    def test_run_qfit_ligand_5lpl(self):
        """5LPL: bromodomain bound to inhibitor XDM3c (71X)"""
        self._run_qfit_ligand_and_compare(
            pdb_file_name="5LPL_001.pdb",
            mtz_file_name="5LPL_composite_omit_map.mtz",
            selection="A,1201",
            smiles="CCc1c(c([nH]c1C(=O)NC2c3cc(ccc3CCC2O)Cl)C)C(=O)C",
            extra_args=["-nc", "9000"],
            n_atoms_ligand=26,
            expected_nconfs=3,  # TODO check this
            expected_global_rmsd=1.0,
            expected_atom_rmsds=())

    # FIXME debug QP solver discrepancy
    @pytest.mark.skip(reason="FIXME debug CCTBX failure")
    def test_qfit_ligand_solver_5o3r(self):
        """
        Test the performance of the QP and MIQP solvers (as well as the
        underlying density-handling routines) given a pair of known
        AMP conformations in 5O3R.
        """
        MIN_OCC_QP = {
            "qfit": [0.269, 0.148],
            # XXX these numbers are for the CCTBX map summation on the qfit
            # gridding - when CCTBX gridding is used, the occupancies are
            # [0.334, 0.028] which is obviously wrong
            "cctbx": [0.259, 0.134],
        }
        OCC_MIQP = {
            "qfit": [0.221749, 0.2],
            "cctbx": [0.2, 0.2]
        }
        pdb_multi = op.join(self.DATA, "5O3R_multiconformer_ligand_only.pdb")
        multi_conf = Structure.fromfile(pdb_multi)
        for transformer in ["qfit", "cctbx"]:
            qfit_ligand = self._mock_main(
                pdb_file_name="5O3R_001.pdb.gz",
                mtz_file_name="5O3R_composite_omit_map.mtz",
                selection="C,200",
                smiles="c1nc(c2c(n1)n(cn2)C3C(C(C(O3)COP(=O)(O)O)O)O)N",
                transformer=transformer)
                #extra_args=["--no-expand-p1"])
                #extra_args=["--transformer-map-coeffs", "cctbx"])
            qfit_ligand._coor_set = []
            qfit_ligand._bs = []
            for altloc in ["A", "B"]:
                conf = multi_conf.extract("altloc", altloc)
                assert len(conf.coor) == len(qfit_ligand.ligand.coor)
                qfit_ligand._coor_set.append(conf.coor)
                qfit_ligand._bs.append(conf.b)
            qfit_ligand._convert(save_debug_maps_prefix=transformer)
            qfit_ligand._solve_qp()
            min_occ = MIN_OCC_QP[transformer]
            assert np.all(qfit_ligand._occupancies >= min_occ), \
                f"assertion failed for {transformer}"
            qfit_ligand._solve_miqp(
                threshold=qfit_ligand.options.threshold,
                cardinality=qfit_ligand.options.ligand_cardinality)
            np.testing.assert_allclose(qfit_ligand._occupancies,
                                       OCC_MIQP[transformer],
                                       atol=0.000001)
