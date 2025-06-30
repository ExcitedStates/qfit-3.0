"""
Test qfit_ligand using synthetic structures
"""

import subprocess
import string
import os.path as op
import os

from iotbx.file_reader import any_file
import numpy as np
import pytest

from qfit.structure import Structure
from qfit.structure.math import dihedral_angle

from .test_qfit_protein_synth import SyntheticMapRunner, POST_MERGE_ONLY
from .test_qfit_ligand import QfitLigandRunner


class TestQfitLigandSyntheticData(QfitLigandRunner, SyntheticMapRunner):
    TRANSFORMER = "qfit"
    COMMON_OPTIONS = [
        "--label",
        "FWT,PHIFWT",
        "--rmsd-cutoff", "0.1",
        "--debug",
        "--write_intermediate_conformers"
    ]

    def _run_qfit_ligand(self,
                         pdb_file_multi,
                         pdb_file_single,
                         selection,
                         smiles_string,
                         high_resolution,
                         extra_args=()):
        fmodel_mtz = self._create_fmodel(pdb_file_multi,
                                         high_resolution=high_resolution)
        qfit_args = [
            "qfit_ligand",
            fmodel_mtz,
            pdb_file_single,
            selection,
            "--smiles", smiles_string,
            "--resolution",
            str(high_resolution),
            "--transformer", self.TRANSFORMER,
        ] + self.COMMON_OPTIONS + list(extra_args)
        print(" ".join(qfit_args))
        subprocess.check_call(qfit_args)
        return fmodel_mtz

    def _validate_new_fmodel(
        self,
        fmodel_in,
        high_resolution,
        expected_correlation=0.99,
        model_name="multiconformer_ligand_bound_with_protein.pdb"):
        fmodel_out = self._create_fmodel(model_name,
                                         high_resolution=high_resolution)
        print(op.abspath(model_name))
        self._compare_maps(fmodel_in, fmodel_out, expected_correlation)

    def _check_zxi_n15_dxyz(self, n15_coor):
        deltas = []
        for i, xyz1 in enumerate(n15_coor):
            for j, xyz2 in enumerate(n15_coor):
                if i != j:
                    # the conformational change is a flip along the Z axis
                    deltas.append(np.abs(np.sum(xyz1[2] - xyz2[2])))
        assert np.max(deltas) > 2.5

    def _validate_zxi_conformers(self):
        """
        Check for distinct multiple conformations of ZXI:N15 in the final
        complete model.
        """
        s = Structure.fromfile("multiconformer_ligand_bound_with_protein.pdb")
        lig = s.extract("resn", "ZXI")
        self._check_zxi_n15_dxyz(lig.coor[lig.name == "N15"])
        return s

    def _validate_write_intermediate_conformers_zxi(self):
        """
        Check for distinct multiple conformations of ZXI:N15 in the expected
        intermediate output files.
        """
        for prefix in ["miqp_solution_", "conformer_"]:
            intermediates = []
            for fn in os.listdir("."):
                if fn.startswith(prefix):
                    intermediates.append(Structure.fromfile(fn))
            assert len(intermediates) > 1
            n15_coor = []
            for s in intermediates:
                assert np.all(s.resn == "ZXI")
                sel = s.name == "N15"
                assert np.sum(sel) == 1
                n15_coor.append(s.coor[sel][0])
            self._check_zxi_n15_dxyz(n15_coor)

    def test_qfit_ligand_zxi_complex_p1(self):
        """
        Build alternate conformers for a 4-Iodobenzylamine (ZXI) molecule
        alongside a tripeptide in a P1 cell.  The expected model has a second
        position for the N15 atom.
        """
        d_min = 1.3
        (pdb_multi, pdb_single) = self._get_start_models("ZXI_complex")
        fmodel_in = self._run_qfit_ligand(pdb_multi, pdb_single, "A,10",
            "c1cc(ccc1CN)I", d_min)
        self._validate_new_fmodel(fmodel_in, d_min, expected_correlation=0.989)
        s = self._validate_zxi_conformers()
        protein = s.extract("resn", "ZXI", "!=")
        assert len(protein.coor) == 19
        assert np.all(protein.altloc == "")
        self._validate_write_intermediate_conformers_zxi()

    def test_qfit_ligand_solver_zxi(self):
        """
        Test the behavior of the QP and MIQP solvers in qfit_ligand, using
        the synthetic ZXI examples.
        """
        d_min = 1.0
        # FIXME This should be >0.49 for both conformers, but the cctbx
        # transformer currently loses sensitivity
        MIN_OCC = 0.47
        # the standalone ZXI in P21 behaves similarly; since it's synthetic
        # data anyway the symmetry expansion behavior won't matter as much
        (pdb_multi, pdb_single) = self._get_start_models("ZXI_complex")
        fmodel_mtz = self._create_fmodel(pdb_multi,
                                         high_resolution=d_min)
        multi_conf = Structure.fromfile(pdb_multi)
        for transformer in ["qfit", "cctbx"]:
            qfit_ligand = self._setup_qfit_ligand(
                pdb_in=pdb_single,
                mtz_in=fmodel_mtz,
                selection="A,10",
                smiles="c1cc(ccc1CN)I",
                transformer=transformer,
                labels="FWT,PHIFWT")
            qfit_ligand._coor_set = []
            qfit_ligand._bs = []
            ligand_multi = multi_conf.extract("resn", "ZXI")
            for altloc in ["A", "B"]:
                conf = ligand_multi.extract("altloc", ("", altloc))
                assert len(conf.coor) == len(qfit_ligand.ligand.coor)
                qfit_ligand._coor_set.append(conf.coor)
                qfit_ligand._bs.append(conf.b)
            qfit_ligand._convert(save_debug_maps_prefix=transformer)
            qfit_ligand._solve_qp()
            assert len(qfit_ligand._occupancies) == 2
            assert (np.all(qfit_ligand._occupancies >= MIN_OCC) and
                    np.all(qfit_ligand._occupancies < 0.5)), \
                f"assertion failed for transformer {transformer} after QP"
            qfit_ligand._solve_miqp(
                threshold=qfit_ligand.options.threshold,
                cardinality=qfit_ligand.options._ligand_cardinality)
            assert len(qfit_ligand._occupancies) == 2
            assert (np.all(qfit_ligand._occupancies >= MIN_OCC) and
                    np.all(qfit_ligand._occupancies < 0.5)), \
                f"assertion failed for transformer {transformer} after MIQP"


@pytest.mark.slow
@POST_MERGE_ONLY
class TestQfitLigandSyntheticDataLegacy(TestQfitLigandSyntheticData):
    TRANSFORMER = "qfit"
