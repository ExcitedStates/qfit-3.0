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


class TestQfitLigandSyntheticData(SyntheticMapRunner):
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

    def _get_built_dihedrals(self, pdb_single, atom_names):
        assert len(atom_names) == 4
        pdb_in = any_file(pdb_single)
        pdb_out = any_file("multiconformer_ligand_bound_with_protein.pdb")
        atoms = pdb_out.file_object.hierarchy.atoms()
        n_confs = len(atoms) // len(pdb_in.file_object.hierarchy.atoms())
        sel_cache = pdb_out.file_object.hierarchy.atom_selection_cache()
        dihedrals = []
        for i in range(n_confs):
            altloc = string.ascii_uppercase[i]
            points = []
            for name in atom_names:
                isel = sel_cache.iselection(f"altloc {altloc} AND name {name}")
                assert len(isel) == 1
                points.append(atoms[isel[0]].xyz)
            angle = dihedral_angle(points)
            dihedrals.append(angle)
        return dihedrals

    def _validate_ppi(self, pdb_single, fmodel_in, d_min):
        dihedrals = self._get_built_dihedrals(pdb_single,
                                              ["O1", "C1", "C2", "C3"])
        print(dihedrals)
        n_confs = len(dihedrals)
        n_angles = [0,0]
        for angle in dihedrals:
            if abs(angle) < 10:
                n_angles[0] = n_angles[0] + 1
            elif abs(angle) > 170:
                n_angles[1] = n_angles[1] + 1
        assert n_angles[0] > 0 and n_angles[1] > 0 and sum(n_angles) == n_confs
        self._validate_new_fmodel(fmodel_in, d_min, expected_correlation=0.99)

    @pytest.mark.skip(reason="No longer working with RDKit")
    def test_qfit_ligand_ppi_p1(self):
        """
        Build alternate conformers for a propanoic acid (PPI) molecule in
        a P1 cell.  The expected model has a second conformer with the C3
        atom flipped 180 degrees.
        """
        d_min = 1.2
        (pdb_multi, pdb_single) = self._get_start_models("PPI")
        fmodel_in = self._run_qfit_ligand(pdb_multi, pdb_single, "A,1",
                                          "CCC(=O)O", d_min)
        self._validate_ppi(pdb_single, fmodel_in, d_min)

    #@pytest.mark.slow
    @pytest.mark.skip(reason="FIXME needs more debugging")
    def test_qfit_ligand_trs_p21(self):
        """
        Build alternate conformers for a tris (TRS) molecule in a P21 cell.
        The expected model has a second position for the O2 atom.
        """
        d_min = 1.2
        (pdb_multi, pdb_single) = self._get_start_models("TRS")
        fmodel_in = self._run_qfit_ligand(pdb_multi, pdb_single, "A,1",
                                          "C(C(CO)(CO)[NH3+])O", d_min)
        self._validate_new_fmodel(fmodel_in, d_min, expected_correlation=0.987)
        dihedrals = self._get_built_dihedrals(pdb_single, ["N", "C", "C2", "O2"])
        # brute force check for second O2 conformation
        # FIXME energy minimize these models!
        print(dihedrals)
        n_confs = len(dihedrals)
        n_angles = [0,0]
        for angle in dihedrals:
            if 160 < angle < 180:
                n_angles[0] = n_angles[0] + 1
            elif -50 > angle > -75:
                n_angles[1] = n_angles[1] + 1
        assert n_angles[0] > 0 and n_angles[1] > 0 and sum(n_angles) == n_confs

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

    def test_qfit_ligand_zxi_p21(self):
        """
        Build alternate conformers for a 4-Iodobenzylamine (ZXI) molecule in
        a P21 cell.  The expected model has a second position for the N15 atom.
        """
        d_min = 1.3
        (pdb_multi, pdb_single) = self._get_start_models("ZXI")
        fmodel_in = self._run_qfit_ligand(pdb_multi, pdb_single, "A,1",
            "c1cc(ccc1CN)I", d_min)
        self._validate_new_fmodel(fmodel_in, d_min, expected_correlation=0.99)
        s = self._validate_zxi_conformers()
        self._validate_write_intermediate_conformers_zxi()

    @pytest.mark.slow
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


@pytest.mark.slow
@POST_MERGE_ONLY
class TestQfitLigandSyntheticDataLegacy(TestQfitLigandSyntheticData):
    TRANSFORMER = "qfit"
