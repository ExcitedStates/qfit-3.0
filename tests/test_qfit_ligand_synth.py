"""
Test qfit_ligand using synthetic structures
"""

import subprocess
import string

from iotbx.file_reader import any_file
import pytest

from qfit.structure.math import dihedral_angle

from .test_qfit_protein_synth import SyntheticMapRunner

class TestQfitLigandSyntheticData(SyntheticMapRunner):

    def _run_qfit_ligand(self,
                         pdb_file_multi,
                         pdb_file_single,
                         selection,
                         high_resolution):
        fmodel_mtz = self._create_fmodel(pdb_file_multi,
                                         high_resolution=high_resolution)
        qfit_args = [
            "qfit_ligand",
            fmodel_mtz,
            pdb_file_single,
            selection,
            "--resolution",
            str(high_resolution),
            "--label",
            "FWT,PHIFWT",
            "--rmsd-cutoff", "0.1",
            "--dihedral-stepsize", "20",
            "--debug",
        ]
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

    @pytest.mark.fast
    def test_qfit_ligand_ppi_p1(self):
        """
        Build alternate conformers for a propanoic acid (PPI) molecule in
        a P1 cell.  The expected model has a second conformer with the C3
        atom flipped 180 degrees.
        """
        d_min = 1.25
        (pdb_multi, pdb_single) = self._get_start_models("PPI")
        fmodel_in = self._run_qfit_ligand(pdb_multi, pdb_single, "A,1", d_min)
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

    def test_qfit_ligand_trs_p21(self):
        """
        Build alternate conformers for a tris (TRS) molecule in a P21 cell.
        The expected model has a second position for the O2 atom.
        """
        d_min = 1.35
        (pdb_multi, pdb_single) = self._get_start_models("TRS")
        fmodel_in = self._run_qfit_ligand(pdb_multi, pdb_single, "A,1", d_min)
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
        self._validate_new_fmodel(fmodel_in, d_min, expected_correlation=0.99)
