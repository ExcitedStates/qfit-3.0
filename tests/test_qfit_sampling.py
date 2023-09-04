"""
Test protein residue sampling methods in qfit.qfit.QFitRotamericResidue
"""

import numpy as np

from qfit.qfit import QFitRotamericResidue, QFitOptions
from qfit.structure import Structure
from qfit.xtal.volume import XMap

from .test_qfit_protein_synth import SyntheticMapRunner

def _get_residue_rmsd(r1, r2):
    return np.sqrt(np.sum(np.power(r2.coor - r1.coor, 2)))


def _get_best_rmsds(reference_confs, new_confs, rmsd_max_same_conf):
    pairs, rmsds = [], []
    for i, conf1 in enumerate(reference_confs):
        for j, conf2 in enumerate(new_confs):
            rmsd = _get_residue_rmsd(conf1, conf2)
            print(f"{i} {j} {rmsd}")
            if rmsd < rmsd_max_same_conf:
                pairs.append((i, j))
                rmsds.append(rmsd)
                break
    return pairs, rmsds


def _get_multi_conf_residues(multi_conf):
    reference_confs = []
    k = 1
    max_k = len(list(multi_conf.residues)) - 1
    while k < max_k:
        reference_confs.append(list(multi_conf.residues)[k])
        k += 3
    return reference_confs


class TestQfitResidueSampling(SyntheticMapRunner):

    def _get_qfit_options(self):
        options = QFitOptions()
        options.rotamer_neighborhood = 10
        options.sample_backbone_amplitude = 0.1
        return options

    def _load_qfit_inputs(self, pdb_file, mtz_file):
        structure = Structure.fromfile(pdb_file)
        xmap = XMap.fromfile(mtz_file, label="FWT,PHIFWT")
        return structure, xmap

    def _run_sample(self, residue, structure, xmap, options):
        with self._run_in_tmpdir():
            runner = QFitRotamericResidue(residue, structure, xmap, options)
            runner._sample_sidechain()  # pylint: disable=protected-access
            return runner

    def _run_sample_sidechain_3mer(self, pdb_file, mtz_file, options):
        structure, xmap = self._load_qfit_inputs(pdb_file, mtz_file)
        residue = list(structure.residues)[1]
        return self._run_sample(residue, structure, xmap, options)

    def _run_setup_and_sample_sidechain_3mer(self, peptide_name, high_resolution):
        pdb_multi, pdb_single = self._get_start_models(peptide_name)
        fmodel_mtz = self._create_fmodel(pdb_multi, high_resolution=high_resolution)
        options = self._get_qfit_options()
        result = self._run_sample_sidechain_3mer(pdb_single, fmodel_mtz, options)
        multi_conf = Structure.fromfile(pdb_multi)
        return result, multi_conf

    def _run_sample_sidechain_3mer_and_validate(self,
                                                peptide_name,
                                                high_resolution,
                                                rmsd_max_same_conf,
                                                rmsd_min_other_conf):
        result, multi = self._run_setup_and_sample_sidechain_3mer(
            peptide_name, high_resolution)
        new_confs = result.get_conformers()
        reference_confs = _get_multi_conf_residues(multi)
        assert len(reference_confs) == len(new_confs)
        pairs, rmsds = _get_best_rmsds(reference_confs, new_confs, rmsd_max_same_conf)
        assert len(pairs) == len(reference_confs)
        assert min(rmsds) < 0.05
        for i, conf1 in enumerate(new_confs[:-1]):
            for conf2 in new_confs[i+1:]:
                rmsd = _get_residue_rmsd(conf1, conf2)
                assert rmsd > rmsd_min_other_conf

    def test_sample_sidechain_3mer_ser_p21(self):
        self._run_sample_sidechain_3mer_and_validate(
            peptide_name="ASA",
            high_resolution=1.5,
            rmsd_max_same_conf=0.15,
            rmsd_min_other_conf=2.15)

    def test_sample_sidechain_3mer_lys_p21(self):
        self._run_sample_sidechain_3mer_and_validate(
            peptide_name="AKA",
            high_resolution=1.2,
            rmsd_max_same_conf=0.6,  # this seems quite high
            rmsd_min_other_conf=5)

    def test_sample_sidechain_3mer_phe_p21(self):
        result, multi = self._run_setup_and_sample_sidechain_3mer("AFA", 1.5)
        new_confs = result.get_conformers()
        # I think the third conformer is a flipped ring?  In the full program
        # it appears to get pruned later
        assert 2 <= len(new_confs) <= 3
        reference_confs = _get_multi_conf_residues(multi)
        pairs, rmsds = _get_best_rmsds(reference_confs, new_confs, 0.6)
        assert len(pairs) == 2

    def test_sample_sidechain_serine_space_group_symops(self):
        """
        Search for rotamers of an isolated two-conformer Ser residue, iterating
        over several different spacegroups and all of their symmetry operators
        """
        d_min = 1.5
        for pdb_multi, pdb_single in self._get_all_serine_monomer_crystals():
            fmodel_mtz = self._create_fmodel(pdb_multi, d_min)
            for pdb_symm in self._iterate_symmetry_mate_models(pdb_single):
                print(pdb_symm)
                structure, xmap = self._load_qfit_inputs(pdb_symm, fmodel_mtz)
                residue = list(structure.residues)[0]
                options = self._get_qfit_options()
                result = self._run_sample(residue, structure, xmap, options)
                rotamers = set([])
                for residue in result.get_conformers():
                    rotamers.add(self._get_rotamer(residue, chi_radius=12))
                assert rotamers == {(-177,), (-65,)}
