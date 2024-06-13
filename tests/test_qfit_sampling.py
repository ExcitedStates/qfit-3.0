"""
Test protein residue sampling methods in qfit.qfit.QFitRotamericResidue
"""

import numpy as np
import pytest

from qfit.qfit import QFitRotamericResidue, QFitOptions
from qfit.solvers import available_qp_solvers, available_miqp_solvers
from qfit.structure import Structure
from qfit.xtal.volume import XMap

from .test_qfit_protein_synth import QfitProteinSyntheticDataRunner

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


class TestQfitResidueSampling(QfitProteinSyntheticDataRunner):

    def _get_qfit_options(self):
        options = QFitOptions()
        options.rotamer_neighborhood = 10
        options.sample_backbone_amplitude = 0.1
        options.qp_solver = next(iter(available_qp_solvers.keys()))
        options.miqp_solver = next(iter(available_miqp_solvers.keys()))
        # XXX default values don't work for Phe and Lys tests
        options.dofs_per_iteration = 2
        options.dihedral_stepsize = 10
        # TODO make this the default
        options.transformer = "cctbx"
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
        self._validate_conformers(multi, result, rmsd_max_same_conf,
                                  rmsd_min_other_conf)

    def _validate_conformers(self,
                             input_model,
                             result,
                             rmsd_max_same_conf,
                             rmsd_min_other_conf):
        reference_confs = _get_multi_conf_residues(input_model)
        new_confs = result.get_conformers()
        assert len(new_confs) == len(reference_confs)
        pairs, rmsds = _get_best_rmsds(reference_confs, new_confs, rmsd_max_same_conf)
        assert len(pairs) == len(reference_confs)
        assert min(rmsds) < 0.05
        for i, conf1 in enumerate(new_confs[:-1]):
            for conf2 in new_confs[i+1:]:
                rmsd = _get_residue_rmsd(conf1, conf2)
                assert rmsd > rmsd_min_other_conf

    # TODO figure out why this always finds 3 confs using CCTBX transformer
    #@pytest.mark.fast
    @pytest.mark.skip(reason="FIXME consistently finds 3 conformations down to 0.5 Angstrom resolution")
    def test_sample_sidechain_3mer_ser_p21(self):
        self._run_sample_sidechain_3mer_and_validate(
            peptide_name="ASA",
            high_resolution=1.3,
            rmsd_max_same_conf=0.15,
            rmsd_min_other_conf=2.15)

    @pytest.mark.skip(reason="FIXME only 2 conformations found")
    def test_sample_sidechain_3mer_lys_p21(self):
        self._run_sample_sidechain_3mer_and_validate(
            peptide_name="AKA",
            high_resolution=1.1,
            rmsd_max_same_conf=0.6,  # this seems quite high
            rmsd_min_other_conf=5)

    @pytest.mark.fast
    def test_sample_sidechain_3mer_phe_p21(self):
        result, multi = self._run_setup_and_sample_sidechain_3mer("AFA", 1.2)
        new_confs = result.get_conformers()
        # I think the third conformer is a flipped ring?  In the full program
        # it appears to get pruned later
        assert 2 <= len(new_confs) <= 3
        reference_confs = _get_multi_conf_residues(multi)
        pairs, rmsds = _get_best_rmsds(reference_confs, new_confs, 0.6)
        assert len(pairs) == 2

    @pytest.mark.slow
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

    def _run_sampling_rebuilt_3mer(self,
                                   resname,
                                   d_min=1.5,
                                   rmsd_max_same_conf=0.15,
                                   rmsd_min_other_conf=2.15):
        pdb_multi, pdb_single = self._create_mock_multi_conf_3mer(resname)
        fmodel_mtz = self._create_fmodel(pdb_multi, high_resolution=d_min)
        options = self._get_qfit_options()
        result = self._run_sample_sidechain_3mer(pdb_single, fmodel_mtz,
                                                 options)
        multi_conf = Structure.fromfile(pdb_multi)
        self._validate_conformers(multi_conf, result,
                                  rmsd_max_same_conf=rmsd_max_same_conf,
                                  rmsd_min_other_conf=rmsd_min_other_conf)

    def test_sampling_rebuilt_tripeptide_arg(self):
        self._run_sampling_rebuilt_3mer("ARG", d_min=1.6)

    def test_sampling_rebuilt_tripeptide_asn(self):
        self._run_sampling_rebuilt_3mer("ASN", d_min=1.4)

    def test_sampling_rebuilt_tripeptide_asp(self):
        self._run_sampling_rebuilt_3mer("ASP")

    def test_sampling_rebuilt_tripeptide_cys(self):
        self._run_sampling_rebuilt_3mer("CYS", d_min=2.0)

    @pytest.mark.skip(reason="FIXME needs more debugging")
    def test_sampling_rebuilt_tripeptide_gln(self):
        self._run_sampling_rebuilt_3mer("GLN")

    @pytest.mark.skip(reason="FIXME needs more debugging")
    def test_sampling_rebuilt_tripeptide_glu(self):
        self._run_sampling_rebuilt_3mer("GLU")

    def test_sampling_rebuilt_tripeptide_his(self):
        self._run_sampling_rebuilt_3mer("HIS")

    def test_sampling_rebuilt_tripeptide_ile(self):
        self._run_sampling_rebuilt_3mer("ILE", d_min=1.4)

    def test_sampling_rebuilt_tripeptide_leu(self):
        self._run_sampling_rebuilt_3mer("LEU")

    def test_sampling_rebuilt_tripeptide_lys(self):
        self._run_sampling_rebuilt_3mer("LYS", d_min=1.6)

    def test_sampling_rebuilt_tripeptide_met(self):
        self._run_sampling_rebuilt_3mer("MET", d_min=1.6)

    def test_sampling_rebuilt_tripeptide_phe(self):
        self._run_sampling_rebuilt_3mer("PHE", d_min=1.35)

    def test_sampling_rebuilt_tripeptide_ser(self):
        self._run_sampling_rebuilt_3mer("SER", d_min=2.0)

    def test_sampling_rebuilt_tripeptide_thr(self):
        self._run_sampling_rebuilt_3mer("THR", d_min=1.8)

    def test_sampling_rebuilt_tripeptide_trp(self):
        self._run_sampling_rebuilt_3mer("TRP", d_min=2.0)

    @pytest.mark.skip(reason="FIXME needs more debugging")
    def test_sampling_rebuilt_tripeptide_tyr(self):
        self._run_sampling_rebuilt_3mer("TYR", d_min=1.5)

    def test_sampling_rebuilt_tripeptide_val(self):
        self._run_sampling_rebuilt_3mer("VAL", d_min=1.8)
