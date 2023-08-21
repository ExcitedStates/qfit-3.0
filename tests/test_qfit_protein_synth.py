"""
Relatively fast integration tests of qfit_protein using synthetic data for
a variety of small peptides with several alternate conformers.
"""

from collections import defaultdict
import subprocess
import unittest
import tempfile
import os.path as op
import os
import sys

from iotbx.file_reader import any_file
import pytest

from qfit.structure import Structure
from qfit.utils.mock_utils import BaseTestRunner


class SyntheticMapRunner(BaseTestRunner):
    DATA = op.join(op.dirname(__file__), "data")

    def _get_file_path(self, base_name):
        return op.join(self.DATA, base_name)

    def _get_start_models(self, peptide_name):
        return (
            self._get_file_path(f"{peptide_name}_multiconf.pdb"),
            self._get_file_path(f"{peptide_name}_single.pdb"),
        )


class TestQfitProteinSyntheticData(SyntheticMapRunner):

    def _run_qfit_cli(self, pdb_file_multi, pdb_file_single, high_resolution):
        fmodel_mtz = self._create_fmodel(pdb_file_multi,
                                         high_resolution=high_resolution)
        os.symlink(pdb_file_single, "single.pdb")
        qfit_args = [
            "qfit_protein",
            fmodel_mtz,
            pdb_file_single,
            "--resolution",
            str(high_resolution),
            "--label",
            "FWT,PHIFWT",
            "--backbone-amplitude",
            "0.1",
            "--rotamer-neighborhood",
            "10",
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
        model_name="multiconformer_model2.pdb",
    ):
        fmodel_out = self._create_fmodel(model_name,
                                         high_resolution=high_resolution)
        # correlation of the single-conf 7-mer fmodel is 0.922
        self._compare_maps(fmodel_in, fmodel_out, expected_correlation)

    def _run_and_validate_identical_rotamers(
        self,
        pdb_multi,
        pdb_single,
        d_min,
        chi_radius=SyntheticMapRunner.CHI_RADIUS,
        expected_correlation=0.99,
        model_name="multiconformer_model2.pdb",
    ):
        fmodel_mtz = self._run_qfit_cli(pdb_multi, pdb_single, high_resolution=d_min)
        self._validate_new_fmodel(
            fmodel_in=fmodel_mtz,
            high_resolution=d_min,
            expected_correlation=expected_correlation
        )
        rotamers_in = self._get_model_rotamers(pdb_multi, chi_radius)
        rotamers_out = self._get_model_rotamers(model_name, chi_radius)
        for resi in rotamers_in.keys():
            assert rotamers_in[resi] == rotamers_out[resi]
        return rotamers_out

    def _run_kmer_and_validate_identical_rotamers(
        self, peptide_name, d_min, chi_radius=SyntheticMapRunner.CHI_RADIUS
    ):
        (pdb_multi, pdb_single) = self._get_start_models(peptide_name)
        return self._run_and_validate_identical_rotamers(
            pdb_multi, pdb_single, d_min, chi_radius
        )

    def _run_serine_monomer(self, space_group_symbol):
        (pdb_multi, pdb_single) = self._get_serine_monomer_with_symmetry(
            space_group_symbol)
        return self._run_and_validate_identical_rotamers(
            pdb_multi, pdb_single, d_min=1.5, chi_radius=5)

    def test_qfit_protein_ser_basic_box(self):
        """A single two-conformer Ser residue in a perfectly cubic P1 cell"""
        (pdb_multi, pdb_single) = self._get_serine_monomer_inputs()
        return self._run_and_validate_identical_rotamers(
            pdb_multi, pdb_single, d_min=1.5, chi_radius=5)

    def test_qfit_protein_ser_p1(self):
        """A single two-conformer Ser residue in an irregular triclinic cell"""
        self._run_serine_monomer("P1")

    def test_qfit_protein_ser_p21(self):
        """A single two-conformer Ser residue in a P21 cell"""
        self._run_serine_monomer("P21")

    def test_qfit_protein_ser_p4212(self):
        """A single two-conformer Ser residue in a P4212 cell"""
        self._run_serine_monomer("P4212")

    def test_qfit_protein_ser_p6322(self):
        """A single two-conformer Ser residue in a P6322 cell"""
        self._run_serine_monomer("P6322")

    def test_qfit_protein_ser_c2221(self):
        """A single two-conformer Ser residue in a C2221 cell"""
        self._run_serine_monomer("C2221")

    def test_qfit_protein_ser_i212121(self):
        """A single two-conformer Ser residue in a I212121 cell"""
        self._run_serine_monomer("I212121")

    def test_qfit_protein_ser_i422(self):
        """A single two-conformer Ser residue in a I422 cell"""
        self._run_serine_monomer("I422")

    def test_qfit_protein_3mer_arg_p21(self):
        """Build an Arg residue with two conformers"""
        self._run_kmer_and_validate_identical_rotamers("ARA", d_min=1.0, chi_radius=8)

    def test_qfit_protein_3mer_lys_p21(self):
        """Build a Lys residue with three rotameric conformations"""
        rotamers = self._run_kmer_and_validate_identical_rotamers(
            "AKA", d_min=1.2, chi_radius=15
        )
        assert len(rotamers) == 3  # just to be certain

    def test_qfit_protein_3mer_ser_p21(self):
        """Build a Ser residue with two rotamers at moderate resolution"""
        self._run_kmer_and_validate_identical_rotamers("ASA", 1.65, chi_radius=15)

    def test_qfit_protein_3mer_trp_2conf_p21(self):
        """
        Build a Trp residue with two rotamers at medium resolution
        """
        pdb_multi = self._get_file_path("AWA_2conf.pdb")
        pdb_single = self._get_file_path("AWA_single.pdb")
        rotamers = self._run_and_validate_identical_rotamers(
            pdb_multi,
            pdb_single,
            d_min=2.00009,
            chi_radius=15,
            # FIXME the associated CCP4 map input test consistently has a lower
            # correlation than the MTZ input version
            expected_correlation=0.9845,
        )
        # this should not find a third distinct conformation (although it may
        # have overlapped conformations of the same rotamer)
        assert len(rotamers[2]) == 2

    @pytest.mark.skipif(sys.platform == "darwin", reason="FIXME: Skipping due to CPLEX Error 5002 in CI tests")
    def test_qfit_protein_3mer_trp_3conf_p21(self):
        """
        Build a Trp residue with three different rotamers, two of them
        with overlapped 5-member rings
        """
        pdb_multi = self._get_file_path("AWA_3conf.pdb")
        pdb_single = self._get_file_path("AWA_single.pdb")
        rotamers = self._run_and_validate_identical_rotamers(
            pdb_multi, pdb_single, d_min=0.8
        )
        assert len(rotamers[2]) == 3
        s = Structure.fromfile("multiconformer_model2.pdb")
        trp_confs = [r for r in s.residues if r.resn[0] == "TRP"]
        # FIXME with the minimized model we get 4 confs, at any resolution
        # assert len(trp_confs) == 3

    def _validate_phe_3mer_confs(
        self, pdb_file_multi, new_model_name="multiconformer_model2.pdb"
    ):
        rotamers_in = self._get_model_rotamers(pdb_file_multi)
        rotamers_out = self._get_model_rotamers(new_model_name, chi_radius=15)
        # Phe2 should have two rotamers, but this may occasionally appear as
        # three due to the ring flips, and we can't depend on which orientation
        # the ring ends up in
        assert (-177, 80) in rotamers_out[2]  # this doesn't flip???
        assert (-65, -85) in rotamers_out[2] or (-65, 85) in rotamers_out[2]

    def test_qfit_protein_3mer_phe_p21(self):
        """
        Build a Phe residue with two conformers in P21 at medium resolution
        """
        d_min = 1.5
        (pdb_multi, pdb_single) = self._get_start_models("AFA")
        fmodel_in = self._run_qfit_cli(pdb_multi, pdb_single, high_resolution=d_min)
        self._validate_phe_3mer_confs(pdb_multi)
        self._validate_new_fmodel(fmodel_in=fmodel_in, high_resolution=d_min)

    def test_qfit_protein_3mer_phe_p21_mmcif(self):
        """
        Build a Phe residue with two conformers using mmCIF input
        """
        d_min = 1.5
        (pdb_multi, pdb_single) = self._get_start_models("AFA")
        cif_single = "single_conf.cif"
        s = Structure.fromfile(pdb_single)
        s.tofile(cif_single)
        fmodel_in = self._run_qfit_cli(pdb_multi, cif_single, high_resolution=d_min)
        self._validate_phe_3mer_confs(pdb_multi, "multiconformer_model.cif")
        self._validate_new_fmodel(
            fmodel_in=fmodel_in,
            high_resolution=d_min, model_name="multiconformer_model.cif"
        )

    def test_qfit_protein_3mer_phe_p1(self):
        """
        Build a Phe residue with two conformers in a smaller P1 cell at
        medium resolution
        """
        d_min = 1.5
        new_models = []
        for pdb_file in self._get_start_models("AFA"):
            new_models.append(self._replace_symmetry(
                new_symmetry=("P1", (12, 6, 10, 90, 105, 90)),
                pdb_file=pdb_file))
        (pdb_multi, pdb_single) = new_models
        fmodel_in = self._run_qfit_cli(pdb_multi, pdb_single, high_resolution=d_min)
        self._validate_phe_3mer_confs(pdb_multi)
        self._validate_new_fmodel(fmodel_in=fmodel_in,
                                  high_resolution=d_min)

    def test_qfit_protein_7mer_peptide_p21(self):
        """
        Build a 7-mer peptide with multiple residues in double conformations
        """
        d_min = 1.3
        (pdb_multi, pdb_single) = self._get_start_models("GNNAFNS")
        fmodel_in = self._run_qfit_cli(pdb_multi, pdb_single, high_resolution=d_min)
        self._validate_7mer_confs(pdb_multi)
        self._validate_new_fmodel(fmodel_in, d_min, 0.95)

    def test_qfit_protein_7mer_peptide_p1(self):
        """
        Build a 7-mer peptide with multiple residues in double conformations
        in a smaller P1 cell.
        """
        d_min = 1.3
        new_models = []
        for pdb_file in self._get_start_models("GNNAFNS"):
            new_models.append(self._replace_symmetry(
                new_symmetry=("P1", (30, 10, 15, 90, 105, 90)),
                pdb_file=pdb_file))
        (pdb_multi, pdb_single) = new_models
        fmodel_in = self._run_qfit_cli(pdb_multi, pdb_single, high_resolution=d_min)
        self._validate_7mer_confs(pdb_multi)
        self._validate_new_fmodel(fmodel_in, d_min, 0.95)

    def _validate_7mer_confs(self, pdb_file_multi):
        rotamers_in = self._get_model_rotamers(pdb_file_multi)
        rotamers_out = self._get_model_rotamers(
            "multiconformer_model2.pdb", chi_radius=15
        )
        # Phe5 should have two rotamers, but this may occasionally appear as
        # three due to the ring flips, and we can't depend on which orientation
        # the ring ends up in
        assert (-177, 80) in rotamers_out[5]  # this doesn't flip???
        assert (-65, -85) in rotamers_out[5] or (-65, 85) in rotamers_out[5]
        # Asn are also awkward because of flips
        assert len(rotamers_out[3]) >= 2
        assert len(rotamers_out[6]) >= 2
        # these are all of the alt confs present in the fmodel structure
        assert rotamers_in[3] - rotamers_out[3] == set()
        assert rotamers_in[2] - rotamers_out[2] == set()

    def test_qfit_protein_3mer_lys_p6322_all_sites(self):
        """
        Iterate over all symmetry operators in the P6(3)22 space group and
        confirm that qFit builds three distinct rotamers starting from
        the symmetry mate coordinates
        """
        d_min = 1.2
        pdb_multi = self._get_file_path("AKA_p6322_3conf.pdb")
        pdb_single_start = self._get_file_path("AKA_p6322_single.pdb")
        cwd = os.getcwd()
        for i_op, pdb_single in enumerate(
            self._iterate_symmetry_mate_models(pdb_single_start)
        ):
            print(f"running with model {op.basename(pdb_single)}")
            with self._run_in_tmpdir(f"op{i_op}"):
                rotamers = self._run_and_validate_identical_rotamers(
                    pdb_multi, pdb_single, d_min=d_min, chi_radius=15
                )
                assert len(rotamers[2]) == 3

    def test_qfit_protein_3mer_arg_sensitivity(self):
        """
        Build a low-occupancy Arg conformer.
        """
        d_min = 1.20059
        # FIXME this test is very sensitive to slight differences in input and
        # OS - in some circumstances it can detect occupancy as low as 0.28,
        # but not when using CCP4 input
        occ_B = 0.32
        (pdb_multi_start, pdb_single) = self._get_start_models("ARA")
        pdb_in = any_file(pdb_multi_start)
        symm = pdb_in.file_object.crystal_symmetry()
        pdbh = pdb_in.file_object.hierarchy
        cache = pdbh.atom_selection_cache()
        atoms = pdbh.atoms()
        occ = atoms.extract_occ()
        sele1 = cache.selection("altloc A")
        sele2 = cache.selection("altloc B")
        occ.set_selected(sele1, 1 - occ_B)
        occ.set_selected(sele2, occ_B)
        atoms.set_occ(occ)
        pdb_multi_new = "ARA_low_occ.pdb"
        pdbh.write_pdb_file(pdb_multi_new, crystal_symmetry=symm)
        self._run_and_validate_identical_rotamers(pdb_multi_new, pdb_single, d_min)

    def test_qfit_protein_3mer_multiconformer(self):
        """
        Build a 3-mer peptide with three continuous conformations and one or
        two alternate rotamers for each residue
        """
        d_min = 1.2
        (pdb_multi, pdb_single) = self._get_start_models("SKH")
        rotamers = self._run_and_validate_identical_rotamers(
            pdb_multi, pdb_single, d_min=d_min, chi_radius=15
        )
        # TODO this test should also evaluate the occupancies, which are not
        # constrained between residues
        assert len(rotamers[1]) == 2
        assert len(rotamers[2]) == 3
        assert len(rotamers[3]) == 2
