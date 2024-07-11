# """
# Relatively fast integration tests of qfit_protein using synthetic data for
# a variety of small peptides with several alternate conformers.
# """

# import subprocess
# import tempfile
# import os.path as op
# import os

# from iotbx.file_reader import any_file
# import cctbx.crystal
# import pytest

# from qfit.qfit import QFitOptions, QFitSegment
# from qfit.solvers import available_qp_solvers, available_miqp_solvers
# from qfit.structure import Structure
# from qfit.xtal.volume import XMap
# from qfit.utils.mock_utils import BaseTestRunner

# DISABLE_SLOW = os.environ.get("QFIT_ENABLE_SLOW_TESTS", None) is None

# class SyntheticMapRunner(BaseTestRunner):
#     DATA = op.join(op.dirname(__file__), "data")

#     def _get_file_path(self, base_name):
#         return op.join(self.DATA, base_name)

#     def _get_start_models(self, peptide_name):
#         return (
#             self._get_file_path(f"{peptide_name}_multiconf.pdb"),
#             self._get_file_path(f"{peptide_name}_single.pdb"),
#         )


# class QfitProteinSyntheticDataRunner(SyntheticMapRunner):
#     COMMON_SAMPLING_ARGS = [
#         "--backbone-amplitude",
#         "0.1",
#         "--rotamer-neighborhood",
#         "10",
#         # XXX this is required for many of these tests to work
#         "--dofs-per-iteration",
#         "2",
#         "--dihedral-stepsize",
#         "10",
#         "--transformer", "cctbx"
#     ]

#     def _run_qfit_cli(self, pdb_file_multi, pdb_file_single, high_resolution,
#                       extra_args=(), em=False):
#         fmodel_mtz = self._create_fmodel(pdb_file_multi,
#                                          high_resolution=high_resolution,
#                                          em=em)
#         qfit_args = [
#             "qfit_protein",
#             fmodel_mtz,
#             pdb_file_single,
#             #"--debug",
#             "--resolution",
#             str(high_resolution),
#             "--label",
#             "FWT,PHIFWT",
#         ] + self.COMMON_SAMPLING_ARGS + list(extra_args)
#         if em:
#             qfit_args.append("--cryo_em")
#         print(" ".join(qfit_args))
#         subprocess.check_call(qfit_args)
#         return fmodel_mtz

#     def _validate_new_fmodel(
#         self,
#         fmodel_in,
#         high_resolution,
#         cc_min=0.99,
#         model_name="multiconformer_model2.pdb",
#         em=False
#     ):
#         fmodel_out = self._create_fmodel(model_name,
#                                          high_resolution=high_resolution,
#                                          em=em,
#                                          reference_file=fmodel_in)
#         # correlation of the single-conf 7-mer fmodel is 0.922
#         self._compare_maps(fmodel_in, fmodel_out, cc_min)

#     def _run_and_validate_identical_rotamers(
#         self,
#         pdb_multi,
#         pdb_single,
#         d_min,
#         chi_radius=SyntheticMapRunner.CHI_RADIUS,
#         cc_min=0.99,
#         model_name="multiconformer_model2.pdb",
#         extra_args=(),
#         em=False,
#     ):
#         fmodel_mtz = self._run_qfit_cli(pdb_multi, pdb_single,
#                                         high_resolution=d_min,
#                                         extra_args=extra_args,
#                                         em=em)
#         self._validate_new_fmodel(
#             fmodel_in=fmodel_mtz,
#             high_resolution=d_min,
#             cc_min=cc_min,
#             em=em
#         )
#         rotamers_in = self._get_model_rotamers(pdb_multi, chi_radius)
#         rotamers_out = self._get_model_rotamers(model_name, chi_radius)
#         for resi in rotamers_in.keys():
#             assert rotamers_in[resi] == rotamers_out[resi]
#         return rotamers_out


# class TestQfitProteinSimple(QfitProteinSyntheticDataRunner):

#     def test_qfit_protein_ser_basic_box(self):
#         """A single two-conformer Ser residue in a perfectly cubic P1 cell"""
#         (pdb_multi, pdb_single) = self._get_serine_monomer_inputs()
#         return self._run_and_validate_identical_rotamers(
#             pdb_multi, pdb_single, d_min=1.5, chi_radius=5)

#     def test_qfit_segment_ser_p1(self):
#         """
#         Run just the segment sampling routine on a single two-conformer Ser
#         residue in an irregular triclinic cell
#         """
#         (pdb_multi, pdb_single) = self._get_serine_monomer_inputs()
#         fmodel_mtz = self._create_fmodel(pdb_multi, high_resolution=1.5)
#         xmap = XMap.fromfile(fmodel_mtz, label="FWT,PHIFWT")
#         structure = Structure.fromfile(pdb_single)
#         assert structure.n_residues() == 1
#         assert len(list(structure.extract("record", "ATOM").residue_groups)) == 1
#         options = QFitOptions()
#         options.qp_solver = next(iter(available_qp_solvers.keys()))
#         options.miqp_solver = next(iter(available_miqp_solvers.keys()))
#         qfit = QFitSegment(structure, xmap, options)
#         multiconf = qfit()

#     def test_qfit_protein_ser_basic_box_cryo_em(self):
#         """A single two-conformer Ser residue in a perfectly cubic EM map"""
#         (pdb_multi, pdb_single) = self._get_serine_monomer_inputs()
#         return self._run_and_validate_identical_rotamers(
#             pdb_multi, pdb_single, d_min=1.5, chi_radius=5, em=True)


# @pytest.mark.slow
# class TestQfitProteinSyntheticData(QfitProteinSyntheticDataRunner):

#     # cutoff for multiple tests that behave differently for Xray vs EM
#     MIN_CC_PHE = 0.989
#     D_MIN_7MER = 1.3

#     def _run_kmer_and_validate_identical_rotamers(
#         self, peptide_name, d_min, chi_radius=SyntheticMapRunner.CHI_RADIUS,
#         cc_min=0.99, extra_args=()
#     ):
#         (pdb_multi, pdb_single) = self._get_start_models(peptide_name)
#         return self._run_and_validate_identical_rotamers(
#             pdb_multi, pdb_single, d_min, chi_radius, cc_min=cc_min,
#             extra_args=extra_args
#         )

#     def _run_serine_monomer(self, space_group_symbol, cc_min=0.99):
#         (pdb_multi, pdb_single) = self._get_serine_monomer_with_symmetry(
#             space_group_symbol)
#         return self._run_and_validate_identical_rotamers(
#             pdb_multi, pdb_single, d_min=1.5, chi_radius=5,
#             cc_min=cc_min)

#     def test_qfit_protein_ser_p1(self):
#         """A single two-conformer Ser residue in an irregular triclinic cell"""
#         self._run_serine_monomer("P1")

#     def test_qfit_protein_ser_p21(self):
#         """A single two-conformer Ser residue in a P21 cell"""
#         self._run_serine_monomer("P21")

#     def test_qfit_protein_ser_p4212(self):
#         """A single two-conformer Ser residue in a P4212 cell"""
#         self._run_serine_monomer("P4212")

#     def test_qfit_protein_ser_p6322(self):
#         """A single two-conformer Ser residue in a P6322 cell"""
#         self._run_serine_monomer("P6322")

#     def test_qfit_protein_ser_c2221(self):
#         """A single two-conformer Ser residue in a C2221 cell"""
#         self._run_serine_monomer("C2221")

#     def test_qfit_protein_ser_i212121(self):
#         """A single two-conformer Ser residue in a I212121 cell"""
#         self._run_serine_monomer("I212121")

#     def test_qfit_protein_ser_i422(self):
#         """A single two-conformer Ser residue in a I422 cell"""
#         self._run_serine_monomer("I422")

#     def test_qfit_protein_3mer_lys_p21(self):
#         """Build a Lys residue with three rotameric conformations"""
#         rotamers = self._run_kmer_and_validate_identical_rotamers(
#             "AKA", d_min=1.2, chi_radius=15, cc_min=0.9885
#         )
#         assert len(rotamers) == 3  # just to be certain

#     def test_qfit_protein_3mer_ser_p21(self):
#         """Build a Ser residue with two rotamers at moderate resolution"""
#         self._run_kmer_and_validate_identical_rotamers("ASA", 1.5, chi_radius=15)

#     def test_qfit_protein_3mer_ser_p21_parallel(self):
#         """Build a Ala-Ser-Ala model in parallel"""
#         self._run_kmer_and_validate_identical_rotamers("ASA", 1.5,
#             chi_radius=15, extra_args=("--nproc", "3"))

#     def test_qfit_protein_3mer_trp_2conf_p21(self):
#         """
#         Build a Trp residue with two rotamers at medium resolution
#         """
#         pdb_multi = self._get_file_path("AWA_2conf.pdb")
#         pdb_single = self._get_file_path("AWA_single.pdb")
#         rotamers = self._run_and_validate_identical_rotamers(
#             pdb_multi,
#             pdb_single,
#             d_min=1.95, #1.7,
#             chi_radius=15,
#             cc_min=0.973
#         )
#         # this should not find a third distinct conformation (although it may
#         # have overlapped conformations of the same rotamer)
#         assert len(rotamers[2]) == 2

#     def test_qfit_protein_3mer_trp_3conf_p21(self):
#         """
#         Build a Trp residue with three different rotamers, two of them
#         with overlapped 5-member rings
#         """
#         pdb_multi = self._get_file_path("AWA_3conf.pdb")
#         pdb_single = self._get_file_path("AWA_single.pdb")
#         rotamers = self._run_and_validate_identical_rotamers(
#             pdb_multi, pdb_single, d_min=0.95, cc_min=0.985, chi_radius=15
#         )
#         assert len(rotamers[2]) == 3
#         s = Structure.fromfile("multiconformer_model2.pdb")
#         trp_confs = [r for r in s.residues if r.resn[0] == "TRP"]
#         # FIXME with the minimized model we get 4 confs, at any resolution
#         assert len(trp_confs) >= 3

#     def _validate_phe_3mer_confs(
#         self, pdb_file_multi, new_model_name="multiconformer_model2.pdb"
#     ):
#         #rotamers_in = self._get_model_rotamers(pdb_file_multi)
#         rotamers_out = self._get_model_rotamers(new_model_name, chi_radius=15)
#         # Phe2 should have two rotamers, but this may occasionally appear as
#         # three due to the ring flips, and we can't depend on which orientation
#         # the ring ends up in
#         assert (-177, 80) in rotamers_out[2]  # this doesn't flip???
#         assert (-65, -85) in rotamers_out[2] or (-65, 85) in rotamers_out[2]

#     def test_qfit_protein_3mer_phe_p21(self):
#         """
#         Build a Phe residue with two conformers in P21 at medium resolution
#         """
#         d_min = 1.5
#         (pdb_multi, pdb_single) = self._get_start_models("AFA")
#         fmodel_in = self._run_qfit_cli(pdb_multi, pdb_single, high_resolution=d_min)
#         self._validate_phe_3mer_confs(pdb_multi)
#         self._validate_new_fmodel(fmodel_in=fmodel_in,
#                                   high_resolution=d_min,
#                                   cc_min=self.MIN_CC_PHE)

#     def test_qfit_protein_3mer_phe_p21_mmcif(self):
#         """
#         Build a Phe residue with two conformers using mmCIF input
#         """
#         d_min = 1.5
#         (pdb_multi, pdb_single) = self._get_start_models("AFA")
#         cif_single = "single_conf.cif"
#         s = Structure.fromfile(pdb_single)
#         s.tofile(cif_single)
#         fmodel_in = self._run_qfit_cli(pdb_multi, cif_single, high_resolution=d_min)
#         self._validate_phe_3mer_confs(pdb_multi, "multiconformer_model2.cif")
#         self._validate_new_fmodel(
#             fmodel_in=fmodel_in,
#             high_resolution=d_min,
#             model_name="multiconformer_model2.cif",
#             cc_min=self.MIN_CC_PHE
#         )

#     def test_qfit_protein_3mer_phe_p1(self):
#         """
#         Build a Phe residue with two conformers in a smaller P1 cell at
#         medium resolution
#         """
#         d_min = 1.5
#         new_models = []
#         for pdb_file in self._get_start_models("AFA"):
#             new_models.append(self._replace_symmetry(
#                 new_symmetry=("P1", (12, 6, 10, 90, 105, 90)),
#                 pdb_file=pdb_file))
#         (pdb_multi, pdb_single) = new_models
#         fmodel_in = self._run_qfit_cli(pdb_multi, pdb_single, high_resolution=d_min)
#         self._validate_phe_3mer_confs(pdb_multi)
#         self._validate_new_fmodel(fmodel_in=fmodel_in,
#                                   high_resolution=d_min,
#                                   cc_min=self.MIN_CC_PHE)

#     @pytest.mark.skipif(DISABLE_SLOW, reason="Redundant P21 test for 7-mer building")
#     def test_qfit_protein_7mer_peptide_p21(self):
#         """
#         Build a 7-mer peptide with multiple residues in double conformations
#         """
#         (pdb_multi, pdb_single) = self._get_start_models("GNNAFNS")
#         fmodel_in = self._run_qfit_cli(pdb_multi, pdb_single,
#                                        high_resolution=self.D_MIN_7MER)
#         self._validate_7mer_confs(pdb_multi)
#         self._validate_new_fmodel(fmodel_in, self.D_MIN_7MER, 0.95)

#     def test_qfit_protein_7mer_peptide_p1(self):
#         """
#         Build a 7-mer peptide with multiple residues in double conformations
#         in a smaller P1 cell.
#         """
#         new_models = []
#         for pdb_file in self._get_start_models("GNNAFNS"):
#             new_models.append(self._replace_symmetry(
#                 new_symmetry=("P1", (30, 10, 15, 90, 105, 90)),
#                 pdb_file=pdb_file))
#         (pdb_multi, pdb_single) = new_models
#         fmodel_in = self._run_qfit_cli(pdb_multi, pdb_single,
#                                        high_resolution=self.D_MIN_7MER)
#         self._validate_7mer_confs(pdb_multi)
#         self._validate_new_fmodel(fmodel_in, self.D_MIN_7MER, 0.95)

#     def _validate_7mer_confs(self, pdb_file_multi):
#         rotamers_in = self._get_model_rotamers(pdb_file_multi)
#         rotamers_out = self._get_model_rotamers(
#             "multiconformer_model2.pdb", chi_radius=15
#         )
#         # Phe5 should have two rotamers, but this may occasionally appear as
#         # three due to the ring flips, and we can't depend on which orientation
#         # the ring ends up in
#         assert (-177, 80) in rotamers_out[5]  # this doesn't flip???
#         assert (-65, -85) in rotamers_out[5] or (-65, 85) in rotamers_out[5]
#         # Asn are also awkward because of flips
#         assert len(rotamers_out[3]) >= 2
#         assert len(rotamers_out[6]) >= 2
#         # these are all of the alt confs present in the fmodel structure
#         assert rotamers_in[3] - rotamers_out[3] == set()
#         assert rotamers_in[2] - rotamers_out[2] == set()

#     @pytest.mark.skipif(DISABLE_SLOW, reason="Slow P6322 symmetry test disabled")
#     def test_qfit_protein_3mer_lys_p6322_all_sites(self):
#         """
#         Iterate over all symmetry operators in the P6(3)22 space group and
#         confirm that qFit builds three distinct rotamers starting from
#         the symmetry mate coordinates
#         """
#         d_min = 1.2
#         pdb_multi = self._get_file_path("AKA_p6322_3conf.pdb")
#         pdb_single_start = self._get_file_path("AKA_p6322_single.pdb")
#         for i_op, pdb_single in enumerate(
#             self._iterate_symmetry_mate_models(pdb_single_start)
#         ):
#             print(f"running with model {op.basename(pdb_single)}")
#             with self._run_in_tmpdir(f"op{i_op}"):
#                 rotamers = self._run_and_validate_identical_rotamers(
#                     pdb_multi, pdb_single, d_min=d_min, chi_radius=15
#                 )
#                 assert len(rotamers[2]) == 3

#     def test_qfit_protein_3mer_arg_sensitivity(self):
#         """
#         Build a low-occupancy Arg conformer.
#         """
#         # XXX this test is very sensitive to slight differences in input and
#         # OS - in some circumstances it can detect occupancy as low as 0.28,
#         # but not when using CCP4 input
#         d_min = 1.18
#         occ_B = 0.32
#         (pdb_multi_start, pdb_single) = self._get_start_models("ARA")
#         pdb_in = any_file(pdb_multi_start)
#         symm = pdb_in.file_object.crystal_symmetry()
#         pdbh = pdb_in.file_object.hierarchy
#         cache = pdbh.atom_selection_cache()
#         atoms = pdbh.atoms()
#         occ = atoms.extract_occ()
#         sele1 = cache.selection("altloc A")
#         sele2 = cache.selection("altloc B")
#         occ.set_selected(sele1, 1 - occ_B)
#         occ.set_selected(sele2, occ_B)
#         atoms.set_occ(occ)
#         pdb_multi_new = "ARA_low_occ.pdb"
#         pdbh.write_pdb_file(pdb_multi_new, crystal_symmetry=symm)
#         self._run_and_validate_identical_rotamers(pdb_multi_new, pdb_single, d_min)

#     def test_qfit_protein_3mer_arg_rebuild(self):
#         d_min = 1.2
#         (pdb_multi_start, pdb_single) = self._get_start_models("ARA")
#         s = Structure.fromfile(pdb_single)
#         s = s.extract("name", ("N", "CA", "CB", "C", "O")).copy()
#         pdb_single_partial = "ara_single_partial.pdb"
#         s.tofile(pdb_single_partial)
#         self._run_and_validate_identical_rotamers(pdb_multi_start,
#                                                   pdb_single_partial,
#                                                   d_min)

#     def test_qfit_protein_3mer_multiconformer(self):
#         """
#         Build a 3-mer peptide with three continuous conformations and one or
#         two alternate rotamers for each residue
#         """
#         d_min = 1.2
#         (pdb_multi, pdb_single) = self._get_start_models("SKH")
#         rotamers = self._run_and_validate_identical_rotamers(
#             pdb_multi, pdb_single, d_min=d_min, chi_radius=15
#         )
#         # TODO this test should also evaluate the occupancies, which are not
#         # constrained between residues
#         assert len(rotamers[1]) == 2
#         assert len(rotamers[2]) == 3
#         assert len(rotamers[3]) == 2


# @pytest.mark.cryoem
# @pytest.mark.slow
# class TestQfitProteinSyntheticCryoEM(TestQfitProteinSyntheticData):
#     """
#     Cryo-EM tests using CCP4 map input
#     """

#     # XXX the three '3mer_phe' tests have consistently lower model-map CC in
#     # this test
#     MIN_CC_PHE = 0.975
#     D_MIN_7MER = 1.15
#     # By convention, EM structures in the PDB have placeholder CRYST1 records
#     # with P1 symmetry and unit cell parameters (1,1,1,90,90,90); this is
#     # recognized by iotbx.pdb and results in undefined symmetry in the loaded
#     # structure object.  We define it literally here so we can substitute it
#     # in the input PDB file for qfit_protein; it will be transformed to None
#     # when read back in.  (Note that mmcif does *not* use this convention, or
#     # at least iotbx does not respect it, so for those inputs we just leave
#     # out the symmetry entirely.)
#     MOCK_SYMMETRY = cctbx.crystal.symmetry(space_group_symbol="P1",
#                                            unit_cell=(1,1,1,90,90,90))

#     def _run_qfit_cli(self, pdb_file_multi, pdb_file_single, high_resolution,
#                       extra_args=(), em=True):
#         fmodel_mtz = self._create_fmodel(pdb_file_multi,
#                                          high_resolution=high_resolution,
#                                          em=True)
#         xmap = XMap.from_mtz(fmodel_mtz, label="FWT,PHIFWT")
#         xmap.tofile("fmodel_1.ccp4")
#         ext = pdb_file_single[-4:]
#         pdb_file_single_em = tempfile.NamedTemporaryFile(suffix=ext).name
#         self._replace_symmetry(
#             new_symmetry=self.MOCK_SYMMETRY if ext == ".pdb" else None,
#             pdb_file=pdb_file_single,
#             output_pdb_file=pdb_file_single_em)
#         qfit_args = [
#             "qfit_protein",
#             "fmodel_1.ccp4",
#             pdb_file_single_em,
#             #"--debug",
#             "--resolution",
#             str(high_resolution),
#         ] + self.COMMON_SAMPLING_ARGS + list(extra_args)
#         qfit_args.append("--cryo_em")
#         print(" ".join(qfit_args))
#         subprocess.check_call(qfit_args)
#         return fmodel_mtz

#     def _run_and_validate_identical_rotamers(self, *args, **kwds):
#         super()._run_and_validate_identical_rotamers(*args, **kwds, em=True)

#     def test_qfit_protein_3mer_multiconformer(self):
#         pytest.skip("failing for cryo-EM")

#     def test_qfit_protein_3mer_trp_2conf_p21(self):
#         pytest.skip("failing for cryo-EM")

#     def test_qfit_protein_3mer_trp_3conf_p21(self):
#         pytest.skip("failing for cryo-EM")

#     def test_qfit_protein_3mer_lys_p21(self):
#         pytest.skip("failing for cryo-EM")


# @pytest.mark.slow
# class TestQfitProteinSidechainRebuild(QfitProteinSyntheticDataRunner):
#     """
#     Integration tests for qfit_protein with sidechain rebuilding, covering
#     all non-PRO/GLY/ALA residues
#     """

#     def _run_rebuilt_multi_conformer_tripeptide(
#             self,
#             resname,
#             d_min=1.5,
#             set_b_iso=10,
#             chi_radius=SyntheticMapRunner.CHI_RADIUS):
#         """
#         Create a fake two-conformer structure for an A*A peptide where the
#         central residue is the specified type, and a single-conformer starting
#         model with truncated sidechains
#         """
#         pdb_multi, pdb_single = self._create_mock_multi_conf_3mer(resname,
#             set_b_iso=set_b_iso)
#         return self._run_and_validate_identical_rotamers(
#             pdb_multi, pdb_single, d_min, chi_radius)

#     # NOTE the resolution is increased for several residues that fail the
#     # test otherwise.  it is unclear why these specific residues pose
#     # problems, but this class provides a framework for further probing of
#     # how qFit is affected by resolution, B-factor, and sidechain structure

#     def test_qfit_protein_rebuilt_tripeptide_arg(self):
#         self._run_rebuilt_multi_conformer_tripeptide("ARG", d_min=1.4)

#     def test_qfit_protein_rebuilt_tripeptide_asn(self):
#         self._run_rebuilt_multi_conformer_tripeptide("ASN")

#     def test_qfit_protein_rebuilt_tripeptide_asp(self):
#         self._run_rebuilt_multi_conformer_tripeptide("ASP")

#     def test_qfit_protein_rebuilt_tripeptide_cys(self):
#         self._run_rebuilt_multi_conformer_tripeptide("CYS")

#     def test_qfit_protein_rebuilt_tripeptide_gln(self):
#         self._run_rebuilt_multi_conformer_tripeptide("GLN")

#     # TODO figure out why this fails at lower resolution
#     def test_qfit_protein_rebuilt_tripeptide_glu(self):
#         self._run_rebuilt_multi_conformer_tripeptide("GLU",
#             d_min=1.3, set_b_iso=8)

#     def test_qfit_protein_rebuilt_tripeptide_his(self):
#         self._run_rebuilt_multi_conformer_tripeptide("HIS")

#     def test_qfit_protein_rebuilt_tripeptide_ile(self):
#         self._run_rebuilt_multi_conformer_tripeptide("ILE", d_min=1.4)

#     def test_qfit_protein_rebuilt_tripeptide_leu(self):
#         self._run_rebuilt_multi_conformer_tripeptide("LEU")

#     def test_qfit_protein_rebuilt_tripeptide_lys(self):
#         self._run_rebuilt_multi_conformer_tripeptide("LYS", d_min=1.4)

#     def test_qfit_protein_rebuilt_tripeptide_met(self):
#         self._run_rebuilt_multi_conformer_tripeptide("MET")

#     # XXX in addition to not dealing with ring flips, this test fails to find
#     # a second conformation on ubuntu+python3.9
#     @pytest.mark.skip(reason="Unstable, needs to account for ring flips in comparison")
#     def test_qfit_protein_rebuilt_tripeptide_phe(self):
#         self._run_rebuilt_multi_conformer_tripeptide("PHE", d_min=1.3)

#     def test_qfit_protein_rebuilt_tripeptide_ser(self):
#         self._run_rebuilt_multi_conformer_tripeptide("SER")

#     def test_qfit_protein_rebuilt_tripeptide_thr(self):
#         self._run_rebuilt_multi_conformer_tripeptide("THR")

#     def test_qfit_protein_rebuilt_tripeptide_trp(self):
#         self._run_rebuilt_multi_conformer_tripeptide("TRP")

#     def test_qfit_protein_rebuilt_tripeptide_tyr(self):
#         self._run_rebuilt_multi_conformer_tripeptide("TYR")

#     def test_qfit_protein_rebuilt_tripeptide_val(self):
#         self._run_rebuilt_multi_conformer_tripeptide("VAL")
