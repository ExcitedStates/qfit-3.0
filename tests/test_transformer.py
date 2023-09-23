import os.path as op

import numpy as np
import pytest

from qfit.structure import Structure
from qfit.xtal.transformer import FFTTransformer
from qfit.xtal.volume import XMap

from .test_qfit_protein_synth import SyntheticMapRunner


class TestTransformer(SyntheticMapRunner):

    def _load_qfit_inputs(self, pdb_file, mtz_file):
        structure = Structure.fromfile(pdb_file)
        xmap = XMap.fromfile(mtz_file, label="FWT,PHIFWT")
        return structure, xmap

    def _run_transformer(self, pdb_multi, mtz_file, corr_min=0.999):
        structure, xmap = self._load_qfit_inputs(pdb_multi, mtz_file)
        map_data = xmap.array.copy().flatten()
        transformer = FFTTransformer(structure, xmap)
        assert transformer.hkl is not None
        transformer.density()
        map_data2 = xmap.array.copy().flatten()
        assert np.corrcoef(map_data, map_data2)[0][1] > corr_min
        return transformer

    def _run_all(self, peptide_name, d_min, corr_min=0.999):
        pdb_multi, pdb_single = self._get_start_models(peptide_name)
        fmodel_mtz = self._create_fmodel(pdb_multi, high_resolution=d_min)
        self._run_transformer(pdb_multi, fmodel_mtz, corr_min)

    @pytest.mark.fast
    def test_transformer_water_p1(self):
        pdb_file = self._get_water_pdb()
        fmodel_mtz = self._create_fmodel(pdb_file, high_resolution=2.0)
        self._run_transformer(pdb_file, fmodel_mtz)

    @pytest.mark.fast
    def test_transformer_3mer_ser_p21(self):
        self._run_all("ASA", 1.5)

    @pytest.mark.fast
    def test_transformer_3mer_lys_p21(self):
        self._run_all("AKA", 1.2)

    @pytest.mark.fast
    def test_transformer_3mer_trp_3conf_p21(self):
        pdb_multi = self._get_file_path("AWA_2conf.pdb")
        pdb_single = self._get_file_path("AWA_single.pdb")
        fmodel_mtz = self._create_fmodel(pdb_multi, high_resolution=0.8)
        self._run_transformer(pdb_multi, fmodel_mtz)

    def test_transformer_3mer_lys_p6322(self):
        pdb_multi = self._get_file_path("AKA_p6322_3conf.pdb")
        pdb_single_start = self._get_file_path("AKA_p6322_single.pdb")
        assert op.isfile(pdb_multi)
        fmodel_mtz = self._create_fmodel(pdb_multi, high_resolution=0.8)
        self._run_transformer(pdb_multi, fmodel_mtz)

    def test_transformer_ser_monomer_space_groups(self):
        """
        Test transformer behavior with a Ser monomer in several different
        space groups.
        """
        for pdb_multi, pdb_single in self._get_all_serine_monomer_crystals():
            fmodel_mtz = self._create_fmodel(pdb_multi, high_resolution=1.4)
            self._run_transformer(pdb_multi, fmodel_mtz, 0.998)
