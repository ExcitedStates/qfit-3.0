import copy
import os.path as op

import numpy as np

from qfit.structure import Structure
from qfit.transformer import FFTTransformer
from qfit.volume import XMap

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
        self._create_fmodel(pdb_multi, high_resolution=d_min)
        self._run_transformer(pdb_multi, "fmodel.mtz", corr_min)

    def test_transformer_water_p1(self):
        pdb_file = self._get_file_path("water.pdb")
        self._create_fmodel(pdb_file, high_resolution=2.0)
        self._run_transformer(pdb_file, "fmodel.mtz")

    def test_transformer_ser_p1(self):
        self._run_all("Ser", 1.5)

    def test_transformer_3mer_ser_p21(self):
        self._run_all("ASA", 1.5)

    def test_transformer_3mer_lys_p21(self):
        self._run_all("AKA", 1.2)

    def test_transformer_3mer_trp_3conf_p21(self):
        pdb_multi = self._get_file_path("AWA_2conf.pdb")
        pdb_single = self._get_file_path("AWA_single.pdb")
        self._create_fmodel(pdb_multi, high_resolution=0.8)
        self._run_transformer(pdb_multi, "fmodel.mtz")

    def test_transformer_3mer_lys_p6322(self):
        pdb_multi = self._get_file_path("AKA_p6322_3conf.pdb")
        pdb_single_start = self._get_file_path("AKA_p6322_single.pdb")
        assert op.isfile(pdb_multi)
        self._create_fmodel(pdb_multi, high_resolution=0.8)
        self._run_transformer(pdb_multi, "fmodel.mtz")
