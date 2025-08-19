import tempfile
import copy
import os.path as op

import pytest

from qfit.structure import Structure
from qfit.utils.mock_utils import BaseTestRunner
from qfit.xtal.scaler import MapScaler
from qfit.xtal.transformer import get_transformer
from qfit.xtal.volume import XMap


class TestMapScaler(BaseTestRunner):
    DATA = op.join(op.dirname(__file__), "..", "qfit_ligand_test")
    MTZ = op.join(DATA, "3NM0_composite_omit_map.mtz")
    PDB = op.join(DATA, "3NM0.pdb")
    MTZ2 = op.join(DATA, "5C40_composite_omit_map.mtz")
    PDB2 = op.join(DATA, "5C40.pdb")

    def test_map_scaler_water_synthetic_data(self):
        transformer = "qfit"
        pdb_tmp = self._get_water_pdb()
        mtz_out = tempfile.NamedTemporaryFile(suffix="-water-fmodel.mtz").name
        self._create_fmodel(pdb_tmp, 1.0, mtz_out)
        xmap1 = XMap.fromfile(mtz_out, label="FWT,PHIFWT", transformer=transformer)
        xmap_start = copy.deepcopy(xmap1)
        structure = Structure.fromfile(pdb_tmp)
        scaler = MapScaler(xmap1)
        (s, k) = scaler.scale(structure, transformer=transformer)
        # XXX I wish I understood where these numbers come from
        assert s == pytest.approx(0.327, abs=0.001)
        assert k == pytest.approx(0.056, abs=0.001)
        # test qfit transformer
        xmap_qfit = XMap.fromfile(mtz_out, label="FWT,PHIFWT", transformer=transformer)
        scaler_qfit = MapScaler(xmap_qfit)
        (s, k) = scaler_qfit.scale(structure, transformer=transformer)
        assert s == pytest.approx(0.32, abs=0.01)
        assert k == pytest.approx(0.056, abs=0.001)
        # test fft transformer
        xmap2 = XMap.fromfile(mtz_out, label="FWT,PHIFWT", transformer=transformer)
        scaler_fft = MapScaler(xmap2)
        (s, k) = scaler_fft.scale(structure, enable_fft=True, transformer=transformer)
        assert s == pytest.approx(0.32, abs=0.01)
        assert k == pytest.approx(0.056, abs=0.001)
        # density recycling
        xmap3 = copy.deepcopy(xmap_start)
        tx = get_transformer(transformer, structure, xmap3)
        tx.reset(full=True)
        tx.density()
        scaler3 = MapScaler(xmap3)
        (s, k) = scaler3.scale(structure, transformer=transformer)
        assert s == pytest.approx(1.0, abs=0.001)
        # FIXME why does k = 0.138?
        #assert k == pytest.approx(0.0, abs=0.0001)

    def test_map_scaler_tripeptide_synthetic_data(self):
        transformer = "cctbx"
        pdb_multi = op.join(self.DATA, "..", "data", "AKA_p6322_3conf.pdb")
        fmodel_mtz = self._create_fmodel(pdb_multi, high_resolution=0.8)
        xmap1 = XMap.fromfile(fmodel_mtz, label="FWT,PHIFWT", transformer=transformer)
        structure = Structure.fromfile(pdb_multi)
        scaler = MapScaler(xmap1)
        (s, k) = scaler.scale(structure)
        assert s == pytest.approx(0.2513, abs=0.001)
        assert k == pytest.approx(0.0480, abs=0.001)
        # test qFit transformer
        xmap_qfit = XMap.fromfile(fmodel_mtz, label="FWT,PHIFWT",
                                  transformer=transformer)
        scaler_qfit = MapScaler(xmap_qfit)
        (s, k) = scaler_qfit.scale(structure, transformer=transformer)
        assert s == pytest.approx(0.235, abs=0.01)
        assert k == pytest.approx(0.103, abs=0.001)
        # test fft transformer
        xmap2 = XMap.fromfile(fmodel_mtz, label="FWT,PHIFWT", transformer=transformer)
        scaler_fft = MapScaler(xmap2)
        (s, k) = scaler_fft.scale(structure, enable_fft=True)
        assert s == pytest.approx(0.251, abs=0.01)
        assert k == pytest.approx(0.048, abs=0.001)
        # density recycling
        xmap3 = copy.deepcopy(xmap1)
        tx = get_transformer(transformer, structure, xmap3)
        tx.reset(full=True)
        tx.density()
        scaler3 = MapScaler(xmap3)
        (s, k) = scaler3.scale(structure, transformer=transformer)
        assert s == pytest.approx(1.0, abs=0.001)
        assert k == pytest.approx(0.0, abs=0.002)

    def test_map_scaler_3nm0(self):
        structure = Structure.fromfile(self.PDB)
        xmap = XMap.fromfile(self.MTZ, label="2FOFCWT,PH2FOFCWT", transformer="cctbx")
        scaler = MapScaler(xmap)
        (s, k) = scaler.scale(structure, transformer="cctbx")
        assert s == pytest.approx(0.18, abs=0.001)
        assert k == pytest.approx(0.4908, abs=0.0001)
        # density recycling
        xmap2 = copy.deepcopy(xmap)
        tx = get_transformer("cctbx", structure, xmap2)
        tx.reset(full=True)
        tx.density()
        scaler2 = MapScaler(xmap2)
        (s, k) = scaler2.scale(structure, transformer="cctbx")
        assert s == 1.0
        assert k == 0.0

    def test_map_scaler_qfit_3nm0(self):
        structure = Structure.fromfile(self.PDB)
        xmap = XMap.fromfile(self.MTZ, label="2FOFCWT,PH2FOFCWT")
        scaler = MapScaler(xmap)
        (s, k) = scaler.scale(structure, transformer="qfit")
        assert s == pytest.approx(0.185, abs=0.001)
        assert k == pytest.approx(0.512, abs=0.01)

    def test_map_scaler_5c40(self):
        structure = Structure.fromfile(self.PDB2)
        xmap = XMap.fromfile(self.MTZ2, label="2FOFCWT,PH2FOFCWT")
        scaler = MapScaler(xmap)
        radius = 0.5 + xmap.resolution.high / 3.0
        (s, k) = scaler.scale(structure, radius=radius, transformer="cctbx")
        assert s == pytest.approx(0.3287, abs=0.001)
        assert k == pytest.approx(0.418, abs=0.001)

    def test_map_scaler_qfit_5c40(self):
        structure = Structure.fromfile(self.PDB2)
        xmap = XMap.fromfile(self.MTZ2,
                             label="2FOFCWT,PH2FOFCWT",
                             transformer="qfit")
        scaler = MapScaler(xmap)
        (s, k) = scaler.scale(structure, transformer="qfit")
        assert s == pytest.approx(0.3228, abs=0.001)
        assert k == pytest.approx(0.488, abs=0.01)
