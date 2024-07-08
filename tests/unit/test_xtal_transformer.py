import numpy as np
import pytest

from qfit.structure import Structure
from qfit.xtal.transformer import get_transformer, FFTTransformer
from qfit.xtal.volume import XMap

from .base_test_case import UnitBase


class TransformerBase(UnitBase):
    IMPLEMENTATION = "cctbx"

    def _get_transformer(self, *args, **kwds):
        return get_transformer(self.IMPLEMENTATION, *args, **kwds)

    def _load_qfit_inputs(self, pdb_file, mtz_file):
        structure = Structure.fromfile(pdb_file)
        xmap = XMap.fromfile(mtz_file, label="FWT,PHIFWT")
        return structure, xmap

    def _get_lys_3mer_p6322_fmodel(self):
        pdb_multi = self._get_file_path("AKA_p6322_3conf.pdb")
        fmodel_mtz = self._create_fmodel(pdb_multi, high_resolution=0.8)
        return pdb_multi, fmodel_mtz

    def _run_transformer(self, structure, xmap, cc_min=0.99):
        rmask = 0.5 + xmap.resolution.high / 3.0  # from qfit.py
        xmap_orig = xmap.array.copy()
        t = self._get_transformer(structure, xmap)
        t.reset(full=True)
        t.mask(rmax=rmask)
        mask = t.get_masked_selection()
        t.reset(full=True)
        t.density()
        xmap_new = t.xmap.array.copy()
        ccs = np.corrcoef(xmap_new[mask].flatten(), xmap_orig[mask].flatten())
        assert ccs[0][1] >= cc_min
        t.reset(full=True)
        if self.IMPLEMENTATION == "cctbx":
            for dens in t.get_conformers_densities([structure.coor],
                                                   [structure.b]):
                ccs = np.corrcoef(xmap_new[mask].flatten(),
                                  dens[mask].flatten())
                assert ccs[0][1] >= 0.999999999


class TestTransformer(TransformerBase):

    def _run_all(self, pdb_multi, d_min, cc_min=0.99):
        fmodel_mtz = self._create_fmodel(pdb_multi, high_resolution=d_min)
        structure, xmap = self._load_qfit_inputs(pdb_multi, fmodel_mtz)
        self._run_transformer(structure, xmap, cc_min)

    def _run_3mer(self, peptide_name, d_min, cc_min=0.99):
        pdb_multi, pdb_single = self._get_start_models(peptide_name)
        self._run_all(pdb_multi, d_min, cc_min=cc_min)

    @pytest.mark.fast
    def test_transformer_3mer_ser_p21(self):
        self._run_3mer("ASA", d_min=1.5)

    @pytest.mark.fast
    def test_transformer_3mer_lys_p21(self):
        self._run_3mer("AKA", d_min=1.2, cc_min=0.935)

    def test_transformer_mask_water_p1(self):
        """
        Test mask calculations at different resolutions, for a single water
        molecule (without hydrogens) in an asymmetrical P1 box.
        """
        pdb_file = self._get_water_pdb()
        RESO = [1.0, 1.5, 2.0]
        SHAPES = [(18, 24, 25), (12, 15, 18), (9, 12, 15)]
        # XXX these are the original output (as number of masked grid points)
        # of the old implementation using the mask_points C extension, with
        # the rmask as determined below.  they are reproducible using the
        # CCTBX around_atoms mask implementation
        SIZES = [206, 110, 88]
        for d_min, shape, msize in zip(RESO, SHAPES, SIZES):
            fmodel_mtz = self._create_fmodel(pdb_file, high_resolution=d_min)
            structure, xmap = self._load_qfit_inputs(pdb_file, fmodel_mtz)
            assert xmap.n_real() == shape
            t = self._get_transformer(structure, xmap)
            rmask = 0.5 + d_min / 3.0  # from qfit.py
            t.reset(full=True)
            t.mask(rmax=rmask)
            mask = t.xmap.array > 0
            assert np.sum(mask) == msize
            # repeat with extracted map - masked area should have same size
            # XXX note that the padding here is less than default, to deal
            # with the artifically small unit cell
            xsub = xmap.extract(structure.coor, padding=2.0)
            assert xsub.n_real() != shape
            t2 = self._get_transformer(structure, xsub)
            t2.reset(full=True)
            t2.mask(rmax=rmask)
            mask2 = t2.xmap.array > 0
            assert np.sum(mask2) == msize

    def test_transformer_mask_p21(self):
        """
        Test mask calculation at various resolutions and radii, using the
        central Ser residue in a 3-mer in a P21 box.
        """
        pdb_multi, pdb_single = self._get_start_models("ASA")
        RESOLUTIONS = [1.0, 2.0]
        SHAPES = [(38, 36, 39), (20, 19, 22)]
        # XXX these are values if symmetry is applied
        #MASKS = [(2682, 7362, 11132), (351, 991, 1497)]
        MASKS = [(1341, 3700, 5650), (176, 495, 749)]
        RMAXES = (0.8333333333, 1.25, 1.5)
        for d_min, shape, masks in zip(RESOLUTIONS, SHAPES, MASKS):
            fmodel_mtz = self._create_fmodel(pdb_multi, high_resolution=d_min)
            structure, xmap = self._load_qfit_inputs(pdb_multi, fmodel_mtz)
            ssub = structure.extract("resi", 2)
            assert str(ssub.crystal_symmetry.space_group_info()) == "P 1 21 1"
            xsub = xmap.extract(ssub.coor)
            #xsub.tofile("ASA_extract.ccp4")
            assert xsub.n_real() == shape
            t = self._get_transformer(ssub, xsub)
            for rmax, mask_size in zip(RMAXES, masks):
                t.reset(full=True)
                t.mask(rmax=rmax)
                #t.xmap.tofile(f"ASA_mask_{impl}.ccp4")
                mask = t.xmap.array > 0
                assert np.sum(mask) == mask_size

    def test_transformer_3mer_lys_p6322(self):
        pdb_multi, fmodel_mtz = self._get_lys_3mer_p6322_fmodel()
        structure, xmap = self._load_qfit_inputs(pdb_multi, fmodel_mtz)
        self._run_transformer(structure, xmap, cc_min=0.85)

    def _run_transformer_rebuilt_3mer(self, resname, d_min=1.5, cc_min=1.0):
        pdb_multi, pdb_single = self._create_mock_multi_conf_3mer(resname)
        fmodel_mtz = self._create_fmodel(pdb_multi, high_resolution=d_min)
        structure, xmap = self._load_qfit_inputs(pdb_multi, fmodel_mtz)
        ssub = structure.extract("resi", 2)
        xsub = xmap.extract(ssub.coor)
        self._run_transformer(ssub, xsub, cc_min)

    #MIN_CC_GLU = 0.94
    #MIN_CC_LYS = 0.945
    #MIN_CC_TRP = 0.95
    # TODO figure out why these values are lower than in the old qFit
    # implementation or the FFT
    MIN_CC_GLU = 0.929
    MIN_CC_LYS = 0.928
    MIN_CC_TRP = 0.934

    def test_transformer_rebuilt_tripeptide_glu(self):
        self._run_transformer_rebuilt_3mer("GLU", cc_min=self.MIN_CC_GLU)

    def test_transformer_rebuilt_tripeptide_lys(self):
        self._run_transformer_rebuilt_3mer("LYS", cc_min=self.MIN_CC_LYS)

    def test_transformer_rebuilt_tripeptide_ser(self):
        self._run_transformer_rebuilt_3mer("SER", cc_min=0.899)

    def test_transformer_rebuilt_tripeptide_trp(self):
        self._run_transformer_rebuilt_3mer("TRP", cc_min=self.MIN_CC_TRP)


class TestFFTTransformer(TestTransformer):
    IMPLEMENTATION = "fft"
    MIN_CC_GLU = 0.94
    MIN_CC_LYS = 0.945
    MIN_CC_TRP = 0.95

    def _run_transformer_rebuilt_3mer(self, *args, **kwds):
        pytest.skip("not applicable to this class")


class TestCompareTransformers(TransformerBase):
    IMPLEMENTATION = "cctbx"

    def _run_fft_transformer(self, pdb_multi, mtz_file, cc_min=0.9):
        structure, xmap = self._load_qfit_inputs(pdb_multi, mtz_file)
        map_data = xmap.array.copy().flatten()
        transformer = FFTTransformer(structure, xmap)
        assert transformer.hkl is not None
        transformer.reset(full=True)
        transformer.density()
        map_data2 = xmap.array.copy().flatten()
        assert np.corrcoef(map_data, map_data2)[0][1] > cc_min
        return transformer

    def _run_all_fft(self, peptide_name, d_min, cc_min=0.99):
        pdb_multi, pdb_single = self._get_start_models(peptide_name)
        fmodel_mtz = self._create_fmodel(pdb_multi, high_resolution=d_min)
        self._run_fft_transformer(pdb_multi, fmodel_mtz, cc_min)

    @pytest.mark.fast
    def test_fft_transformer_water_p1(self):
        pdb_file = self._get_water_pdb()
        fmodel_mtz = self._create_fmodel(pdb_file, high_resolution=1.5)
        t1 = self._run_fft_transformer(pdb_file, fmodel_mtz)
        structure, xmap = self._load_qfit_inputs(pdb_file, fmodel_mtz)
        t2 = self._get_transformer(structure, xmap)
        t2.reset(full=True)
        t2.density()
        ccs = np.corrcoef(t1.xmap.array.flatten(), t2.xmap.array.flatten())
        assert ccs[0][1] > 0.969

    @pytest.mark.fast
    def test_fft_transformer_3mer_ser_p21(self):
        self._run_all_fft("ASA", 1.5, cc_min=0.999)

    @pytest.mark.fast
    def test_fft_transformer_3mer_lys_p21(self):
        self._run_all_fft("AKA", 1.2, cc_min=0.915)

    def test_fft_transformer_3mer_lys_p6322(self):
        pdb_multi, fmodel_mtz = self._get_lys_3mer_p6322_fmodel()
        self._run_fft_transformer(pdb_multi, fmodel_mtz, cc_min=0.88)

    # TODO figure out why this works much worse with cctbx transform
    @pytest.mark.fast
    def test_fft_transformer_3mer_trp_3conf_p21(self):
        pdb_multi = self._get_file_path("AWA_3conf.pdb")
        pdb_single = self._get_file_path("AWA_single.pdb")
        fmodel_mtz = self._create_fmodel(pdb_multi, high_resolution=2.0)
        self._run_fft_transformer(pdb_multi, fmodel_mtz)
        structure, xmap = self._load_qfit_inputs(pdb_multi, fmodel_mtz)
        assert np.all(xmap.unit_cell_shape == [30, 20, 40])
        assert xmap.n_real() == (30, 20, 40)
        ssub = structure.extract("resi", 1)
        coor = ssub.coor
        assert len(coor) == 5
        xsub = xmap.extract(coor)
        assert np.all(xsub.unit_cell_shape == xmap.unit_cell_shape)
        assert xsub.n_real() == (18, 17, 18)
        assert np.all(xsub.offset == [-4, -3, -5])
        xsub_orig = xsub.array.copy()
        # XXX this was 0.79 with the old qfit transformer
        MASKED_MIN_CC = 0.755
        self._run_transformer(ssub, xsub, MASKED_MIN_CC)

    def test_fft_transformer_ser_monomer_space_groups(self):
        """
        Test transformer behavior with a Ser monomer in several different
        space groups.
        """
        for pdb_multi, pdb_single in self._get_all_serine_monomer_crystals():
            fmodel_mtz = self._create_fmodel(pdb_multi, high_resolution=1.4)
            self._run_fft_transformer(pdb_multi, fmodel_mtz, 0.86)
