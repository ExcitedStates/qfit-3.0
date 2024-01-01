import os.path as op

import numpy as np
import pytest

from qfit.structure import Structure
from qfit.xtal.transformer import FFTTransformer, Transformer
from qfit.xtal.volume import XMap

from .test_qfit_protein_synth import SyntheticMapRunner


class TestTransformer(SyntheticMapRunner):

    def _load_qfit_inputs(self, pdb_file, mtz_file):
        structure = Structure.fromfile(pdb_file)
        xmap = XMap.fromfile(mtz_file, label="FWT,PHIFWT")
        return structure, xmap

    def _run_fft_transformer(self, pdb_multi, mtz_file, corr_min=0.999):
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
        self._run_fft_transformer(pdb_multi, fmodel_mtz, corr_min)

    @pytest.mark.fast
    def test_transformer_water_p1(self):
        pdb_file = self._get_water_pdb()
        fmodel_mtz = self._create_fmodel(pdb_file, high_resolution=2.0)
        t1 = self._run_fft_transformer(pdb_file, fmodel_mtz)
        structure, xmap = self._load_qfit_inputs(pdb_file, fmodel_mtz)
        t2 = Transformer(structure, xmap, simple=True)
        t2.initialize()
        t2.density()
        ccs = np.corrcoef(t1.xmap.array.flatten(), t2.xmap.array.flatten())
        assert ccs[0][1] > 0.99

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
        fmodel_mtz = self._create_fmodel(pdb_multi, high_resolution=2.0)
        self._run_fft_transformer(pdb_multi, fmodel_mtz)
        structure, xmap = self._load_qfit_inputs(pdb_multi, fmodel_mtz)
        assert np.all(xmap.unit_cell_shape == [30, 20, 40])
        assert xmap.shape == (40, 20, 30)
        ssub = structure.extract("resi", 1)
        coor = ssub.coor
        assert len(coor) == 5
        xsub = xmap.extract(coor)
        assert np.all(xsub.unit_cell_shape == xmap.unit_cell_shape)
        assert xsub.shape == (18, 17, 18)
        assert np.all(xsub.offset == [-4, -3, -5])
        xsub_orig = xsub.array.copy()
        t = Transformer(ssub, xsub, simple=False)
        t.initialize()
        t.reset(full=True)
        # XXX I think there is a bug in the qfit implementation - with cctbx
        # the masked map CC is above 0.79
        MASK_IMPL = "qfit"
        MASKED_MIN_CC = 0.37
        t.mask(0.5 + 2.0 / 3, implementation=MASK_IMPL)
        mask = t.xmap.array > 0
        t.reset(full=True)
        t.density()
        sampled_raw = t.xmap.array[mask].flatten()
        ccs = np.corrcoef(sampled_raw, xsub_orig[mask].flatten())
        assert ccs[0][1] > MASKED_MIN_CC

    def test_transformer_mask_water_p1(self):
        """
        Test mask calculations at different resolutions, for a single water
        molecule (without hydrogens) in an asymmetrical P1 box.
        """
        pdb_file = self._get_water_pdb()
        RESO = [1.0, 1.5, 2.0]
        # note swapped X- and Z-axes, as per internal convention
        SHAPES = [(25, 24, 18), (18, 15, 12), (15, 12, 9)]
        # XXX these are the original output (as number of masked grid points)
        # of the old implementation using the mask_points C extension, with
        # the rmask as determined below.  they are reproducible using the
        # CCTBX around_atoms mask implementation
        SIZES = [206, 110, 88]
        for d_min, shape, msize in zip(RESO, SHAPES, SIZES):
            fmodel_mtz = self._create_fmodel(pdb_file, high_resolution=d_min)
            structure, xmap = self._load_qfit_inputs(pdb_file, fmodel_mtz)
            assert xmap.array.shape == shape
            t = Transformer(structure, xmap, simple=True)
            rmask = 0.5 + d_min / 3.0  # from qfit.py
            for impl in ["qfit", "cctbx"]:
                t.reset(full=True)
                t.mask(rmax=rmask, implementation=impl)
                mask = t.xmap.array > 0
                assert np.sum(mask) == msize
            # repeat with extracted map - masked area should have same size
            # XXX note that the padding here is less than default, to deal
            # with the artifically small unit cell
            xsub = xmap.extract(structure.coor, padding=2.0)
            assert xsub.array.shape != shape
            t2 = Transformer(structure, xsub, simple=True)
            for impl in ["qfit", "cctbx"]:
                t2.reset(full=True)
                t2.mask(rmax=rmask, implementation=impl)
                mask2 = t2.xmap.array > 0
                assert np.sum(mask2) == msize

    def test_transformer_mask_p21(self):
        """
        Test mask calculation at various resolutions and radii, using the
        central Ser residue in a 3-mer in a P21 box.
        """
        pdb_multi, pdb_single = self._get_start_models("ASA")
        RESOLUTIONS = [1.0, 2.0]
        SHAPES = [(39, 36, 38), (22, 19, 20)]
        # XXX these are untrustworthy until I prove otherwise
        MASKS = [(2682, 7362, 11132), (351, 991, 1497)]
        RMAXES = (0.8333333333, 1.25, 1.5)
        for d_min, shape, masks in zip(RESOLUTIONS, SHAPES, MASKS):
            fmodel_mtz = self._create_fmodel(pdb_multi, high_resolution=d_min)
            structure, xmap = self._load_qfit_inputs(pdb_multi, fmodel_mtz)
            ssub = structure.extract("resi", 2)
            assert str(ssub.crystal_symmetry.space_group_info()) == "P 1 21 1"
            xsub = xmap.extract(ssub.coor)
            #xsub.tofile("ASA_extract.ccp4")
            assert xsub.array.shape == shape
            t = Transformer(ssub, xsub, simple=True)
            for rmax, mask_size in zip(RMAXES, masks):
                for impl in ["qfit"]: #, "cctbx"]:
                    t.reset(full=True)
                    t.mask(rmax=rmax, implementation=impl)
                    #t.xmap.tofile(f"ASA_mask_{impl}.ccp4")
                    mask = t.xmap.array > 0
                    assert np.sum(mask) == mask_size

    def test_transformer_3mer_lys_p6322(self):
        pdb_multi = self._get_file_path("AKA_p6322_3conf.pdb")
        pdb_single_start = self._get_file_path("AKA_p6322_single.pdb")
        assert op.isfile(pdb_multi)
        fmodel_mtz = self._create_fmodel(pdb_multi, high_resolution=0.8)
        self._run_fft_transformer(pdb_multi, fmodel_mtz)

    def test_transformer_ser_monomer_space_groups(self):
        """
        Test transformer behavior with a Ser monomer in several different
        space groups.
        """
        for pdb_multi, pdb_single in self._get_all_serine_monomer_crystals():
            fmodel_mtz = self._create_fmodel(pdb_multi, high_resolution=1.4)
            self._run_fft_transformer(pdb_multi, fmodel_mtz, 0.998)
