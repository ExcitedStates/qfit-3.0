import tempfile
import os.path as op

import pytest
import numpy as np
from iotbx.file_reader import any_file
import iotbx.ccp4_map
from cctbx import maptbx
from scitbx.array_family import flex
import boost_adaptbx.boost.python as bp
asu_map_ext = bp.import_ext("cctbx_asymmetric_map_ext")

from qfit.xtal.volume import XMap

from .base_test_case import UnitBase


class TestVolumeXmapIO(UnitBase):
    def test_xmap_from_mtz_water(self):
        d_min = 2.0
        def _validate_map(xmap):
            assert xmap.resolution.high == pytest.approx(d_min, abs=0.001)
            assert (
                str(xmap.unit_cell)
                == "UnitCell(a=4.000000, b=5.000000, c=6.000000, alpha=90.000000, beta=90.000000, gamma=90.000000)"
            )
            assert xmap.array.size == 1620
            assert tuple(xmap.unit_cell_shape) == (9, 12, 15)
            assert tuple(xmap.origin) == (0.0, 0.0, 0.0)
            assert xmap.n_real() == (9, 12, 15)
            # assert list(xmap.array[0:5]) == ""
            assert xmap.value_at(0, 0, 0) == pytest.approx(-0.719, abs=0.001)
            assert xmap.value_at(4, 5, 6) == pytest.approx(4.1893, abs=0.001)
            assert tuple(xmap.offset) == (0, 0, 0)
            symm = xmap.get_p1_crystal_symmetry()
            assert symm.unit_cell().parameters() == (4.0, 5.0, 6.0, 90, 90, 90)
            assert str(symm.space_group_info()) == "P 1"

        MTZ = self.make_water_fmodel_mtz(d_min)
        print(f"MTZ: {MTZ}")
        xmap1 = XMap.fromfile(MTZ, label="FWT,PHIFWT")
        # print([str(x) for x in xmap1.unit_cell.space_group.symop_list])
        _validate_map(xmap1)
        map_tmp = tempfile.NamedTemporaryFile(suffix=".ccp4").name
        xmap1.tofile(map_tmp)
        xmap2 = XMap.fromfile(map_tmp, resolution=d_min)
        _validate_map(xmap2)

    def test_xmap_from_mtz_protein(self):
        d_min = 1.3922

        def _validate_map(xmap):
            assert xmap.resolution.high == pytest.approx(d_min, abs=0.001)
            assert (
                str(xmap.unit_cell)
                == "UnitCell(a=8.000000, b=12.000000, c=15.000000, alpha=90.000000, beta=90.000000, gamma=90.000000)"
            )
            assert xmap.array.size == 41472
            assert tuple(xmap.unit_cell_shape) == (24, 36, 48)
            assert tuple(xmap.origin) == (0.0, 0.0, 0.0)
            assert xmap.n_real() == (24, 36, 48)
            # assert list(xmap.array[0:5]) == ""
            assert xmap.value_at(0, 0, 0) == pytest.approx(-0.4325, abs=0.0001)
            assert xmap.value_at(11, 16, 20) == pytest.approx(-0.60875, abs=0.0001)
            assert tuple(xmap.offset) == (0, 0, 0)

        MTZ = self.make_tiny_fmodel_mtz()
        xmap1 = XMap.fromfile(MTZ, label="FWT,PHIFWT")
        # print([str(x) for x in xmap1.unit_cell.space_group.symop_list])
        _validate_map(xmap1)
        # XXX d_max is ignored when we read back the map file later
        assert xmap1.resolution.low == pytest.approx(9.3704, abs=0.001)
        assert xmap1.hkl.size == 354 * 3
        # I/O recycling
        map_tmp = tempfile.NamedTemporaryFile(suffix=".ccp4").name
        xmap1.tofile(map_tmp)
        xmap2 = XMap.fromfile(map_tmp, resolution=d_min)
        _validate_map(xmap2)

        # extract region
        def _validate_extract_map(xmap):
            assert xmap.resolution.high == pytest.approx(d_min, abs=0.001)
            assert (
                str(xmap.unit_cell)
                == "UnitCell(a=8.000000, b=12.000000, c=15.000000, alpha=90.000000, beta=90.000000, gamma=90.000000)"
            )
            assert tuple(xmap.unit_cell_shape) == (24, 36, 48)
            assert xmap.array.size == 33852
            assert xmap.n_real() == (28, 31, 39)
            assert tuple(xmap.origin) == (0.0, 0.0, 0.0)
            assert tuple(xmap.offset) == (-7, 11, 6)
            assert xmap.value_at(0, 0, 0) == pytest.approx(2.19792, abs=0.00001)

        coor = [[1.0, 7.0, 5.0], [4.0, 11.0, 11.0]]
        xmap3 = xmap1.extract(coor)
        _validate_extract_map(xmap3)
        xmap3.write_map_file(map_tmp)
        xmap4 = XMap.from_mapfile(map_tmp, resolution=d_min)
        _validate_extract_map(xmap4)
        # canonical unit cell
        assert xmap1.is_canonical_unit_cell()
        xmap8 = xmap1.canonical_unit_cell()
        _validate_map(xmap8)

        # test interpolation
        pdbh = any_file(self.TINY_PDB).file_object.hierarchy
        sel_cache = pdbh.atom_selection_cache()
        sel = sel_cache.iselection("not element H and (occupancy > 0.5 or bfactor < 19)")
        xyz1 = pdbh.select(sel).atoms().extract_xyz().as_numpy_array()
        sel2 = sel_cache.iselection("(name N or name CA or name C) and altloc A")
        xyz2 = pdbh.select(sel).atoms().extract_xyz().as_numpy_array()

        def _validate_interpolation(xmap):
            # XXX why does CD1 in the B conformer have 0 sigma density?
            density = xmap.interpolate(xyz1)
            assert np.all(density > 0.75)
            assert np.mean(density) > 3
            density = xmap.interpolate(xyz2)
            assert density[0] == pytest.approx(6.11, abs=0.1)
            assert density[1] == pytest.approx(4.36, abs=0.1)
            assert density[2] == pytest.approx(5.01, abs=0.1)

        _validate_interpolation(xmap1)
        _validate_interpolation(xmap2)
        # interpolate in extracted map
        density = xmap3.interpolate(xyz2)
        assert density[0] == pytest.approx(6.11, abs=0.1)
        assert density[1] == pytest.approx(4.36, abs=0.1)
        assert density[2] == pytest.approx(5.01, abs=0.1)

    def test_xmap_from_mtz_large(self):
        d_min = 1.39089

        def _validate_map(xmap):
            assert xmap.resolution.high == pytest.approx(d_min, abs=0.001)
            assert (
                str(xmap.unit_cell)
                == "UnitCell(a=43.096001, b=52.591999, c=89.249001, alpha=90.000000, beta=90.000000, gamma=90.000000)"
            )
            assert xmap.unit_cell.space_group.number == 19
            assert xmap.array.size == 5529600
            assert xmap.n_real() == (128, 160, 270)
            assert tuple(xmap.origin) == (0.0, 0.0, 0.0)
            assert tuple(xmap.offset) == (0, 0, 0)
            assert xmap.value_at(0, 0, 0) == pytest.approx(0.12935, abs=0.00001)
            assert xmap.value_at(20, 20, 20) == pytest.approx(0.44267, abs=0.00001)
            assert xmap.value_at(-1, -1, -1) == pytest.approx(-0.30319, abs=0.00001)

        mtz_file = op.join(self.DATA_BIG, "3k0n_map.mtz")
        xmap1 = XMap.fromfile(mtz_file, label="2FOFCWT,PH2FOFCWT")
        _validate_map(xmap1)
        assert xmap1.is_canonical_unit_cell()
        xmap2 = xmap1.canonical_unit_cell()
        _validate_map(xmap2)

    def test_ccp4_map_io_with_offset(self):
        d_min = 1.3922

        def _validate_map(xmap):
            assert xmap.resolution.high == pytest.approx(d_min, abs=0.001)
            assert (
                str(xmap.unit_cell)
                == "UnitCell(a=8.000000, b=12.000000, c=15.000000, alpha=90.000000, beta=90.000000, gamma=90.000000)"
            )
            assert tuple(xmap.unit_cell_shape) == (24, 36, 48)
            assert xmap.array.size == 117992
            assert tuple(xmap.origin) == (0, 0, 0)
            assert xmap.n_real() == (43, 49, 56)
            assert xmap.value_at(0, 0, 0) == pytest.approx(0.4904, abs=0.0001)
            assert xmap.value_at(20, 20, 20) == pytest.approx(0.3415, abs=0.0001)
            assert tuple(xmap.offset) == (-15, 8, -3)

        MTZ = self.make_tiny_fmodel_mtz()
        from mmtbx.command_line import mtz2map
        from libtbx.utils import null_out

        tmp_dir = tempfile.mkdtemp("ccp4-map")
        args = [
            MTZ,
            self.TINY_PDB,
            f"output.directory={tmp_dir}",
            "output.prefix=tiny",
            "buffer=5.0",
            "grid_resolution_factor=0.25",
        ]
        mtz2map.run(args, log=null_out())
        map_out = op.join(tmp_dir, "tiny_1.ccp4")
        xmap = XMap.fromfile(map_out, resolution=d_min)
        _validate_map(xmap)
        # recycling
        map_tmp = tempfile.NamedTemporaryFile(suffix=".ccp4").name
        xmap.tofile(map_tmp)
        xmap2 = XMap.fromfile(map_tmp, resolution=d_min)
        _validate_map(xmap2)
        # mrc input
        map_tmp3 = tempfile.NamedTemporaryFile(suffix=".mrc").name
        xmap.write_map_file(map_tmp3)
        xmap4 = XMap.from_mapfile(map_tmp3, resolution=d_min)
        _validate_map(xmap4)

    def test_read_write_ccp4_map(self):
        d_min = 1.39

        def _validate_map(xmap):
            assert xmap.resolution.high == d_min
            assert xmap.array.size == 5529600
            assert xmap.n_real() == (128, 160, 270)
            assert (
                str(xmap.unit_cell)
                == "UnitCell(a=43.096001, b=52.591999, c=89.249001, alpha=90.000000, beta=90.000000, gamma=90.000000)"
            )
            assert xmap.value_at(0, 0, 0) == pytest.approx(0.12935, abs=0.0001)
            assert xmap.value_at(100, 100, 100) == pytest.approx(-0.7894, abs=0.0001)
            assert xmap.value_at(-1, -1, -1) == pytest.approx(-0.30319, abs=0.0001)
            assert xmap.voxelspacing[0] == pytest.approx(0.33668, abs=0.0001)
            assert xmap.voxelspacing[1] == pytest.approx(0.3287, abs=0.0001)
            assert xmap.voxelspacing[2] == pytest.approx(0.33055, abs=0.0001)
            assert tuple(xmap.offset) == (0, 0, 0)

        map_file = op.join(self.DATA_BIG, "3k0n.ccp4")
        xmap1 = XMap.fromfile(map_file, resolution=d_min)
        _validate_map(xmap1)
        map_tmp = tempfile.NamedTemporaryFile(suffix=".ccp4").name
        xmap1.tofile(map_tmp)
        xmap2 = XMap.fromfile(map_tmp, resolution=d_min)
        _validate_map(xmap2)
        map_tmp2 = tempfile.NamedTemporaryFile(suffix=".ccp4").name
        xmap2.write_map_file(map_tmp2)
        xmap3 = XMap.fromfile(map_tmp2, resolution=d_min)
        _validate_map(xmap3)
        # mrc input
        map_tmp3 = tempfile.NamedTemporaryFile(suffix=".mrc").name
        xmap3.write_map_file(map_tmp3)
        xmap4 = XMap.fromfile(map_tmp3, resolution=d_min)
        _validate_map(xmap4)
        # extract region
        coor = [[0, 7, 5], [4, 11, 11]]

        def _validate_extract_map(xmap):
            assert xmap.resolution.high == d_min
            assert xmap.n_real() == (30, 31, 37)
            assert (
                str(xmap.unit_cell)
                == "UnitCell(a=43.096001, b=52.591999, c=89.249001, alpha=90.000000, beta=90.000000, gamma=90.000000)"
            )
            assert xmap.array[0][0][0] == pytest.approx(-1.4275, abs=0.0001)
            assert tuple(xmap.offset) == (-9, 12, 6)

        xmap5 = xmap1.extract(coor)
        _validate_extract_map(xmap5)
        xmap5.write_map_file(map_tmp)
        xmap6 = XMap.from_mapfile(map_tmp, resolution=d_min)
        _validate_extract_map(xmap6)

    # XXX not sure this still adds much value...
    def test_cctbx_io_with_fft(self):
        MTZ = self.make_tiny_fmodel_mtz()
        miller_arrays = any_file(MTZ).file_object.as_miller_arrays()
        coeffs = miller_arrays[0]
        fft_map = coeffs.fft_map(
            resolution_factor=0.25,
            symmetry_flags=maptbx.use_space_group_symmetry)
        fft_map.apply_sigma_scaling()
        tmp_map = tempfile.NamedTemporaryFile(suffix="map.ccp4").name
        fft_map.as_ccp4_map(tmp_map)
        d_min = coeffs.d_min()
        def _validate_map(xmap):
            assert xmap.resolution.high == pytest.approx(d_min, abs=0.001)
            assert (
                str(xmap.unit_cell)
                == "UnitCell(a=8.000000, b=12.000000, c=15.000000, alpha=90.000000, beta=90.000000, gamma=90.000000)"
            )
            # These are parameters expected from CCTBX FFT gridding
            assert xmap.array.size == 41472
            assert tuple(xmap.unit_cell_shape) == (24, 36, 48)
            assert tuple(xmap.origin) == (0.0, 0.0, 0.0)
            assert xmap.n_real() == (24, 36, 48)
            # assert list(xmap.array[0:5]) == ""
            assert xmap.value_at(0, 0, 0) == pytest.approx(-0.4325, abs=0.0001)
            assert xmap.value_at(22, 18, 12) == pytest.approx(-0.55146, abs=0.0001)
            assert tuple(xmap.offset) == (0, 0, 0)
        xmap1 = XMap.fromfile(tmp_map, resolution=coeffs.d_min())
        _validate_map(xmap1)
        xmap2 = XMap.fromfile(MTZ, label="FWT,PHIFWT")
        _validate_map(xmap2)

    def test_canonical_unit_cell(self):
        tmp_mtz = self.make_tiny_fmodel_mtz()
        xmap_mtz = XMap.fromfile(tmp_mtz, label="FWT,PHIFWT")
        assert xmap_mtz.n_real() == (24, 36, 48)
        assert np.all(xmap_mtz.unit_cell_shape == (24, 36, 48))
        coeffs = any_file(tmp_mtz).file_object.as_miller_arrays()[0]
        fft_map = coeffs.fft_map(
            resolution_factor=0.25,
            symmetry_flags=maptbx.use_space_group_symmetry)
        real_map = fft_map.apply_sigma_scaling().real_map_unpadded()
        # Create ASU version of map
        asu_map = asu_map_ext.asymmetric_map(coeffs.space_group().type(),
                                             real_map)
        asu_rm = asu_map.data()
        tmp_map = tempfile.NamedTemporaryFile(suffix="map.ccp4").name
        iotbx.ccp4_map.write_ccp4_map(  # pylint: disable=no-member
            file_name=tmp_map,
            unit_cell=coeffs.unit_cell(),
            space_group=coeffs.space_group(),
            unit_cell_grid=real_map.focus(),
            map_data=asu_rm,
            labels=flex.std_string(["qfit-test"]))  # pylint: disable=no-member
        # now load as a qFit map and expand to p1
        xmap_asu = XMap.fromfile(tmp_map, resolution=coeffs.d_min())
        assert xmap_asu.n_real() == (13, 19, 49)
        assert np.all(xmap_asu.unit_cell_shape == [24, 36, 48])
        assert not xmap_asu.is_canonical_unit_cell()
        assert xmap_mtz.is_canonical_unit_cell()
        xmap_p1 = xmap_asu.canonical_unit_cell()
        assert xmap_p1.shape == xmap_mtz.shape
        assert np.all(xmap_p1.unit_cell_shape == xmap_mtz.unit_cell_shape)
        corr = np.corrcoef(xmap_p1.array.flatten(), xmap_mtz.array.flatten())
        assert np.all(corr[0][1] >= 0.99999999999)
