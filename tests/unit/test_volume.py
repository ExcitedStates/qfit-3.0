import unittest
import tempfile
import os.path as op

import pytest
import numpy as np

from qfit.volume import XMap

from .base_test_case import UnitBase


class TestVolumeXmapIO(UnitBase):
    def test_xmap_from_mtz(self):
        d_min = 1.3922

        def _validate_map(xmap):
            assert xmap.resolution.high == pytest.approx(d_min, abs=0.001)
            assert (
                str(xmap.unit_cell)
                == "UnitCell(a=8.000000, b=12.000000, c=15.000000, alpha=90.000000, beta=90.000000, gamma=90.000000)"
            )
            assert xmap.array.size == 28160
            assert tuple(xmap.unit_cell_shape) == (22, 32, 40)
            assert tuple(xmap.origin) == (0.0, 0.0, 0.0)
            assert xmap.array.shape == (40, 32, 22)
            # assert list(xmap.array[0:5]) == ""
            assert xmap.array[0][0][0] == pytest.approx(-0.4325, abs=0.0001)
            assert xmap.array[20][16][11] == pytest.approx(0.1438, abs=0.0001)
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
            assert tuple(xmap.unit_cell_shape) == (22, 32, 40)
            assert xmap.array.size == 24024
            assert xmap.array.shape == (33, 28, 26)
            assert tuple(xmap.origin) == (0.0, 0.0, 0.0)
            assert tuple(xmap.offset) == (-6, 10, 5)
            assert xmap.array[0][0][0] == pytest.approx(1.6798, abs=0.00001)

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

    def test_xmap_from_mtz_large(self):
        d_min = 1.39089

        def _validate_map(xmap):
            assert xmap.resolution.high == pytest.approx(d_min, abs=0.001)
            assert (
                str(xmap.unit_cell)
                == "UnitCell(a=43.096001, b=52.591999, c=89.249001, alpha=90.000000, beta=90.000000, gamma=90.000000)"
            )
            assert xmap.unit_cell.space_group.number == 19
            assert xmap.array.size == 4580856
            assert xmap.array.shape == (252, 149, 122)
            assert tuple(xmap.origin) == (0.0, 0.0, 0.0)
            assert tuple(xmap.offset) == (0, 0, 0)
            assert xmap.array[0][0][0] == pytest.approx(0.12935, abs=0.00001)
            assert xmap.array[20][20][20] == pytest.approx(-0.0732, abs=0.00001)
            assert xmap.array[-1][-1][-1] == pytest.approx(-0.33892, abs=0.00001)

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
            assert xmap.array.shape == (56, 49, 43)
            assert xmap.array[0][0][0] == pytest.approx(0.4904, abs=0.0001)
            assert xmap.array[20][20][20] == pytest.approx(0.3415, abs=0.0001)
            assert tuple(xmap.offset) == (-15, 8, -3)

        MTZ = self.make_tiny_fmodel_mtz()
        from mmtbx.command_line import mtz2map
        from libtbx.utils import null_out

        tmp_dir = tempfile.mkdtemp("ccp4-map")
        args = [
            MTZ,
            self.TINY_PDB,
            f"output.directory={tmp_dir}",
            f"output.prefix=tiny",
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
            assert xmap.array.shape == (270, 160, 128)
            assert (
                str(xmap.unit_cell)
                == "UnitCell(a=43.096001, b=52.591999, c=89.249001, alpha=90.000000, beta=90.000000, gamma=90.000000)"
            )
            assert xmap.array[0][0][0] == pytest.approx(0.12935, abs=0.0001)
            assert xmap.array[100][100][100] == pytest.approx(-0.7894, abs=0.0001)
            assert xmap.array[-1][-1][-1] == pytest.approx(-0.30319, abs=0.0001)
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
            assert xmap.array.shape == (37, 31, 30)
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
