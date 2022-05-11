import unittest
import os.path as op

import pytest

from qfit.volume import XMap

from .base_test_case import UnitBase


class TestVolume(UnitBase):
    def test_xmap_from_mtz(self):
        MTZ = self.make_tiny_fmodel_mtz()
        xmap = XMap.fromfile(MTZ, label="FWT,PHIFWT")
        assert xmap.resolution.high == pytest.approx(1.3922, abs=0.001)
        assert xmap.resolution.low == pytest.approx(9.3704, abs=0.001)
        assert (
            str(xmap.unit_cell)
            == "UnitCell(a=8.000000, b=12.000000, c=15.000000, alpha=90.000000, beta=90.000000, gamma=90.000000)"
        )
        assert xmap.array.size == 28160
        assert xmap.hkl.size == 354 * 3
        assert list(xmap.origin) == [0.0, 0.0, 0.0]
        assert xmap.array.shape == (40, 32, 22)
        # assert list(xmap.array[0:5]) == ""
        assert xmap.array[0][0][0] == pytest.approx(-0.4325, abs=0.0001)
        assert xmap.array[20][16][11] == pytest.approx(0.1438, abs=0.0001)
