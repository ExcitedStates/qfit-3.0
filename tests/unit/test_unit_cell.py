import unittest

import numpy as np
import pytest

from qfit.unitcell import UnitCell

class TestUnitCell(unittest.TestCase):

    def test_unit_cell_p2(self):
        uc = UnitCell(40, 50, 60, 90, 108, 90, "P2")
        assert uc.calc_v() == pytest.approx(0.9510565, abs=0.000001)
        assert uc.calc_volume() == pytest.approx(114126.78196, abs=0.00001)
        assert list(uc.abc) == [40, 50, 60]
        coords = uc.calc_frac_to_orth((0.5,0.5,0.5))
        assert coords[0] == pytest.approx(10.7295, abs=0.0001)
        assert coords[1] == pytest.approx(25, abs=0.0000000001)
        assert coords[2] == pytest.approx(28.5317, abs=0.0001)
        coords_frac = uc.calc_orth_to_frac(coords)
        assert coords_frac[0] == pytest.approx(0.5, abs=0.0000001)
        assert coords_frac[1] == pytest.approx(0.5, abs=0.0000001)
        assert coords_frac[2] == pytest.approx(0.5, abs=0.0000001)
        uc2 = uc.copy()
        assert np.all(uc2.abc == uc.abc)
        assert uc2.calc_v() == pytest.approx(0.9510565, abs=0.000001)
        assert uc2.calc_volume() == pytest.approx(114126.78196, abs=0.00001)

    def test_unit_cell_p212121(self):
        uc = UnitCell(40, 50, 60, 90, 90, 90, "P212121")
        assert uc.calc_v() == 1.0
        assert uc.calc_volume() == 120000
        coords = uc.calc_frac_to_orth((0.5,0.5,0.5))
        assert coords[0] == pytest.approx(20, abs=0.0000000001)
        assert coords[1] == pytest.approx(25, abs=0.0000000001)
        assert coords[2] == pytest.approx(30, abs=0.0000000001)
        coords_frac = uc.calc_orth_to_frac(coords)
        assert coords_frac[0] == pytest.approx(0.5, abs=0.0000001)
        assert coords_frac[1] == pytest.approx(0.5, abs=0.0000001)
        assert coords_frac[2] == pytest.approx(0.5, abs=0.0000001)

    def test_unit_cell_p6322(self):
        uc = UnitCell(40, 40, 60, 90, 90, 120, "P6322")
        assert uc.calc_v() == pytest.approx(0.8660254, abs=0.000001)
        assert uc.calc_volume() == pytest.approx(83138.43876, abs=0.00001)
        coords = uc.calc_frac_to_orth((0.5,0.5,0.5))
        assert coords[0] == pytest.approx(10, abs=0.0000000001)
        assert coords[2] == pytest.approx(30, abs=0.0000000001)
        assert coords[1] == pytest.approx(17.3205, abs=0.0001)
        coords_frac = uc.calc_orth_to_frac(coords)
        assert coords_frac[0] == pytest.approx(0.5, abs=0.0000001)
        assert coords_frac[1] == pytest.approx(0.5, abs=0.0000001)
        assert coords_frac[2] == pytest.approx(0.5, abs=0.0000001)
