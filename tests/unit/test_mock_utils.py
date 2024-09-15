import os.path as op

from qfit.structure import Structure
from qfit.utils.mock_utils import BaseTestRunner


class TestMockUtils(BaseTestRunner):
    DATA = op.join(op.dirname(op.dirname(__file__)), "data")

    def test_check_max_rmsd(self):
        s = Structure.fromfile(op.join(self.DATA, "AWA_3conf.pdb"))
        altA = s.extract("altloc", "A")
        altB = s.extract("altloc", "B")
        altC = s.extract("altloc", "C")
        self._check_max_rmsd([altA, altB, altC],
                             expected_global_rmsd=2.69,
                             expected_atom_rmsds=[("CH2", 4.77)])
