import subprocess
import os.path as op

import numpy as np

from qfit.utils.mock_utils import BaseTestRunner
from qfit.structure import Structure


class TestRemoveDuplicatesTool(BaseTestRunner):
    DATA = op.join(op.dirname(__file__), "data")

    def test_command_line_remove_duplicates(self):
        pdb_in = op.join(self.DATA, "AWA_3conf.pdb")
        s = Structure.fromfile(pdb_in)
        pdb_tmp = "tmp_in.pdb"
        s.tofile(pdb_tmp)
        assert s.total_length == 52
        args = ["remove_duplicates", pdb_tmp]
        subprocess.check_call(args)
        s2 = Structure.fromfile(pdb_tmp + ".fixed")
        assert s2.total_length == 52
        assert np.all(s2.name == s.name)
        ss = s.copy()
        altloc = ss.altloc
        sel_c = altloc == "C"
        altloc[sel_c] = "A"
        ss.altloc = altloc
        ss.tofile("setConfCtoA.pdb")
        args = ["remove_duplicates", "setConfCtoA.pdb"]
        subprocess.check_call(args)
        ss2 = Structure.fromfile("setConfCtoA.pdb.fixed")
        assert ss2.total_length == 38
