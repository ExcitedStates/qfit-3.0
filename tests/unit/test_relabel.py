"""Tests for qfit.relabel."""

import os
import subprocess
import sys

from .base_test_case import UnitBase

_RELABEL_SCRIPT = """
import sys
from qfit.structure import Structure
from qfit.relabel import Relabeller, RelabellerOptions
structure = Structure.fromfile(sys.argv[1]).reorder()
options = RelabellerOptions(nSims=200, nChains=2)
result = Relabeller(structure, options).run()
print("".join(result.altloc))
"""


class TestRelabeller(UnitBase):

    def test_labels_independent_of_hash_seed(self):
        # Altlocs used to be iterated from a set(), whose order depends on
        # PYTHONHASHSEED, so labels changed from run to run.
        pdb = self._get_file_path("SKH_multiconf.pdb")
        outputs = []
        for seed in ("0", "1", "2"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            result = subprocess.run(
                [sys.executable, "-c", _RELABEL_SCRIPT, pdb],
                env=env, capture_output=True, text=True, check=True)
            outputs.append(result.stdout.strip().splitlines()[-1])
        assert outputs[0] == outputs[1] == outputs[2]
