"""
Fast qfit_protein integration tests using ccp4 maps as input
"""

# FIXME resolve different outcomes from MTZ-input base class

import subprocess
import os.path as op
import os

import pytest

from qfit.structure import Structure
from qfit.volume import XMap

from .test_qfit_protein_synth import TestQfitProteinSyntheticData as _BaseClass

# workaround for pytest discovery behavior
_BaseClass.__test__ = False


@pytest.mark.skip("For development purposes, redundant with MTZ-driven test")
class TestQfitProteinSyntheticDataCcp4Map(_BaseClass):
    __test__ = True

    def _run_qfit_cli(self, pdb_file_multi, pdb_file_single, high_resolution):
        self._create_fmodel(pdb_file_multi, high_resolution=high_resolution)
        os.symlink(pdb_file_single, "single.pdb")
        xmap = XMap.from_mtz("fmodel.mtz", label="FWT,PHIFWT")
        xmap.tofile("fmodel_1.ccp4")
        qfit_args = [
            "qfit_protein",
            "fmodel_1.ccp4",
            pdb_file_single,
            "--resolution",
            str(high_resolution),
            "--backbone-amplitude",
            "0.1",
            "--rotamer-neighborhood",
            "10",
            "--debug",
        ]
        print(" ".join(qfit_args))
        return subprocess.check_call(qfit_args)
