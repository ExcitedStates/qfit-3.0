"""
Additional utilities for unit-testing with mock data
"""

import tempfile
import unittest
import os.path as op

from qfit.utils.test_utils import BaseTestRunner

class UnitBase(BaseTestRunner):
    TESTS_DIR = op.dirname(op.dirname(__file__))
    DATA = op.join(TESTS_DIR, "data")
    DATA_BIG = op.join(TESTS_DIR, "basic_qfit_protein_test")
    TINY_PDB = op.join(DATA, "phe113_fake_uc.pdb")
    TINY_CIF = op.join(DATA, "phe113_fake_uc.cif.gz")
    EXAMPLES = op.join(op.dirname(TESTS_DIR), "example")

    def make_tiny_fmodel_mtz(self):
        """
        Use 'phenix.fmodel' to generate Fcalc map coefficients in MTZ format
        """
        mtz_out = tempfile.NamedTemporaryFile(suffix="-fmodel.mtz").name
        self._create_fmodel(self.TINY_PDB, 1.39, mtz_out)
        print(mtz_out)
        return mtz_out

    def make_water_fmodel_mtz(self, d_min):
        pdb_tmp = self._get_water_pdb()
        mtz_out = tempfile.NamedTemporaryFile(suffix="-water-fmodel.mtz").name
        return self._create_fmodel(pdb_tmp, d_min, mtz_out)
