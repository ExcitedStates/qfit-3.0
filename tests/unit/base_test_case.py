"""
Core utilities for unit-testing with mocked data
"""
import tempfile
import unittest
import os.path as op


class UnitBase(unittest.TestCase):
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
        from mmtbx.command_line import fmodel
        from libtbx.utils import null_out

        args = [
            self.TINY_PDB,
            "high_resolution=1.39",
            "r_free_flags_fraction=0.1",
            "output.label=FWT",
            "output.file_name={}".format(mtz_out),
        ]
        fmodel.run(args=args, log=null_out())
        print(mtz_out)
        return mtz_out
