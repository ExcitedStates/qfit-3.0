import os.path as op

import pytest

from qfit.structure import Structure
from qfit.structure.ligand import Ligand

from .base_test_case import UnitBase


class TestStructureLigand(UnitBase):
    @pytest.mark.skip("TODO - need a good input example")
    def test_ligand_from_cif(self):
        PDB = op.join(self.DATA, "3ov1_ACT.pdb")
        CIF = op.join(self.DATA, "ACT.cif")
        structure = Structure.fromfile(PDB)
        structure_ligand = structure.extract("resi 1 and chain B")
        ligand = Ligand.from_structure(structure_ligand, CIF)
        assert ligand.connectivity == []  # FIXME
