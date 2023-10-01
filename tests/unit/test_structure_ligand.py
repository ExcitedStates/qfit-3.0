import os.path as op

import pytest

from qfit.structure import Structure
from qfit.structure.ligand import Ligand

from .base_test_case import UnitBase


class TestStructureLigand(UnitBase):

    def setUp(self):
        super().setUp()
        ppi_single_file = op.join(self.DATA, "PPI_single.pdb")
        self._STRUCTURE_PPI_SINGLE = Structure.fromfile(ppi_single_file)
        trs_single_file = op.join(self.DATA, "TRS_single.pdb")
        self._STRUCTURE_TRS_SINGLE = Structure.fromfile(trs_single_file)

    @pytest.mark.skip("TODO - need a good input example")
    def test_ligand_from_cif(self):
        PDB = op.join(self.DATA, "3ov1_ACT.pdb")
        CIF = op.join(self.DATA, "ACT.cif")
        structure = Structure.fromfile(PDB)
        structure_ligand = structure.extract("resi 1 and chain B")
        ligand = Ligand.from_structure(structure_ligand, CIF)
        assert ligand.connectivity == []  # FIXME

    def test_ligand_from_structure_ppi(self):
        s = self._STRUCTURE_PPI_SINGLE
        lig = Ligand.from_structure(s)
        assert repr(lig) == "Ligand: PPI. Number of atoms: 5."
        assert lig.id == (1, "")
        assert lig.shortcode == "1"
        assert not lig.clashes()
        assert len(lig.connectivity) == 5
        assert lig.get_bonds() == [['C1', 'C2'], ['C1', 'O1'], ['C1', 'O2'], ['C2', 'C1'], ['C2', 'C3'], ['C3', 'C2'], ['O1', 'C1'], ['O2', 'C1']]
        assert lig.rigid_clusters() == [[0, 3, 4], [1, 2]]
        assert lig.ring_paths() == []
        assert lig.rotatable_bonds() == [(0, 1)]

    def test_ligand_from_structure_tris(self):
        s = self._STRUCTURE_TRS_SINGLE
        lig = Ligand.from_structure(s)
        assert repr(lig) == "Ligand: TRS. Number of atoms: 8."
        assert lig.id == (1, "")
        assert lig.shortcode == "1"
        assert not lig.clashes()
        assert len(lig.connectivity) == 8
        assert lig.get_bonds() == [['N', 'C'], ['C', 'N'], ['C', 'C1'], ['C', 'C2'], ['C', 'C3'], ['C1', 'C'], ['C1', 'O1'], ['C2', 'C'], ['C2', 'O2'], ['C3', 'C'], ['C3', 'O3'], ['O1', 'C1'], ['O2', 'C2'], ['O3', 'C3']]
        assert lig.rigid_clusters() == [[1, 0], [2, 5], [3, 6], [4, 7]]
        assert lig.ring_paths() == []
        assert lig.rotatable_bonds() == [(1, 2), (1, 3), (1, 4)]


class TestStructureCovalentLigand(UnitBase):

    @pytest.mark.skip("TODO - need a good input example")
    def test_structure_covalent_ligand(self):
        ...
