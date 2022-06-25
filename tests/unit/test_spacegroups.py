import pytest

from qfit.spacegroups import SpaceGroup, SpaceGroupList, GetSpaceGroup

from .base_test_case import UnitBase

def _compare_symops(g1, g2):
    assert len(g1.symop_list) == len(g2.symop_list)
    for op1, op2 in zip(g1.symop_list, g2.symop_list):
        assert (op1.R == op2.R).all()
        assert (op1.t == op2.t).all()


class TestSpaceGroups(UnitBase):
    # https://www.rcsb.org/stats/distribution-space-group
    PDB_SPACE_GROUPS = ['P 21 21 21', 'P 1 21 1', 'C 1 2 1', 'C 2 2 21', 'P 21 21 2', 'P 1', 'P 43 21 2', 'P 41 21 2', 'P 32 2 1', 'P 31 2 1', 'I 2 2 2', 'P 61 2 2', 'P 65 2 2', 'H 3 2', 'H 3', 'P 61', 'P 65', 'P 63', 'P 43', 'P 2 21 21', 'P 41', 'I 4 2 2', 'P 32', 'P 31', 'I 41 2 2', 'P 42 21 2', 'P 63 2 2', 'I 4', 'P 21 3', 'I 2 3', 'P 3 2 1', 'P 4 21 2', 'P 62 2 2', 'P 64 2 2', 'I 1 2 1', 'I 41', 'P 41 2 2', 'P 43 2 2', 'P 21 2 21', 'I 21 3', 'C 2 2 2', 'I 21 21 21', 'P 64', 'P 6', 'P 1 2 1', 'P 62', 'F 4 3 2', 'P 3', 'P 41 3 2', 'F 2 2 2', 'P 2 2 21', 'I 4 3 2', 'P 32 1 2', 'P 31 1 2', 'P 43 3 2', 'P 6 2 2', 'P 4', 'F 2 3', 'P 42', 'I 41 3 2', 'F 41 3 2', 'P 2 3', 'P 42 2 2', 'P 4 3 2', 'P 4 2 2', 'P -1', 'P 42 3 2', 'B 1 1 2', 'P 3 1 2', 'P 2 2 2', 'I 1 21 1', 'P 1 1 21', 'P 21 2 2', 'P 1 21/c 1', 'P 1 21/n 1', 'R 3', 'R 3 2', 'C 1 2/c 1', 'C 1 21 1', 'P 2 21 2', 'I 41/a', 'P 21 21 2 A', 'H -3', 'A 1 2 1', 'C 4 21 2', 'F 4 2 2', 'P -3', 'P 1 1 2', 'P b c a', 'A 1', 'B 2 21 2', 'I -4 2 d', 'I -4 c 2', 'P 3 1 c', 'P c a b', 'P n n a']

    @pytest.mark.skip("For development purposes")
    def test_from_cctbx(self):
        for name in self.PDB_SPACE_GROUPS:
            g1 = GetSpaceGroup(name)
            g2 = SpaceGroup.from_symbol_cctbx(name)
            assert g1.number == g2.number
            assert g1.num_sym_equiv == g2.num_sym_equiv
            assert g1.num_primitive_sym_equiv == g2.num_primitive_sym_equiv
            #assert g1.pdb_name == g2.pdb_name
            assert g1.crystal_system == g2.crystal_system
            #assert g1.point_group_name == g2.point_group_name
        # loop over all groups recognized by CCTBX
        for g1 in sorted(SpaceGroupList, key=lambda x: x.number):
            if g1.short_name.startswith("R"):
                continue
            elif g1.number > 230:
                break
            try:
                g2 = SpaceGroup.from_symbol_cctbx(g1.pdb_name)
            except RuntimeError:
                g2 = SpaceGroup.from_symbol_cctbx(g1.short_name)
            assert g1.number == g2.number
            assert g1.num_sym_equiv == g2.num_sym_equiv
            assert g1.num_primitive_sym_equiv == g2.num_primitive_sym_equiv
            #assert g1.pdb_name == g2.pdb_name
            assert g1.crystal_system == g2.crystal_system
            # FIXME i think some of the hardcoded point groups are wrong
            #assert g1.point_group_name == g2.point_group_name


