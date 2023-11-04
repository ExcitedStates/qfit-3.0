import tempfile
import os.path as op
import subprocess

from qfit.structure import Structure
from qfit.utils.mock_utils import BaseTestRunner


class TestCommandLineTools(BaseTestRunner):

    def test_remove_altconfs(self):
        pdb_multi, _ = self._get_serine_monomer_inputs()
        pdb_in = Structure.fromfile(pdb_multi)
        assert len(list(pdb_in.residues)) == 2
        tmp_out = tempfile.mkdtemp()
        print(pdb_multi)
        args = [
            "remove_altconfs", pdb_multi, "-d", tmp_out
        ]
        subprocess.check_call(args)
        pdb_single = op.join(tmp_out, op.basename(pdb_multi)[:-4] + ".single.pdb")
        pdb_out = Structure.fromfile(pdb_single)
        assert len(list(pdb_out.residues)) == 1
        for atom in pdb_out.atoms:
            assert atom.occ == 1.0
            assert atom.parent().altloc == ""

    def test_qfit_tools_help(self):
        TOOLS = [
            "qfit_protein",
            "qfit_ligand",
            "qfit_density",
            "remove_altconfs"
        ]
        for tool_name in TOOLS:
            subprocess.check_call([tool_name, "--help"])
