import pytest
import os

from qfit import MapScaler, Structure, XMap
from qfit.qfit_protein import QFitProtein, QFitProteinOptions


class Args(object):
    """An empty class to collect "command-line" attributes."""

    pass


class TestQFitProtein:
    def mock_main(self):
        # Pretend that we have done parse_args, set default args
        self.args = Args()
        self.args.map = "../example/3K0N.mtz"  # relative directory from tests/
        self.args.label = "2FOFCWT,PH2FOFCWT"
        self.args.resolution = None
        self.args.scale = True
        self.args.structure = "../example/3K0N.pdb"  # relative directory from tests/
        self.args.hydro = False
        self.args.cplex = True

        # Load default options, override some to reduce computational load
        self.options = QFitProteinOptions()
        self.options.apply_command_args(self.args)
        self.options.sample_backbone_amplitude = 0.10  # default: 0.30
        self.options.rotamer_neighborhood = 30  # default: 60

        # Load structure and prepare it
        self.structure = Structure.fromfile(self.args.structure).reorder()
        if not self.args.hydro:
            self.structure = self.structure.extract("e", "H", "!=")

        # Load & prepare X-ray map
        self.xmap = XMap.fromfile(
            self.args.map, resolution=self.args.resolution, label=self.args.label
        )
        self.xmap = self.xmap.canonical_unit_cell()
        if self.args.scale:
            scaler = MapScaler(self.xmap, scattering=self.options.scattering)
            radius = 1.5
            reso = None
            if self.xmap.resolution.high is not None:
                reso = self.xmap.resolution.high
            elif self.options.resolution is not None:
                reso = self.options.resolution
            if reso is not None:
                radius = 0.5 + reso / 3.0
            scaler.scale(self.structure, radius=radius)

    def test_qfit_protein_simple_run(self):
        # Set up as if we have started in main()
        self.mock_main()

        # Only run qfit on two residues (reduce computational load)
        self.structure = self.structure.extract("resi", (99, 113), "==")
        self.structure = self.structure.reorder()
        assert len(list(self.structure.single_conformer_residues)) == 2

        # Run qfit object
        qfit = QFitProtein(self.structure, self.xmap, self.options)
        multiconformer = qfit._run_qfit_residue()
        mconformer_list = list(multiconformer.residues)
        print(mconformer_list)  # If we fail, this gets printed.
        assert len(mconformer_list) == 5  # Expect: 3*Ser99, 2*Phe113
