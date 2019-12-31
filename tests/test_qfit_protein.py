import pytest
import os

from qfit.qfit_protein import (
    QFitProteinOptions,
    build_argparser,
    prepare_qfit_protein,
    print_run_info,
)


class TestQFitProtein:
    def mock_main(self):
        # Prepare args
        args = [
            "../example/3K0N.mtz",  # mapfile, using relative directory from tests/
            "../example/3K0N.pdb",  # structurefile, using relative directory from tests/
            "-l", "2FOFCWT,PH2FOFCWT",
            "--directory", f"{os.environ['QFIT_OUTPUT_DIR']}",
        ]

        # Add options to reduce computational load
        args.extend([
            "--backbone-amplitude", "0.10",  # default: 0.30
            "--rotamer-neighborhood", "30",  # default: 60
        ])

        # Collect and act on arguments
        p = build_argparser()
        args = p.parse_args(args=args)
        try:
            os.mkdir(args.directory)
        except OSError:
            pass
        print_run_info(args)
        options = QFitProteinOptions()
        options.apply_command_args(args)

        # Build a QFitProtein job
        qfit = prepare_qfit_protein(options)

        # Reach into QFitProtein job,
        # simplify to only run on two residues (reduce computational load)
        qfit.structure = qfit.structure.extract("resi", (99, 113), "==")
        qfit.structure = qfit.structure.reorder()
        assert len(list(qfit.structure.single_conformer_residues)) == 2

        return qfit

    def test_run_qfit_residue(self):
        qfit = self.mock_main()

        # Run qfit object
        multiconformer = qfit._run_qfit_residue()
        mconformer_list = list(multiconformer.residues)
        print(mconformer_list)  # If we fail, this gets printed.
        assert len(mconformer_list) == 5  # Expect: 3*Ser99, 2*Phe113
