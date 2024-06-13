"""Automatically build a multiconformer residue"""
import argparse
import os

from qfit import Structure
from qfit.structure.rotamers import ROTAMERS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("structure", type=str, help="PDB-file containing structure.")
    p.add_argument("model", type=str, help="PDB-file containing the model to be fixed.")

    # Output options
    p.add_argument(
        "-d",
        "--directory",
        type=os.path.abspath,
        default=".",
        metavar="<dir>",
        help="Directory to store results.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Be verbose.")
    args = p.parse_args()

    return args


def main():
    args = parse_args()
    os.makedirs(args.directory, exist_ok=True)

    structure = Structure.fromfile(args.structure).reorder()
    structure = structure.extract("record", "ATOM", "==")
    model = Structure.fromfile(args.model).reorder()
    for chain in structure:
        for residue in chain:
            if residue.resn[0] not in ROTAMERS:
                model = model.combine(residue)

    model.tofile(f"multiconformer_model_fixed.pdb")
