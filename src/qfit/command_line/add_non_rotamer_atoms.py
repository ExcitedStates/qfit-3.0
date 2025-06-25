"""Automatically build a multiconformer residue"""

import numpy as np
import argparse
import logging
import os
import sys
import time
from string import ascii_uppercase
from . import Structure
from .structure.rotamers import ROTAMERS


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
    try:
        os.makedirs(args.directory)
    except OSError:
        pass

    structure = Structure.fromfile(args.structure).reorder()
    structure = structure.extract("record", "ATOM", "==")
    model = Structure.fromfile(args.model).reorder()
    for chain in structure:
        for residue in chain:
            if residue.resn[0] not in ROTAMERS:
                model = model.combine(residue)

    model.tofile(f"multiconformer_model_fixed.pdb")
