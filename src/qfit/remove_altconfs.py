"""Automatically build a multiconformer residue"""

import numpy as np
import argparse
import logging
import os
import sys
import time
from string import ascii_uppercase
from . import Structure
from .structure import residue_type


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("structure", type=str, help="PDB-file containing structure.")

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
    for chain in structure:
        for residue in chain:
            altlocs = sorted(list(set(residue.altloc)))
            resi = residue.resi[0]
            chainid = residue.chain[0]
            if len(altlocs) > 1:
                try:
                    altlocs.remove("")
                except ValueError:
                    pass
                for altloc in altlocs[1:]:
                    sel_str = f"resi {resi} and chain {chainid} and altloc {altloc}"
                    sel_str = f"not ({sel_str})"
                    structure = structure.extract(sel_str)
    structure.q = 1.0
    structure.altloc = ""
    structure.tofile(f"{args.structure[:-4]}.single.pdb")
