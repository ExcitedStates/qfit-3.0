"""Automatically build a multiconformer residue."""

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
    p.add_argument(
        "selection",
        type=str,
        help="Chain, residue id, and optionally insertion code for residue in structure, e.g. A,105, or A,105:A.",
    )

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
    chainid, resi = args.selection.split(",")
    if ":" in resi:
        resi, icode = resi.split(":")
        residue_id = (int(resi), icode)
    else:
        residue_id = int(resi)
        icode = ""
    structure_resi = structure.extract(f"resi {resi} and chain {chainid}")
    if icode:
        structure_resi = structure_resi.extract("icode", icode)
    for chain in structure:
        for residue in chain:
            if len(list(set(residue.altloc))) > 1:
                rtype = residue_type(residue)
                if rtype == "rotamer-residue":
                    for atom in residue.coor:
                        if min(np.linalg.norm(structure_resi.coor - atom, axis=1)) <= 5:
                            print(
                                f"{args.structure[0:4]}\t{chainid}\t{resi}\t{residue.resn[0]}"
                            )
                            break
