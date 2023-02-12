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
    p.add_argument("restraint", type=str, help="EFF file containing the restraints.")

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
    p.add_argument(
        "-occ",
        "--occ_cutoff",
        type=float,
        default=0.01,
        metavar="<float>",
        help="Remove conformers with occupancies below occ_cutoff. Default = 0.01",
    )
    args = p.parse_args()

    return args


def main():
    args = parse_args()
    os.makedirs(args.directory, exist_ok=True)

    structure = Structure.fromfile(args.structure)

    with open(args.restraint, "r") as restraint_file:
        for line in restraint_file.readlines():
            if "{" in line or "}" in line:
                print(line.strip())
            else:
                fields = line.split()
                if "atom_selection_1" in fields:
                    chainid3 = fields[3]
                    resi3 = fields[6]
                    name3 = fields[9]
                elif "atom_selection_2" in fields:
                    chainid4 = fields[3]
                    resi4 = fields[6]
                    name4 = fields[9]
                elif "distance_ideal" in fields:
                    dist = fields[2]
                elif "sigma" in fields:
                    sigma = fields[2]

                    structure_res1 = structure.extract(
                        f"resi {resi3} and chain {chainid3}"
                    )
                    structure_res2 = structure.extract(
                        f"resi {resi4} and chain {chainid4}"
                    )

                    altlocs1 = list(set(structure_res1.altloc))
                    altlocs2 = list(set(structure_res2.altloc))
                    altlocs1.sort()
                    altlocs2.sort()
                    # Decide if this is a case we care about:
                    if len(altlocs1) > 1 or len(altlocs2) > 1:
                        first = True
                        for alt1, alt2 in zip(altlocs1, altlocs2):
                            if not first:
                                print("   }\n   bond {")
                            print(
                                f"     atom_selection_1 = chain {chainid3} and resseq {resi3} and name {name3} and altloc {alt1}"
                            )
                            print(
                                f"     atom_selection_2 = chain {chainid4} and resseq {resi4} and name {name4} and altloc {alt2}"
                            )
                            print(f"     distance_ideal = {dist}")
                            print(f"     sigma = {sigma}")
                            first = False
                    else:
                        print(
                            f"     atom_selection_1 = chain {chainid3} and resseq {resi3} and name {name3}"
                        )
                        print(
                            f"     atom_selection_2 = chain {chainid4} and resseq {resi4} and name {name4}"
                        )
                        print(f"     distance_ideal = {dist}")
                        print(f"     sigma = {sigma}")
