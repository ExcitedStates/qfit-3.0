#!/usr/bin/env python
"""
The purpose of this script is to calculate the B-factor for every residue across the structure. You can choose to calculate the alpha carbon or side chain (all heavy atoms)
This script also returns the average B-factor for alpha carbon which is used in the calc_OP.py script.

INPUT: PDB structure, name of PDB structure, sidechain or CA
OUTPUT: CSV file the b-factor for each residue in the structure, average b-factor for entire structure

example:
Alpha Carbon B-factor:
b_factor.py pdb.pdb --pdb pdb_name --ca

Heavy Atom B-factor:
b_factor.py pdb.pdb --pdb pdb_name --sidechain
"""

import numpy as np
import pandas as pd
from argparse import ArgumentParser
from qfit.structure import Structure


def build_argparser():
    p = ArgumentParser(description=__doc__)
    p.add_argument("structure", type=str, help="PDB-file containing structure.")
    p.add_argument(
        "--ca",
        dest="ca",
        action="store_true",
        help="Return B factor for alpha carbons only",
    )
    p.add_argument(
        "--sidechain",
        dest="sc",
        action="store_true",
        help="Return B factor for sidechain atoms only",
    )
    p.add_argument("--pdb", help="Name of the input PDB.")
    return p


def get_bfactor(structure, pdb, ca, sidechain):
    if ca and sidechain:
        print("Both alpha carbon and sidechain selected. Please choose one")
        return
    bfactors = []
    select = structure.extract("record", "ATOM", "==")
    select = select.extract("e", "H", "!=")
    if ca:
        select = select.extract("name", "CA", "==")
    if sidechain:
        select = select.extract("name", (["N", "CA", "C", "O"]), "!=")
    for c in np.unique(select.chain):
        for r in np.unique(select.extract("chain", c, "==").resi):
            b_factor = np.average(
                select.extract(f"resi {r} and chain {c}").b
            ) / np.average(select.extract(f"resi {r} and chain {c}").q)
            bfactors.append(
                tuple(
                    (
                        pdb,
                        np.array2string(r),
                        np.array2string(
                            select.extract(f"resi {r} and chain {c}").resn[0]
                        ),
                        np.array2string(np.unique(c)),
                        b_factor,
                    )
                )
            )
    B_factor = pd.DataFrame(
        bfactors, columns=["PDB", "resi", "resn", "chain", "b_factor"]
    )
    med = B_factor["b_factor"].median()  # median b-factor
    return B_factor, med


def main():
    p = build_argparser()
    args = p.parse_args()
    structure = Structure.fromfile(args.structure).reorder()
    B_factor, median = get_bfactor(structure, args.pdb, args.ca, args.sc)
    B_factor.to_csv(args.pdb + "_B_factors.csv", index=False)
    print(median)  # for analysis scripts to give to OP script
    return median


if __name__ == "__main__":
    main()
