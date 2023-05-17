#!/usr/bin/env python

"""
This script is for an automated way to get the residue ID and chain ID of a ligand of interest to be fed into qFit Ligand

INPUT: PDB structure, ligand name
OUTPUT: A text file named Ligand_name_chain_resi.txt with the residue number and chain of the ligand

example:
get_lig_chain_res.py pdb.pdb lig_name 
"""

import argparse
import numpy as np
from qfit.structure import Structure


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("structure", type=str, help="PDB-file containing structure.")
    p.add_argument("lig_name", type=str, help="Ligand Name")

    args = p.parse_args()
    return args


def main():
    args = parse_args()
    structure = Structure.fromfile(args.structure)
    structure_resi = structure.extract("resn", args.lig_name)
    chain = np.unique(structure_resi.chain)
    resi = np.unique(structure_resi.resi)
    chain2 = " ".join(map(str, chain))
    resi2 = " ".join(map(str, resi))

    with open(args.lig_name + "_chain_resi.txt", "w") as file:
        file.write(chain2 + "," + resi2)


if __name__ == "__main__":
    main()
