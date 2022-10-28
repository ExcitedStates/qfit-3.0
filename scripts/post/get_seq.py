#!/usr/bin/env python
"""
This script will take in a PDB and return a single letter code for every rotamer residue in the PDB. 
"""

import sys
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from qfit.structure import Structure


def build_argparser():
    p = ArgumentParser(description=__doc__)
    p.add_argument("structure", type=str, help="PDB-file containing structure.")
    p.add_argument("--pdb", help="Name of the input PDB.")
    return p


d = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
}


def get_seq(structure, pdb):
    struct = Structure.fromfile(structure).reorder()
    select = struct.extract("record", "ATOM", "==")
    select = select.extract("e", "H", "!=")
    seq = []
    for c in select.chain[0]:
        for r in np.unique(select.extract("chain", c, "==").resi):
            seq.append(d[select.extract(f"chain {c} and resi {r}").resn[0]])
    seq = "".join(seq)
    print(seq)
    with open(pdb + "_seq.txt", "w") as file:
        file.write(str(seq) + "\n")
    return seq


def main():
    p = build_argparser()
    args = p.parse_args()
    get_seq(args.structure, args.pdb)


if __name__ == "__main__":
    main()
