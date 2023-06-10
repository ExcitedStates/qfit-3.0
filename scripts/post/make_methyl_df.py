#!/usr/bin/env python

"""
This script is to produce the intermediate file to calculate crystallographic order parameters. 
INPUT: PDB structure, pdb name
OUTPUT: A tab seperated file with information about each residue and the atom type need to calculate cyrstallographic order parameters.

example:
make_methyl_df.py {pdb}.pdb --pdb {pdb_name} 
"""

import numpy as np
import pandas as pd
import argparse
from qfit.structure import Structure


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("structure", type=str, help="PDB-file containing structure.")
    p.add_argument("--pdb", help="Name of the input PDB.")

    args = p.parse_args()
    return args


args = parse_args()

methyl = []
structure = Structure.fromfile(args.structure).reorder()
structure = structure.extract("record", "HETATM", "!=")
for chain in np.unique(structure.chain):
    for resi in np.unique(structure.extract("chain", chain, "==").resi):
        resname = structure.extract(f"chain {chain} and resi {resi}").resn[0]
        a1 = "HB2"
        a2 = "CB"
        if resname in ["THR", "ILE", "VAL"]:
            a1 = "HB"
            a2 = "CB"
        if resname == "GLY":
            a1 = "HA2"
            a2 = "CA"
        methyl.append(tuple((resi, a1, resi, a2, 1.0000, 0.0000, chain, resname)))

methyl_df = pd.DataFrame(
    methyl, columns=["resi", "a1", "resi", "a2", "hold1", "hold2", "chain", "resn"]
)
methyl_df.to_csv(args.pdb + "_qFit_methyl.dat", sep=" ", index=False)
