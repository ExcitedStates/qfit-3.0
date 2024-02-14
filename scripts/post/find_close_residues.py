#!/usr/bin/env python

"""
10/31/2022: Not frequently used anymore
The purpose of this script is to identify residues with more than one conformer within 5 angstroms any (non-crystallographic) ligands in the PDB


INPUT: PDB structure, name of PDB structure
OUTPUT: CSV file with the residues within 5A of ligand with the number of single and multi conf structures

example:
find_altlocs_near_ligand.py pdb.pdb pdb_name 
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from qfit.structure import Structure
from qfit.structure.rotamers import ROTAMERS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("structure", type=str, help="PDB-file containing structure.")
    p.add_argument("pdb_name", type=str, help="name of PDB.")
    p.add_argument(
        "-dir", type=os.path.abspath, default=".", help="directory of output."
    )
    p.add_argument("ligand", type=str, help="name of ligand")
    p.add_argument("dist", type=float, help="distance")
    args = p.parse_args()
    return args


def main():
    args = parse_args()

    if not args.pdb_name is None:
        pdb_name = args.pdb_name
    else:
        pdb_name = ""

    structure = Structure.fromfile(args.structure)

    ligands = structure.extract("resn", args.ligand, "==")
    receptor = structure.extract("record", "ATOM")
    receptor = receptor.extract("resn", "HOH", "!=")
    close_res = pd.DataFrame()
    alt_loc = pd.DataFrame()

    ligand = structure.extract("resn", args.ligand)
    mask = ligand.e != "H"
    neighbors = {}
    for coor in ligand.coor:
        dist = np.linalg.norm(receptor.coor - coor, axis=1)
        for near_residue, near_chain in zip(
            receptor.resi[dist < args.dist], receptor.chain[dist < args.dist]
        ):
            key = str(near_residue) + " " + near_chain
            if key not in neighbors:
                neighbors[key] = 0
            n = 1
    for key in neighbors.keys():
        residue_id, chain = key.split()
        close_res.loc[n, "res_id"] = residue_id
        close_res.loc[n, "chain"] = chain
        n += 1
    close_res.to_csv(
        pdb_name + "_" + args.ligand + "_" + str(args.dist) + "_closeres.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
