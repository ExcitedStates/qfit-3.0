#!/usr/bin/env python
"""
This script calculates the RMSD between two structures (PDB files) using their atomic coordinates.

Input:
    structure1: Path to the first PDB file (conformer 1).
    structure2: Path to the second PDB file (conformer 2).
    --pdb: Name of the input PDB, used to name the output CSV file.

Output:
    A CSV file with the calculated RMSD value between the two structures.

Example:
    python calc_rmsd.py conformer1.pdb conformer2.pdb --pdb 5C40
"""
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from qfit.structure import Structure, calc_rmsd


def parse_args():
    p = ArgumentParser(description=__doc__)
    p.add_argument("structure1", type=str, help="PDB-file containing structure.")
    p.add_argument("structure2", type=str, help="PDB-file containing structure.")
    p.add_argument("--pdb", type=str, help="Name of the input PDB.")
    args = p.parse_args()
    return args


def main():
    args = parse_args()
    structure1 = Structure.fromfile(args.structure1)  # conformer 1
    structure2 = Structure.fromfile(args.structure2)  # confromer 2
    structure1_coor_set = [structure1.coor]
    structure2_coor_set = [structure2.coor]

    rmsd = calc_rmsd(structure1_coor_set[0], structure2_coor_set[0])
    print(rmsd)

    # Create a DataFrame to store the result
    rmsd_data = pd.DataFrame({"pdb": [args.pdb], "rmsd": [rmsd]})

    # Write to CSV file, name the file as {pdb}_rmsd.csv
    output_file = f"{args.pdb}_rmsd.csv"
    rmsd_data.to_csv(output_file, index=False)
    print(f"RMSD data saved to {output_file}")


if __name__ == "__main__":
    main()
