#!/usr/bin/env python

"""
This script processes a multiconformer protein-ligand or ligand-only PDB file and generates individual PDB files for each ligand conformer present.

INPUT: 
- PDB structure file 
- Ligand chain and residue number 
- Output file name 

OUTPUT: 
- Separate PDB files for each ligand conformer in the provided structure.

EXAMPLE USAGE:
split_multiconformer_ligand.py 5C40.pdb --residue A,401 --output_name deposited

Outputs: deposited_ligand_A.pdb and deposited_ligand_B.pdb

"""

import numpy as np
import pandas as pd
import argparse
import os
from qfit.structure import Structure
from qfit.structure.ligand import _Ligand


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("structure", type=str, help="PDB-file containing structure.")
    p.add_argument("--residue", type=str, help="Chain_ID, Residue_ID for RSCC to be calculated on")
    p.add_argument(
        "-d",
        "--directory",
        default=".",
        metavar="<dir>",
        type=os.path.abspath,
        help="Directory to store results",
    )
    p.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="This will affect the output file naming."
    )

    args = p.parse_args()
    return args


args = parse_args()

# Load structure and prepare it
structure = Structure.fromfile(args.structure)
chainid, resi = args.residue.split(",")

# Extract the ligand and altlocs
structure_ligand = structure.extract(f"resi {resi} and chain {chainid}")
altlocs = sorted(set(structure_ligand.altloc) - {''})

# Handle the case where there are no alternate locations
if not altlocs:
    altlocs = ['']  # This will handle the default case

# Extract the common part of the ligand (without alternate locations)
common_structure = structure_ligand.extract("altloc ''")

# Loop over each altloc
for altloc in altlocs:
    # Extract the structure for the current altloc
    if altloc:
        # structure_altloc = structure_ligand.extract(f"altloc {altloc}")
        alt_structure = structure_ligand.extract(f"altloc {altloc}")

        occupancies = alt_structure.q
        print(occupancies)
        # Combine with common structure
        structure_altloc = common_structure.combine(alt_structure)

        for atom, occupancy in zip(structure_altloc, occupancies):
            atom.q = occupancy


    else:
        # structure_altloc = structure_ligand
        structure_altloc = common_structure

    # Prepare the ligand object
    ligand = _Ligand(
        structure_altloc.data,
        structure_altloc._selection,
        link_data=structure_altloc.link_data,
    )
    ligand.altloc = ""
    #ligand.q = 1

    # Create a file name for the current altloc
    exte = ".pdb"
    output_name_prefix = args.output_name  # Set the prefix to 'qfit' or 'depo' based on output_name

    if altloc:
        ligand_name = os.path.join(args.directory, f"{output_name_prefix}_ligand_{altloc}{exte}")
    else:
        ligand_name = os.path.join(args.directory, f"{output_name_prefix}_ligand_A{exte}")

    # Save the file
    print(f"saving: {ligand}")
    ligand.tofile(ligand_name)
