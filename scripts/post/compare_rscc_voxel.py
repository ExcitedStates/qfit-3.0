#!/usr/bin/env python
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import csv
from qfit.scaler import MapScaler
from qfit.structure import Structure
from qfit.volume import XMap
from qfit.validator import Validator
import os

"""
This script is based on calc_rscc.py and is designed to compare the RSCC (Real-Space Correlation Coefficient) of a two matched structural models derived from the same map. 
It calculates the RSCC for any residue defined by its by its residue number and chain ID (--residue). The script 
calculates the RSCC of each modeled residue within the same voxel space againsts the same map.


To run: 
 compare_rscc_voxel PDB_FILE_deposited.pdb MTZ_FILE_deposited.mtz --comp_pdb PDB_FILE_QFIT.pdb --residue A,401 
 optional: --pdb PDB_NAME --directory /path/for/output/csv/file

"""


def build_argparser():
    p = ArgumentParser(description=__doc__)
    p.add_argument(
        "base_structure", type=str, help="Base PDB-file containing structure."
    )
    p.add_argument("map", type=str, help="X-ray density map in CCP4 or MRC or MTZ file")
    p.add_argument("--comp_pdb", type=str, help="qFit generated pdb file")
    p.add_argument(
        "--residue", type=str, help="Chain_ID, Residue_ID for RSCC to be calculated on"
    )
    p.add_argument(
        "-l",
        "--label",
        default="2FOFCWT,PH2FOFCWT",
        metavar="<F,PHI>",
        help="MTZ column labels to build density. Required if MTZ format",
    )
    p.add_argument("--pdb", type=str, help="name of PDB")
    p.add_argument("--directory", type=str, default="", help="Where to save RSCC info")
    return p


def main():
    p = build_argparser()
    options = p.parse_args()
    # Load structure and prepare it
    dep_structure = Structure.fromfile(options.base_structure)
    gen_strucutre = Structure.fromfile(options.comp_pdb)

    # Get the generated and deposited ligand coordinates
    if options.residue is not None:
        chainid, resi = options.residue.split(",")
        gen_ligand = gen_strucutre.extract(f"resi {resi} and chain {chainid}")
        dep_ligand = dep_structure.extract(f"resi {resi} and chain {chainid}")
    else:
        print("Please provide residue name or residue ID and chain ID")

    # Add the deposited ligand to the combined structure
    combined_structure = gen_ligand.combine(dep_ligand)

    # Load and process the electron density maps:
    dep_xmap = XMap.fromfile(options.map, label=options.label)
    dep_scaler = MapScaler(dep_xmap)
    dep_xmap = dep_xmap.canonical_unit_cell()
    footprint = gen_ligand  # set voxel space around the generated ligand
    dep_scaler.scale(footprint, radius=1.5)

    dep_xmap = dep_xmap.extract(
        gen_ligand.coor, padding=8
    )  # Create a copy of the deposited map around the atomic coordinates provided.

    # Now that the conformers have been generated, the resulting
    # # conformations should be examined via GoodnessOfFit:
    dep_validator = Validator(dep_xmap, dep_xmap.resolution, options.directory)
    dep_rscc = dep_validator.rscc(dep_ligand)
    print(f'Base Structure RSCC: {dep_rscc}')

    gen_rscc = dep_validator.rscc(gen_ligand)
    print(f'Comparison RSCC: {gen_rscc}')

    csv_filename = f"{options.pdb}_rscc.csv"

    # Write to CSV
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(["PDB", "Base_RSCC", "Comparison_RSCC"])
        # Write the data
        writer.writerow([options.pdb, dep_rscc, gen_rscc])


if __name__ == "__main__":
    main()
