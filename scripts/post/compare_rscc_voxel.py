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
This script is based on calc_rscc.py and is designed to compare the RSCC (Real-Space Correlation Coefficient) of a qFit-generated ligand to a deposited ligand.
It calculates the RSCC for a ligand (or any residue) defined by its ligand name (--ligand) or by its residue number and chain ID (--residue). The script takes two ligands, 
with their corresponding electron density maps, as input and calculates the RSCC of each ligand within the same voxel space, ensuring a proper comparison between the deposited and qFit-generated models.

Note: This script only works on MTZ maps containing 2FOFCWT,PH2FOFCWT map labels.


To run: 
 compare_rscc_voxel PDB_FILE_deposited.pdb MTZ_FILE_deposited.mtz --gen_pdb PDB_FILE_QFIT.pdb --gen_map MTZ_FILE_QFIT.mtz --residue A,401 --pdb PDB_NAME --directory /path/for/output/csv/file

"""


def build_argparser():
    p = ArgumentParser(description=__doc__)
    p.add_argument(
        "structure", type=str, help="deposited PDB-file containing structure."
    )
    p.add_argument("map", type=str, help="deposited map.")
    p.add_argument("--gen_pdb", type=str, help="qFit generated pdb file")
    p.add_argument("--gen_map", type=str, help="qFit generated map file")
    p.add_argument(
        "--ligand", type=str, help="name of ligand for RSCC to be calculated on"
    )
    p.add_argument(
        "--residue", type=str, help="Chain_ID, Residue_ID for RSCC to be calculated on"
    )
    p.add_argument("--pdb", type=str, help="name of PDB")
    p.add_argument("--directory", type=str, help="Where to save RSCC info")
    return p


def main():
    p = build_argparser()
    options = p.parse_args()
    # Load structure and prepare it
    dep_structure = Structure.fromfile(options.structure)
    gen_strucutre = Structure.fromfile(options.gen_pdb)

    # Get the generated and deposited ligand coordinates
    if options.ligand is not None:
        gen_ligand = gen_strucutre.extract("resn", options.ligand, "==")
        dep_ligand = dep_structure.extract("resn", options.ligand, "==")
    elif options.residue is not None:
        chainid, resi = options.residue.split(",")
        gen_ligand = gen_strucutre.extract(f"resi {resi} and chain {chainid}")
        dep_ligand = dep_structure.extract(f"resi {resi} and chain {chainid}")
    else:
        print("Please provide ligand name or residue ID and chain ID")

    # Load and process the electron density maps:
    dep_xmap = XMap.fromfile(options.map, label="2FOFCWT,PH2FOFCWT")
    dep_scaler = MapScaler(dep_xmap)
    dep_xmap = dep_xmap.canonical_unit_cell()
    footprint = gen_ligand  # set voxel space around the generated ligand
    dep_scaler.scale(footprint, radius=1.5)

    dep_xmap = dep_xmap.extract(
        gen_ligand.coor, padding=8
    )  # Create a copy of the deposited map around the atomic coordinates provided.

    gen_xmap = XMap.fromfile(options.gen_map, label="2FOFCWT,PH2FOFCWT")
    gen_scaler = MapScaler(gen_xmap)
    gen_xmap = gen_xmap.canonical_unit_cell()
    footprint = gen_ligand  # set voxel space around the generated ligand
    gen_scaler.scale(footprint, radius=1.5)

    gen_xmap = gen_xmap.extract(
        gen_ligand.coor, padding=8
    )  # Create a copy of the map around the atomic coordinates provided.

    ext = ".ccp4"
    scaled_fname = os.path.join(
        options.directory, f"dep{ext}"
    )  # this should be an option

    dep_xmap.tofile(scaled_fname)

    gen_fname = os.path.join(options.directory, f"gen{ext}")  # this should be an option

    gen_xmap.tofile(gen_fname)

    # Now that the conformers have been generated, the resulting
    # # conformations should be examined via GoodnessOfFit:
    dep_validator = Validator(dep_xmap, dep_xmap.resolution, options.directory)
    dep_rscc = dep_validator.rscc(dep_ligand)
    print(dep_rscc)

    gen_validator = Validator(gen_xmap, gen_xmap.resolution, options.directory)
    gen_rscc = gen_validator.rscc(gen_ligand)
    print(gen_rscc)

    csv_filename = f"{options.pdb}_rscc.csv"

    # Write to CSV
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(["PDB", "dep_RSCC", "gen_RSCC"])
        # Write the data
        writer.writerow([options.pdb, dep_rscc, gen_rscc])


if __name__ == "__main__":
    main()
