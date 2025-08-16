#!/usr/bin/env python
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from qfit.scaler import MapScaler
from qfit.structure import Structure
from qfit.volume import XMap
from qfit.validator import Validator
import os

"""
This script is based on calc_rscc.py and is designed to compare the RSCC (Real-Space Correlation Coefficient) of two matched structural models derived from the same map. 
It calculates the RSCC for any residue defined by its residue number and chain ID (--residue). The script 
calculates the RSCC of each modeled residue within the same voxel space against the same map.


To run: 
 compare_rscc_voxel PDB_FILE_deposited.pdb MTZ_FILE_deposited.mtz --comp_pdb PDB_FILE_QFIT.pdb --pdb PDB_NAME
 If you want to run on a single residue use: --residue A,1208; else it will run on every non-water residue in the PDB. 
 optional: --directory /path/for/output/csv/file

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
    gen_structure = Structure.fromfile(options.comp_pdb)

    # Remove water molecules from both structures
    dep_structure = dep_structure.extract("resn", "HOH", "!=")
    gen_structure = gen_structure.extract("resn", "HOH", "!=")

    # Remove anisotropic data
    for attr in ["u00", "u11", "u22", "u01", "u02", "u12"]:
        if attr in dep_structure.data:
            del dep_structure.data[attr]
        if attr in gen_structure.data:
            del gen_structure.data[attr]

    # Initialize data storage for RSCC values
    rscc_data = {
        "Chain": [],
        "Residue": [],
        "Residue_Name": [],
        "Base_RSCC": [],
        "Comparison_RSCC": []
    }

    # Load and process the electron density maps:
    dep_xmap = XMap.fromfile(options.map, label=options.label)
    dep_scaler = MapScaler(dep_xmap)
    dep_xmap = dep_xmap.canonical_unit_cell()

    # If a specific residue is provided, calculate RSCC for that residue
    if options.residue is not None:
        chainid, resi = options.residue.split(",")
        gen_ligand = gen_structure.extract(f"resi {resi} and chain {chainid}")
        dep_ligand = dep_structure.extract(f"resi {resi} and chain {chainid}")

        # Add the deposited ligand to the combined structure
        combined_structure = gen_ligand.combine(dep_ligand)

        # Set footprint to the combined structure
        footprint = combined_structure

        # Scale and extract map
        dep_scaler.scale(footprint, radius=1.5)
        dep_xmap = dep_xmap.extract(combined_structure.coor, padding=8)

        # Validate and calculate RSCC
        dep_validator = Validator(dep_xmap, dep_xmap.resolution, options.directory)
        dep_rscc = dep_validator.rscc(dep_ligand)
        gen_rscc = dep_validator.rscc(gen_ligand)

        # Store RSCC values
        rscc_data["Chain"].append(chainid)
        rscc_data["Residue"].append(resi)
        rscc_data["Residue_Name"].append(dep_ligand.resn[0] if dep_ligand.size > 0 else None)
        rscc_data["Base_RSCC"].append(dep_rscc)
        rscc_data["Comparison_RSCC"].append(gen_rscc)

    else:
        # If no specific residue is provided, calculate RSCC for all non-water residues
        combined_structure = gen_structure.combine(dep_structure)
        footprint = combined_structure

        # Scale and extract map
        dep_scaler.scale(footprint, radius=1.5)
        dep_xmap = dep_xmap.extract(combined_structure.coor, padding=8)

        # Validate and calculate RSCC for each residue
        dep_validator = Validator(dep_xmap, dep_xmap.resolution, options.directory)
        for chain in np.unique(dep_structure.chain):
            for residue in np.unique(dep_structure.extract("chain", chain, '==').resi):
                dep_ligand = dep_structure.extract(f"resi {residue} and chain {chain}")
                gen_ligand = gen_structure.extract(f"resi {residue} and chain {chain}")

                dep_rscc = dep_validator.rscc(dep_ligand)
                gen_rscc = dep_validator.rscc(gen_ligand)

                # Store RSCC values
                rscc_data["Chain"].append(chain)
                rscc_data["Residue"].append(residue)
                rscc_data["Residue_Name"].append(dep_ligand.resn[0] if dep_ligand.size > 0 else None)
                rscc_data["Base_RSCC"].append(dep_rscc)
                rscc_data["Comparison_RSCC"].append(gen_rscc)

    # Write to CSV
    df = pd.DataFrame(rscc_data)
    csv_filename = f"{options.directory}{options.pdb}_rscc.csv"
    df.to_csv(csv_filename, index=False)

if __name__ == "__main__":
    main()
