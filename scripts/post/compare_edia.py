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
This script is designed to compare the EDIA (Electron Density Fit) of two matched structural models derived from the same map. 
It calculates the EDIA for any residue defined by its residue number and chain ID (--residue). The script 
calculates the EDIA of each modeled residue within the same voxel space against the same map.

To run: 
 compare_edia_voxel PDB_FILE_deposited.pdb MTZ_FILE_deposited.mtz --comp_pdb PDB_FILE_QFIT.pdb --residue A,401 
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
        "--residue", type=str, help="Chain_ID, Residue_ID for EDIA to be calculated on"
    )
    p.add_argument(
        "-l",
        "--label",
        default="2FOFCWT,PH2FOFCWT",
        metavar="<F,PHI>",
        help="MTZ column labels to build density. Required if MTZ format",
    )
    p.add_argument("--pdb", type=str, help="name of PDB")
    p.add_argument("--directory", type=str, default="", help="Where to save EDIA info")
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

    # Initialize data storage for EDIA values
    edia_data = {
        "Chain": [],
        "Residue": [],
        "Base_EDIA": [],
        "Comparison_EDIA": []
    }

    # Load and process the electron density maps:
    dep_xmap = XMap.fromfile(options.map, label=options.label)
    dep_scaler = MapScaler(dep_xmap)
    dep_xmap = dep_xmap.canonical_unit_cell()

    # If a specific residue is provided, calculate EDIA for that residue
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

        # Validate and calculate EDIA
        dep_validator = Validator(dep_xmap, dep_xmap.resolution, options.directory)
        dep_edia = dep_validator.edia_like_for_atom(dep_ligand)
        gen_edia = dep_validator.edia_like_for_atom(gen_ligand)

        # Store EDIA values
        edia_data["Chain"].append(chainid)
        edia_data["Residue"].append(resi)
        edia_data["Base_EDIA"].append(dep_edia["edia_like"])
        edia_data["Comparison_EDIA"].append(gen_edia["edia_like"])

    else:
        # If no specific residue is provided, calculate EDIA for all non-water residues
        combined_structure = gen_structure.combine(dep_structure)
        footprint = combined_structure

        # Scale and extract map
        dep_scaler.scale(footprint, radius=1.5)
        dep_xmap = dep_xmap.extract(combined_structure.coor, padding=8)

        # Validate and calculate EDIA for each residue
        dep_validator = Validator(dep_xmap, dep_xmap.resolution, options.directory)
        for chain in np.unique(dep_structure.chain):
            for residue in np.unique(dep_structure.extract("chain", chain, '==').resi):
                dep_ligand = dep_structure.extract(f"resi {residue} and chain {chain}")
                gen_ligand = gen_structure.extract(f"resi {residue} and chain {chain}")

                dep_edia = dep_validator.edia_like_for_atom(dep_ligand)
                gen_edia = dep_validator.edia_like_for_atom(gen_ligand)

                # Store EDIA values
                edia_data["Chain"].append(chain)
                edia_data["Residue"].append(residue)
                edia_data["Base_EDIA"].append(dep_edia["edia_like"])
                edia_data["Comparison_EDIA"].append(gen_edia["edia_like"])

    # Write to CSV
    df = pd.DataFrame(edia_data)
    csv_filename = f"{options.pdb}_edia.csv"
    df.to_csv(csv_filename, index=False)

if __name__ == "__main__":
    main()
