#!/usr/bin/env python
"""
This script scales the occupancies of a specified ligand in a protein-ligand PDB file based on a provided BDC value. The script scales the occupancies by (1 - BDC)

Input:
    PDB1: Protein-ligand PDB file
    PDB2: Ligand only PDB file, contains the ligand you wish to scale.
    BDC_value: The BDC value used for scaling occupancies.

Output:
    A new protein-ligand PDB file with scaled occupancies for the ligand. The new PDB file will be named 
    as `PDB1_scaled.pdb`.

Example:
    python event_map_bdc_scaler.py protein_ligand.pdb ligand_only.pdb 0.80
"""

import sys

def extract_ligand_names(ligand_pdb_file):
    """
    Extract unique ligand residue names (resname) from the given PDB file.
    """
    ligand_names = set()
    
    with open(ligand_pdb_file, "r") as ligfile:
        for line in ligfile:
            if line.startswith("HETATM"):
                ligand_name = line[17:20].strip()  # Extract residue name (columns 18-20)
                ligand_names.add(ligand_name)
    
    return ligand_names

def scale_occupancies(pdb_file, ligand_names, BDC):
    """
    Scales ligand occupancies by 1-BDC in a multiconformer PDB file, 
    but only for HETATMs corresponding to the ligands in ligand_names.
    """
    scaled_pdb = pdb_file.replace(".pdb", "_scaled.pdb")
    BDC_scale = 1 - float(BDC)
    print(f"1 - BDC = {BDC_scale}")
    
    with open(pdb_file, "r") as infile, open(scaled_pdb, "w") as outfile:
        for line in infile:
            if line.startswith("HETATM"):
                occupancy = float(line[54:60].strip())
                ligand_name = line[17:20].strip()  # Extract the ligand name (resname)
                
                # Only scale the occupancy if the ligand is in the ligand_names set
                if ligand_name in ligand_names:
                    scaled_occupancy = occupancy * BDC_scale
                    # Write the line back with the new scaled occupancy
                    outfile.write(line[:54] + f"{scaled_occupancy:6.2f}" + line[60:])
                else:
                    # Write unchanged lines
                    outfile.write(line)
            else:
                # Write non-HETATM lines unchanged
                outfile.write(line)
    
    print(f"Scaled PDB written to {scaled_pdb}")
    return scaled_pdb

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python event_map_bdc_scaler.py <PDB.pdb> <multiconf_lig.pdb> <BDC_value>")
        sys.exit(1)
    
    pdb_file = sys.argv[1]        # Main PDB file to scale
    ligand_pdb_file = sys.argv[2] # PDB file containing the ligand
    BDC_value = sys.argv[3]       # BDC value

    # Extract ligand names from the multiconf_lig file
    ligand_names = extract_ligand_names(ligand_pdb_file)
    print(f"Ligand names found: {ligand_names}")
    
    # Scale the occupancies in pdb_file, only for the specified ligands
    scaled_pdb = scale_occupancies(pdb_file, ligand_names, BDC_value)
    
    sys.exit(0)
