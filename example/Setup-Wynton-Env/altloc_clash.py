import sys
import os
from Bio.PDB import PDBParser, NeighborSearch, is_aa

##########################################################################################################################
# The purpose of this file it to identify strucutures where protein and ligand altlocs of opposite labels physically clash 
##########################################################################################################################

#### INPUTS #### 
pdb_id = sys.argv[1]
chain_id = sys.argv[2]
res_num = int(sys.argv[3])
lig_name = sys.argv[4]
base_dir = sys.argv[5]

# Load PDB
pdb_path = os.path.join(base_dir, pdb_id, f"{pdb_id}.pdb")
if not os.path.isfile(pdb_path):
    print(f"Error: {pdb_path} not found.")
    sys.exit(1)

parser = PDBParser(QUIET=True)
structure = parser.get_structure(pdb_id, pdb_path)

# Separate ligand and protein atoms by altloc
ligand_atoms_A = []
ligand_atoms_B = []
protein_atoms_A = []
protein_atoms_B = []

for model in structure:
    for chain in model:
        for residue in chain:
            #### Ligand atoms ####
            if chain.id == chain_id and residue.id[1] == res_num and residue.get_resname().strip() == lig_name:
                for atom in residue:
                    if atom.is_disordered():
                        for alt_atom in atom.child_dict.values():
                            altloc = alt_atom.get_altloc()
                            if altloc == 'A':
                                ligand_atoms_A.append(alt_atom)
                            elif altloc == 'B':
                                ligand_atoms_B.append(alt_atom)
                    else:
                        altloc = atom.get_altloc()
                        if altloc == 'A':
                            ligand_atoms_A.append(atom)
                        elif altloc == 'B':
                            ligand_atoms_B.append(atom)

            #### Protein atoms ####
            elif residue.id[0] == ' ' and is_aa(residue):   
                for atom in residue:
                    if atom.is_disordered():
                        for alt_atom in atom.child_dict.values():
                            altloc = alt_atom.get_altloc()
                            if altloc == 'A':
                                protein_atoms_A.append(alt_atom)
                            elif altloc == 'B':
                                protein_atoms_B.append(alt_atom)
                    else:
                        altloc = atom.get_altloc()
                        if altloc == 'A':
                            protein_atoms_A.append(atom)
                        elif altloc == 'B':
                            protein_atoms_B.append(atom)

if not ligand_atoms_A and not ligand_atoms_B:
    print(f"Warning: No ligand altlocs found for {pdb_id}")
    sys.exit(0)



# Build neighbor search trees
ns_protein_A = NeighborSearch(protein_atoms_A)
ns_protein_B = NeighborSearch(protein_atoms_B)

clash_found = False

# Ligand A --> clash --> Protein B
for lig_atom in ligand_atoms_A:
    close_atoms = ns_protein_B.search(lig_atom.coord, 1.0) # clash = within 1 Angstrom 
    if close_atoms:
        clash_found = True
        break

# Ligand B --> clash --> Protein A
for lig_atom in ligand_atoms_B:
    close_atoms = ns_protein_A.search(lig_atom.coord, 1.0) # clash = within 1 Angstrom 
    if close_atoms:
        clash_found = True
        break

if clash_found:
    # Output the PDB ID to stdout
    print(pdb_id)
