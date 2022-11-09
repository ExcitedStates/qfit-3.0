# qFit 3.2.2

![](https://github.com/ExcitedStates/qfit-3.0/workflows/tests/badge.svg)

If you use this software, please cite:
- [Riley BT, Wankowicz SA, et al. qFit 3: Protein and ligand multiconformer modeling for X-ray crystallographic and single-particle cryo-EM density maps. Protein Sci. 30, 270–285 (2021)](https://dx.doi.org/10.1002/pro.4001)
- [van Zundert, G. C. P. et al. qFit-ligand Reveals Widespread Conformational Heterogeneity of Drug-Like Molecules in X-Ray Electron Density Maps. J. Med. Chem. 61, 11183–11198 (2018)](https://dx.doi.org/10.1021/acs.jmedchem.8b01292)
- [Keedy, D. A., Fraser, J. S. & van den Bedem, H. Exposing Hidden Alternative Backbone Conformations in X-ray Crystallography Using qFit. PLoS Comput. Biol. 11, e1004507 (2015)](https://dx.doi.org/10.1371/journal.pcbi.1004507)

## Refinement
After *multiconformer_model2.pdb* has been generated, the model must need to be refined. Bear in mind that this final step currently depends on an existing installation of the Phenix software suite. This script is currently written to work with version Phenix 1.20.

[Phenix installation](https://phenix-online.org/documentation/install-setup-run.html)

X-ray crystallography:
`qfit_final_refine_xray.sh /path/to/mtz_file.mtz multiconformer_model2.pdb`

Cryo-EM: 
`qfit_final_refine_cryoem.sh /path/to/ccp4_file.ccp4 original_pdb.pdb multiconformer_model2.pdb`


## Analysis Scripts

### 1. Calculating Order Parameters


Usage: 
`make_methyl_df.py ${PDB}_qFit.pdb`
`calc_OP.py ${PDB}_qFit.dat ${PDB}_qFit.pdb ${PDB}_qFit.out -r ${res} -b ${b_fac}`

Other scripts
`res=$(python get_res.py ${PDB}.pdb ${PDB})`
`b_fac=$(b_factor.py ${PDB}_qFit.pdb --pdb=${PDB})`


### 2. Alpha Carbon RMSD
The purpose of this script is to calculate the distance between each alpha carbon of two sequence matched PDB structures. This script is also dependent on
the two structures having the same numbered residues and chain ids. 

INPUT: 2 PDB structures, names of PDB structures
OUTPUT: CSV file with distance between the alpha carbon atom of every residue between the two input structures

`alpha_carbon_rmsd.py pdb1.pdb pdb2.pdb pdb1_name pdb2_name` 

### 3. B-Factors
The purpose of this script is to calculate the B-factor for every residue across the structure. You can choose to calculate the alpha carbon or side chain (all heavy atoms). This script also returns the average B-factor for alpha carbon which is used in the calc_OP.py script.

INPUT: PDB structure, name of PDB structure, sidechain or CA
OUTPUT: CSV file the b-factor for each residue in the structure, average b-factor for entire structure


Alpha Carbon B-factor:
`b_factor.py pdb.pdb pdb_name --ca`

Heavy Atom B-factor:
`b_factor.py pdb.pdb pdb_name --sidechain`

### 4. Find Multiconformer Residues around Ligands
The purpose of this script is to identify residues with more than one conformer within 5 angstroms any (non-crystallographic) ligands in the PDB
10/31/2022: This script is infrequently used

INPUT: PDB structure, name of PDB structure
OUTPUT: CSV file with the residues within 5A of ligand with the number of single and multi conf structures

`find_altlocs_near_ligand.py pdb.pdb pdb_name` 

### 5. Find largest ligand
The purpose of this script is to identify the largest (non-crystallographic) ligand in the PDB. 

INPUT: PDB structure, name of PDB structure
OUTPUT: A text file named PDB_name_ligand_name.txt with the ligand name inside

`find_altlocs_near_ligand.py pdb.pdb pdb_name` 

### 6. Find Chain & Residue number of ligand
This script is for an automated way to get the residue ID and chain ID of a ligand of interest to be fed into qFit Ligand

INPUT: PDB structure, ligand name
OUTPUT: A text file named Ligand_name_chain_resi.txt with the residue number and chain of the ligand

`get_lig_chain_res.py pdb.pdb pdb_name` 


### 7. Get Sequence
This script will take in a PDB and return a single letter code for every amino acid residue in the PDB. 

INPUT: PDB file and name of PDB
OUTPUT: Text file pdb_name_seq.txt with amino acid sequence as found in PDB

`get_seq.py pdb.pdb --pdb {pdb_name}`

### 8. Get Ligand Occupancy 
This script will take in a PDB and ligand code and return the occupancy and b-factors of each ligand conformer. 

INPUT: PDB file, name of PDB, name of ligand
OUTPUT: Text file {pdb_name}_ligand_occupancy.csv with ligand occupancy information

example:
`lig_occ.py pdb.pdb --pdb {pdb_name} -l {ligand name}`




`qfit_RMSF.py ${PDB}_qFit.pdb --pdb=${PDB}`

`find_largest_lig ${PDB}_qFit.pdb ${PDB}`