# qFit 2024.3

![](https://github.com/ExcitedStates/qfit-3.0/workflows/tests/badge.svg)

If you use this software, please cite:
- [Wankowicz SA, Ravikumar A, Sharma S, Riley BT, Raju A, Hogan DW, van den Bedem H, Keedy DA, & Fraser JS. Uncovering Protein Ensembles: Automated Multiconformer Model Building for X-ray Crystallography and Cryo-EM. eLife. (2023).](https://www.biorxiv.org/content/10.1101/2023.06.28.546963v2.abstract)
- [Riley BT, Wankowicz SA, et al. qFit 3: Protein and ligand multiconformer modeling for X-ray crystallographic and single-particle cryo-EM density maps. Protein Sci. 30, 270–285 (2021)](https://dx.doi.org/10.1002/pro.4001)
- [van Zundert, G. C. P. et al. qFit-ligand Reveals Widespread Conformational Heterogeneity of Drug-Like Molecules in X-Ray Electron Density Maps. J. Med. Chem. 61, 11183–11198 (2018)](https://dx.doi.org/10.1021/acs.jmedchem.8b01292)
- [Keedy, D. A., Fraser, J. S. & van den Bedem, H. Exposing Hidden Alternative Backbone Conformations in X-ray Crystallography Using qFit. PLoS Comput. Biol. 11, e1004507 (2015)](https://dx.doi.org/10.1371/journal.pcbi.1004507)

## Refinement
After *multiconformer_model2.pdb* has been generated, the model must need to be refined. Bear in mind that this final step currently depends on an existing installation of the Phenix software suite. This script is currently written to work with version Phenix 1.21.

[Phenix installation](https://phenix-online.org/documentation/install-setup-run.html)

X-ray crystallography:
`qfit_final_refine_xray.sh /path/to/mtz_file.mtz multiconformer_model2.pdb`

Cryo-EM: 
`qfit_final_refine_cryoem.sh /path/to/ccp4_file.ccp4 original_pdb.pdb multiconformer_model2.pdb`

Segment Only:
This script should only be used after running qFit_protein --only-segment 
`qfit_segment_refine.sh /path/to/mtz_file.mtz multiconformer_model2.pdb`

## Analysis Scripts
These analysis scripts should be used on outputs after refinement. Many can be done using other populate structrual programs, but we have amended many of them here to work with multiconformer models. 

### 1. Calculating Order Parameters
You will first need to run ascript is to produce the intermediate file to calculate crystallographic order parameters. 

INPUT: PDB structure, pdb name

OUTPUT: pdb_name.dat A tab seperated file with information about each residue and the atom type need to calculate cyrstallographic order parameters.

Usage: 
`make_methyl_df.py ${PDB}_qFit.pdb`

You then can use the intermeidate file to calculate the order parameters. Please note you will need the resolution and average alpha-carbon b-factor for normalization (see scripts below). 

INPUT: Intermdiate file (obtained about), PDB structure, output_file name, resolution, average alpha-carbon b-factor

OUTPUT: pdb_name.dat A tab seperated file with information about each residue and the atom type need to calculate cyrstallographic order parameters.

Usage: 
`calc_OP.py ${PDB}.dat ${PDB}_qFit.pdb ${PDB}_qFit_order_parm.out -r ${res} -b ${b_fac}`

If using these scripts, please cite: 
Fenwick, R. B., van den Bedem, H., Fraser, J. S., & Wright, P. E. (2014). Integrated description of protein dynamics from room-temperature X-ray crystallography and NMR. Proceedings of the National Academy of Sciences, 111(4), E445-E454.

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
The purpose of this script is to identify residues with more than one conformer within 5 angstroms any (non-crystallographic) ligands in the PDB. This script is infrequently used.

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


### 8. Get SMILES String
This script will take in a ligand name and return its SMILES string, as specified on the Protein Data Bank.

INPUT: Ligand name

OUTPUT: SMILES string written to console

`get_smiles.py LIG`


### 9. Get Ligand Occupancy 
This script will take in a PDB and ligand code and return the occupancy and b-factors of each ligand conformer. 

INPUT: PDB file, name of PDB, name of ligand

OUTPUT: Text file {pdb_name}_ligand_occupancy.csv with ligand occupancy information

example:
`lig_occ.py pdb.pdb --pdb {pdb_name} -l {ligand name}`

### 10. Get Root Mean Squared Flucuations (RMSF) for each residue 

This script will take in a PDB and ligand code and return the occupancy and b-factors of each ligand conformer. 

INPUT: PDB file, PDB name

OUTPUT: Text file {pdb_name}_qfit_RMSF.csv with weighted RMSF calculated for each amino acid. 

example:
`qfit_RMSF.py {PDB}_qFit.pdb --pdb={PDB}`

### 11. Relabel chains of matching PDB

This script will rename chains in one PDB (holo) one based how close via RMSD is on corresponding PDB (apo).

INPUT: 2 PDB, 2 PDB names

OUTPUT: PDB with renamed chain(s) 

example:
`relabel_chain.py holo_pdb.pdb apo_pdb.pdb --holo_name {holo name} --apo_name {holo name}`

### 12. Subset structures based on proximity to ligand
This script will take in 2 pdbs and a ligand or geometric point in the PDB and the PDB names and output a list of overlapping ligands and a list of close residues (determined by -distance).

INPUT: 2 PDB, 2 PDB names, ligand (optional: distance)

OUTPUT: Text file with list of close residue, text file with list of overlapping ligands 

example:
`subset_structure_AH.py holo_pdb.pdb apo_pdb.pdb --holo_name {holo name} --apo_name {holo name} -ls {ligand name}`


### 13. Water scripts

water_clash.py: This will take in two PDBs, one containing water molecules, one containing only protein or protein/hetatoms.
It will then determine how many clashes occur between the two and adjust accordingly.

INPUT: 2 PDB (one with water molecules, one without), 2 PDB names

OUTPUT: CSV file with output of residues that clash with water molecules.

example: 
`water_clash.py pdb_with_no_water.pdb pdb_with_water.pdb --nowater_name {pdb name} --water_name {pdb name}`

water_stats: This script will take in a PDB structure and output multiple csv files describing the closest protein residue and distance away from water molecules.

INPUT: PDB file

OUTPUT: Multiple CSV file with output of residues that are close to water molecules

example: 
`water_stats.py pdb.pdb --dist {distance between protein and water} --pdb {pdb name}`

### 14. Calculate the RSCC of a ligand and density map 
This script will calculate the RSCC of a ligand defined by its residue number and chain. This script can either calcualte the RSCC of a single model of interest, or the RSCC of two models against the same density map in the same voxel space.

INPUT: Density map, protein-ligand pdb file, chain, residue number 

OUTPUT: RSCC of the input model(s) and map printed to the console 

example: 
`calc_rscc.py MAP_FILE MODEL.pdb CHAIN,RESIDUE_ID -l 2FOFCWT,PH2FOFCWT -comp COMPARISON_MODEL.pdb`

### 15. Calculate the RMSD between two conformers
This script calculates the RMSD between two structures (PDB files) using their atomic coordinates. 

INPUT: 2 PDBs, PDB name

OUTPUT: A CSV file with the calculated RMSD value between the two structures.

example: 
`calc_rmsd.py conformer1.pdb conformer2.pdb --pdb PDB_NAME`
