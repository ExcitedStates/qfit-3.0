## Setting Up Wynton Environment for qFit-Ligand

This tutorial will walk you through how to set up and run qFit-Ligand on a dataset. The information here specifically pertains to the true positive (TP) dataset, but can be applied to any set of PDBs you are interested in. All of the scripts discussed in this file can be found within this folder. 

### 1. Create a New Directory
First, navigate to your Wynton workstation and make a directory for storing your qFit-ligand files. For example:

`mkdir tp_qfit_ligands`

`cd tp_qfit_ligands`

### 2. Create a Text File `tp_ligands.txt` Containing Ligand PDB Information
Inside the directory, create a text file that lists the PDBs containing ligands.

`vi tp_ligands.txt`

Inside the file, store the data in this format:

`PDB,chain,residue number,SMILES,ligand name`

Example entry, make sure there are NO spaces:

`8BSD,A,7220,c1cn(c2c1c(ncn2)N)C3C(C(C(O3)CO)O)O,TBN`

### 3. Create the wget_pdb.sh Script To Fetch PDBs
Create the following script to fetch the PDB and MTZ files from the Protein Data Bank:

`vi wget_pdb.sh`

Note, **base_folder** must be the directory you are working in (i.e. tp_qfit_ligands) and **pdb_filelist** must be the text file you created (i.e. tp_ligands.txt). Then execute the script. 

### 4. Create the pre_qfit_refine.sh Script
Create the following script to process your data, preparing it for a qFit-Ligand run. 

`vi pre_qfit_refine.sh`

This script loops through the folders created by wget_pdb.sh and removes any existing multiconformers from those PDBs. It runs a 5 macrocycle refinement with Phenix. Execute the script.

The important outputs from this script are the refined PDB file, and the composite omit map. These will be used as input for qFit-Ligand in the next script. 

Notes: 

* In the shell script header, replace the value **‘135’** with the number of PDBs in your dataset. If you are running on the TP dataset, there is no need to change this value.
* Make sure you are sourcing the correct environment in the **SET PATHS** section of this script. 

### 5. Create the run_qfit_ligand.sh Script
Create the following script to run a qFit-Ligand job.

`vi run_qfit_ligand.sh`

The **category** in the script (see line 14) allows you to label and organize qFit runs by creating subdirectories in each PDB folder, named based on the chosen category. This helps separate and easily track the results of different runs, ensuring clarity on the purpose of each run and facilitating future reference or comparisons. 

Execute the script with qsub. 

`chmod +x run_qfit_ligand.sh`

`qsub run_qfit_ligand.sh`

The outputs from each qFit-Ligand run are called **multiconformer_ligand_bound_with_protein.pdb** and **multiconformer_ligand_only.pdb**. This script will automatically proceed to run a post-qFit refinement on **multiconformer_ligand_bound_with_protein.pdb**.

The final outputs will be named:

* {pdb_name}_qFit_ligand.mtz
* {pdb_name}_qFit_ligand.pdb

### 6. Create the test_qfit_outputs.sh Script
The purpose of this last script is to analyze the results of your qFit-Ligand runs. First, you must create a directory (in your working directory where all of your scripts are stored) to store the results. 

`cd tp_qfit_ligands`

`mkdir output_data`

`cd output_data`

`mkdir results_from_qfit_run_1`

Go back to your working directory and set up the analysis script.

`cd tp_qfit_ligands`

`vi test_qfit_outputs.sh`

Notes: 

* You can name the results_from_qfit_run_1 folder anything you wish, but it should be clearly associated with the **category** you created in step 5 (line 14 of run_qfit_ligand.sh) 
* Line 22 of test_qfit_outputs.sh should define the path leading to results_from_qfit_run_1.
* Line 25 of test_qfit_outputs.sh should be the same category you created in line 14 of run_qfit_ligand.sh

This script will loop through each folder, and examine results from the specified category.  It performs the following calculations:
* Real space correlation coefficient (comparing RSCC of the qFit-ligand generated model to the RSCC of the deposited model)
* EDIAm of the qFit-ligand generated model
* Total qFit-Ligand runtime 
* Number of output conformers
* Torsion strain of the each of the qFit-Ligand output conformers and deposited conformers  

To execute, 

`chmod +x test_qfit_outputs.sh`

`qsub test_qfit_outputs.sh`
