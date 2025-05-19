## Details on analysis scripts

This folder contains scripts (namely, *test_qfit_outputs.sh*) that call several python files in order to perform the necessary analysis. Here, I will go through the inputs and outputs of each of those scripts, so you can edit **test_qfit_outputs.sh**, should you need to.

### 1. split_multiconformer_ligand.py
This script is in the qFit GitHub repository qfit-3.0/scripts/post. Its purpose is to process a multiconformer protein-ligand or ligand-only PDB file and generate individual PDB files for each ligand conformer present. Here, I use it to generate individual PDB files for every ligand conformer present in the qFit-ligand output model, as well as every conformer present in the deposited model. 

**INPUTS:**
- PDB structure file 

* **--residue**: Ligand chain and residue number 

* **--directory**: Directory where you wish to save the results

* **--output_name**: Desired file name for labeling the output PDBs


Note on naming mechanism: 

`--output_name depo`

This will output files named **depo_ligand_A.pdb** and **depo_ligand_B.pdb** assuming the input pdb file has 2 ligand conformations.

### 2. calc_rscc.py
This script is designed to compute the RSCC (Real-Space Correlation Coefficient) of a ligand model using a density map. This script supports two use cases:

1. Calculate the RSCC of a single model of interest (e.g., a multiconformer model from qFit-ligand).
   
2. Compare the RSCC of two models, e.g. a multiconformer qFit-ligand model and a deposited single-conformer model, 
   by evaluating them against the same density map in the same voxel space.

Here, I am using the second use case. 

**INPUTS:**

- Refined qFit-ligand MTZ
  
- Refined qFit-ligand PDB
  
- Ligand chain and residue number
  
* **--label**: MTZ column labels to build density
* **--comp**: Deposited PDB file, for comparison

### 3. parse_log_file.py
This python file can be found in this folder. It must be in your working directory **tp_qfit_ligands** in order to execute **test_qfit_outputs.sh**. Its purpose is to read and parse the automatically generated qFit-Ligand .log file to determine the number of output conformers and overall runtime.

**INPUTS:**
* **--pdb**: PDB name for labeling in the output CSV file 
* **--path**: Path to data (log file)
* **--output_dir**: Directory where you wish to save the results

### 4. calc_torsion_strain.py
This python file can be found in this folder. It must be in your working directory **tp_qfit_ligands** in order to execute **test_qfit_outputs.sh**. Its purpose is to calculate the torsion strain of a multiconformer ligand. It calculates the strain for the individual conformers, and weights them by occupancy to find total strain. To use this script, you must install the software published in this paper:

https://pubs.acs.org/doi/full/10.1021/acs.jcim.1c00368

https://tldr.docking.org/

Make note the directory where the software is installed. You will need to edit the contents of **calc_torsion_strain.py** to set the correct path, see line 21. 

**INPUTS:**

* **--pdb**: PDB name for labeling in the output CSV file 
* **--path**: Path to data (multiconformer pdb file)
* **--output_dir**: Directory where you wish to save the results
* **--conf_type**: Name of the PDB you want to analyze 

Note that you should label `--conf_type` based on the `--output_name` specified in **split_multiconformer_ligand.py**

For example, I mentioned earlier that when you split a multiconformer model,  `--output_name depo` will output files named **depo_ligand_A.pdb** and **depo_ligand_B.pdb** assuming the input pdb file has 2 ligand conformations. 

Thus, if you want to calculate the strain of these two PDBs, the `--conf_type` should be set to **depo_ligand**. 

Here, specifying `--conf_type qfit_ligand` vs `--conf_type depo_ligand` would calculate the strain of qFit and deposited models, respectively.  

### 5. edia.py
This python file can be found in **qfit-3.0/src/qfit**. Its purpose is to calculate the EDIAm of a multiconformer ligand.

**INPUTS:**

- Refined qFit-ligand CCP4 map
  
- Resolution of map
  
- Refined qFit-ligand PDB file
  
* **--selection**: Ligand chain and residue number 
* **--directory**: Directory where you wish to save the results
* **--pbd_name**: Desired file name for labeling the output PDBs


