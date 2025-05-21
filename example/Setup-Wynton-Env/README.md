# Guide: Running qFit-ligand on Wynton + Environment Setup for Large Datasets

This folder contains all relevant information needed to set up and run **qFit-ligand** jobs on Wynton, as well as process and analyze the resulting data.

## Contents

- `INSTRUCTIONS.md`: Detailed instructions on how to configure directories and job scripts on Wynton.
- `analysis_script_details.md`: Descriptions of all scripts used in the qFit-ligand pipeline, including guidance on how to modify them as needed.

## Minor Note On Submitting qFit-ligand Jobs

The script used to submit **qFit-ligand** jobs is **run_qfit_ligand.sh**.

The standard qFit-ligand command is:

`qfit_ligand composite_omit_map.mtz -sm '${smiles}' -l 2FOFCWT,PH2FOFCWT ${pdb_id}_001.pdb ${chain_res} -p 5`

However, this is a more experimental branch of qFit that uses PLACER (instead of RDKit) for ligand sampling. As a result, the command differs slightly:

`qfit_ligand composite_omit_map.mtz -sm '${smiles}' -l 2FOFCWT,PH2FOFCWT ${pdb_id}.pdb ${chain_res} -p 5 -placer ${pdb_id}_${lig}_model.pdb -targ_chain ${cif_chain}`

### Notes on PLACER Integration

1. **Separate PLACER and qFit Runs**  
   While there probably is a way to import the PLACER model directly into qFit-ligand for sampling (similar to how RDKit is used), this implementation currently runs PLACER and qFit-ligand as separate steps.  
   You must first run PLACER on your structure, then pass the resulting ensemble to qFit-ligand.

   - 1a. When running PLACER, make sure the output follows this naming convention (if you want to use the above command):  
     `${pdb_id}_${lig}_model.pdb`

2. **Use CIF Files with PLACER**  
   PLACER seems to work more reliably when sampling from CIF files. I think this may be because PLACER expects the ligand to be in a different chain than the protein binding pocket, which is a condition usually true for CIFs but not for PDBs.  
   You can manually modify the ligand chain in a PDB file, but I think it is just easier to use the CIF file as input to PLACER.

   However, qFit-ligand still requires a PDB file containing both the protein and ligand. So, both the chain/residue from the PDB and the chain from the CIF (used by PLACER) must be specified.

   - 2a. `${chain_res}`: The chain and residue number of the ligand as defined in the PDB file.
   - 2b. `${cif_chain}`: The chain ID of the same ligand in the original CIF file (should match the chain ID in `${pdb_id}_${lig}_model.pdb`).


