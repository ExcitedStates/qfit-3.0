#!/bin/bash
#$ -l h_vmem=4G
#$ -l mem_free=4G
#$ -t 1-137
#$ -l h_rt=28:00:00
#$ -pe smp 8


#this script will run a pre-qfit refinement based on the input PDB names you have.

#________________________________________________INPUTS________________________________________________#
PDB_file=/wynton/home/fraserlab/jessicaflowers/tp_qfit_ligands/tp_ligands.txt
base_dir='/wynton/home/fraserlab/jessicaflowers/tp_qfit_ligands'

#________________________________________________SET PATHS________________________________________________#
source /wynton/home/fraserlab/jessicaflowers/phenix/phenix-1.20.1-4487/phenix_env.sh
export PATH="/wynton/home/fraserlab/jessicaflowers/miniconda3/bin:$PATH"
source activate qfit_ligand
export PHENIX_OVERWRITE_ALL=true

#________________________________________________RUN PRE-QFIT REFINEMENT________________________________________________#
PDB=$(cat $PDB_file | head -n $SGE_TASK_ID | tail -n 1)

# Extract and format the required information
pdb_id=$(echo "$PDB" | awk -F, '{print $1}')
chain_res=$(echo "$PDB" | awk -F, '{print $(NF-3) "," $(NF-2)}')
lig_name=$(echo "$PDB" | awk -F, '{print $5}')

echo "chain_res ${chain_res}"
echo "lig_name ${lig_name}"

cd ${base_dir}/${pdb_id}
echo "Working in directory ${base_dir}/${pdb_id}"

# Remove altlocs 
single=$(remove_altconfs ${pdb_id}.pdb) 
no_dup=$(remove_duplicates ${pdb_id}.single.pdb) 


# Convert cif file to mtz
phenix.cif_as_mtz ${pdb_id}-sf.cif --merge
echo "${pdb_id}-sf.cif merging into mtz"
if [ ! -f "${pdb_id}-sf.mtz" ]; then
    echo "merge failed"
fi

# Move the converted file to the new mtz. If conversion fails, script uses original mtz
echo "moving cif file"
mv ${pdb_id}-sf.mtz ${pdb_id}.mtz
echo "cif file moved "

# Determine FOBS VS IOBS VS FP
mtzmetadata=`phenix.mtz.dump "${pdb_id}.mtz"`

obstypes=("FP" "FOBS" "F-obs" "I" "IOBS" "I-obs" "F(+)" "I(+)" "FSIM")
ampfields=`grep -E "amplitude|intensity|F\(\+\)|I\(\+\)" <<< "${mtzmetadata}"`

# Clear xray_data_labels variable
xray_data_labels=""

# Is amplitude an Fo?
for field in ${ampfields}; do
  # Check field in obstypes
  if [[ " ${obstypes[*]} " =~ " ${field} " ]]; then
    # Check SIGFo is in the mtz too!
    if grep -F -q -w "SIG$field" <<< "${mtzmetadata}"; then
      xray_data_labels="${field},SIG${field}";
      break
    fi
  fi
done
if [ -z "${xray_data_labels}" ]; then
  echo >&2 "Could not determine Fo field name with corresponding SIGFo in .mtz.";
  echo >&2 "Was not among "${obstypes[*]}". Please check .mtz file\!";
  exit 1;
else
  echo "data labels: ${xray_data_labels}"
fi

# Run ready_set
echo "Running ready set in direcory $PWD"
phenix.ready_set ${pdb_id}.single.pdb.fixed

# Run refinement 

# Check if the .ligands.cif file exists
if [ -f "${pdb_id}.single.pdb.ligands.cif" ]; then
    echo "ready_set worked"
    # Execute phenix.refine with the .ligands.cif file
    phenix.refine ${pdb_id}.mtz ${pdb_id}.single.pdb.updated.pdb \
    refinement.input.monomers.file_name=${pdb_id}.single.pdb.ligands.cif \
    refinement.refine.strategy=individual_sites+individual_adp+occupancies \
    refinement.output.prefix=${pdb_id} \
    refinement.main.number_of_macro_cycles=5 \
    refinement.main.nqh_flips=True \
    refinement.output.write_maps=False \
    refinement.hydrogens.refine=riding \
    refinement.main.ordered_solvent=True \
    refinement.target_weights.optimize_xyz_weight=true \
    refinement.target_weights.optimize_adp_weight=true \
    refinement.input.xray_data.r_free_flags.generate=True \
    refinement.input.xray_data.labels=${xray_data_labels}
else
    echo "ready_set failed, trying elbow"
    phenix.elbow ${pdb_id}.single.pdb.fixed --residue ${lig_name}

    # Check if phenix.elbow generated the CIF file
    if [ -f "elbow.${lig_name}.${pdb_id}_single_pdb_fixed.cif" ]; then
        # Execute phenix.refine with the generated .cif file
        phenix.refine ${pdb_id}.mtz ${pdb_id}.single.pdb.fixed \
        refinement.input.monomers.file_name=elbow.${lig_name}.${pdb_id}_single_pdb_fixed.cif \
        refinement.refine.strategy=individual_sites+individual_adp+occupancies \
        refinement.output.prefix=${pdb_id} \
        refinement.main.number_of_macro_cycles=5 \
        refinement.main.nqh_flips=True \
        refinement.output.write_maps=False \
        refinement.hydrogens.refine=riding \
        refinement.main.ordered_solvent=True \
        refinement.target_weights.optimize_xyz_weight=true \
        refinement.target_weights.optimize_adp_weight=true \
        refinement.input.xray_data.r_free_flags.generate=True \
        refinement.input.xray_data.labels=${xray_data_labels}
    else
        echo "elbow failed"
    fi
fi

#RUN COMPOSITE OMIT MAP
phenix.composite_omit_map ${pdb_id}.mtz ${pdb_id}_001.pdb omit-type=refine nproc=8 r_free_flags.generate=True exclude_bulk_solvent=True

