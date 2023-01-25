#!/bin/bash
# This script works with Phenix version 1.20.

#This script should be used ONLY after using only-segment. If using qFit protein, please use qfit_final_refine_xray.sh.

qfit_usage() {
  echo >&2 "Usage:";
  echo >&2 "  $0 mapfile.mtz [multiconformer_model2.pdb]";
  echo >&2 "";
  echo >&2 "mapfile.mtz and multiconformer_model2.pdb MUST exist in this directory.";
  echo >&2 "Outputs will be written to mapfile_qFit.{pdb|mtz|log}.";
  exit 1;
}

#___________________________SOURCE__________________________________
# Check that Phenix is loaded
if [ -z "${PHENIX}" ]; then
  echo >&2 "I require PHENIX but it's not loaded.";
  echo >&2 "Please \`source phenix_env.sh\` from where it is installed.";
  exit 1;
else
  export PHENIX_OVERWRITE_ALL=true;
fi

# Check that qFit is loaded.
command -v remove_duplicates >/dev/null 2>&1 || {
  echo >&2 "I require qFit (remove_duplicates) but it's not loaded.";
  echo >&2 "Please activate the environment where qFit is installed.";
  echo >&2 "   e.g. conda activate qfit"
  exit 1;
}

# Assert required files exist
mapfile=$1
multiconf=${2:-multiconformer_model2.pdb}
echo "mapfile              : ${mapfile} $([[ -f ${mapfile} ]] || echo '[NOT FOUND]')";
echo "qfit unrefined model : ${multiconf} $([[ -f ${multiconf} ]] || echo '[NOT FOUND]')";
echo "";
if [[ ! -f "${mapfile}" ]] || [[ ! -f "${multiconf}" ]]; then
  qfit_usage;
fi
pdb_name="${mapfile%.mtz}"


#__________________________________DETERMINE RESOLUTION AND (AN)ISOTROPIC REFINEMENT__________________________________
mtzmetadata=`phenix.mtz.dump "${pdb_name}.mtz"`
resrange=`grep "Resolution range:" <<< "${mtzmetadata}"`

echo "${resrange}"

res=`echo "${resrange}" | cut -d " " -f 4 | cut -c 1-5`
res1000=`echo $res | awk '{tot = $1*1000}{print tot }'`

if (( $res1000 < 1550 )); then
  adp='adp.individual.anisotropic="not (water or element H)"'
else
  adp='adp.individual.isotropic=all'
fi

#__________________________________DETERMINE FOBS v IOBS v FP__________________________________
# List of Fo types we will check for
obstypes=("FP" "FOBS" "F-obs" "I" "IOBS" "I-obs" "F(+)" "I(+)")

# Get amplitude fields
ampfields=`grep -E "amplitude|intensity|F\(\+\)|I\(\+\)" <<< "${mtzmetadata}"`
ampfields=`echo "${ampfields}" | awk '{$1=$1};1' | cut -d " " -f 1`

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
  # Start writing refinement parameters into a parameter file
  echo "refinement.input.xray_data.labels=$xray_data_labels" > ${pdb_name}_refine.params
fi

#_____________________________DETERMINE R FREE FLAGS______________________________
gen_Rfree=True
rfreetypes="FREE R-free-flags"
for field in ${rfreetypes}; do
  if grep -F -q -w $field <<< "${mtzmetadata}"; then
    gen_Rfree=False;
    echo "Rfree column: ${field}";
    echo "refinement.input.xray_data.r_free_flags.label=${field}" >> ${pdb_name}_refine.params
    break
  fi
done
echo "refinement.input.xray_data.r_free_flags.generate=${gen_Rfree}" >> ${pdb_name}_refine.params

#__________________________________REMOVE DUPLICATE HET ATOMS__________________________________
remove_duplicates "${multiconf}"


#________________________________REMOVE TRAILING HYDROGENS___________________________________
phenix.pdbtools remove="element H" "${multiconf}.fixed"

#__________________________________GET CIF FILE__________________________________

phenix.ready_set hydrogens=false \
                 trust_residue_code_is_chemical_components_code=true \
                 pdb_file_name="${multiconf}.f_modified.pdb"
# If there are no unknown ligands, ready_set doesn't output a file. We have to do it.
if [ ! -f "${multiconf}.f_modified.updated.pdb" ]; then
  cp -v "${multiconf}.f_modified.pdb" "${multiconf}.f_modified.updated.pdb";
fi
if [ -f "${multiconf}.f_modified.ligands.cif" ]; then
  echo "refinement.input.monomers.file_name='${multiconf}.f_modified.ligands.cif'" >> ${pdb_name}_refine.params
fi

#__________________________________REFINE WITH OCCUPANCIES RESTRAINED__________________________________
# Write refinement parameters into parameters file
echo "refinement.refine.strategy=*individual_sites *individual_adp *occupancies"  > ${pdb_name}_occ_refine.params
echo "refinement.output.prefix=${pdb_name}"                                      >> ${pdb_name}_occ_refine.params
echo "refinement.output.serial=1"                                                >> ${pdb_name}_occ_refine.params
echo "refinement.main.number_of_macro_cycles=5"                                  >> ${pdb_name}_occ_refine.params
echo "refinement.main.nqh_flips=False"                                            >> ${pdb_name}_occ_refine.params
echo "refinement.refine.${adp}"                                                  >> ${pdb_name}_occ_refine.params
echo "refinement.output.write_maps=False"                                        >> ${pdb_name}_occ_refine.params

if [ -f "${multiconf}.f_modified.ligands.cif" ]; then
  echo "refinement.input.monomers.file_name='${multiconf}.f_modified.ligands.cif'" >> ${pdb_name}_occ_refine.params
fi


phenix.refine "${multiconf}.f_modified.updated.pdb" \
                "${pdb_name}.mtz" \
                "${pdb_name}_occ_refine.params" \
                qFit_occupancy.params \
                --overwrite

#__________________________________ADD HYDROGENS__________________________________
# The first round of refinement regularizes geometry from qFit.
# Here we add H with phenix.ready_set. Addition of H to the backbone is important
#   since it introduces planarity restraints to the peptide bond.
# We will also create a cif file for any ligands in the structure at this point.
phenix.ready_set hydrogens=true pdb_file_name="${pdb_name}_001.pdb"
mv "${pdb_name}_001.updated.pdb" "${pdb_name}_001.pdb"

#__________________________________FINAL REFINEMENT__________________________________

# Write refinement parameters into parameters file
echo "refinement.refine.strategy=*individual_sites *individual_adp *occupancies"  > ${pdb_name}_final_refine.params
echo "refinement.output.prefix=${pdb_name}"      >> ${pdb_name}_final_refine.params
echo "refinement.output.serial=5"                >> ${pdb_name}_final_refine.params
echo "refinement.main.number_of_macro_cycles=5"  >> ${pdb_name}_final_refine.params
echo "refinement.main.nqh_flips=True"            >> ${pdb_name}_final_refine.params
echo "refinement.refine.${adp}"                  >> ${pdb_name}_final_refine.params
echo "refinement.output.write_maps=False"        >> ${pdb_name}_final_refine.params
echo "refinement.hydrogens.refine=riding"        >> ${pdb_name}_final_refine.params
echo "refinement.main.ordered_solvent=True"      >> ${pdb_name}_final_refine.params


if [ -f "${pdb_name}_001.ligands.cif" ]; then
  echo "refinement.input.monomers.file_name='${pdb_name}_001.ligands.cif'"  >> ${pdb_name}_final_refine.params
fi

phenix.refine "${pdb_name}_001.pdb" "${pdb_name}_001.mtz" "${pdb_name}_final_refine.params" --overwrite 

#________________________________CHECK FOR REDUCE ERRORS______________________________
if [ -f "reduce_failure.pdb" ]; then
  echo "refinement.refine.strategy=*individual_sites *individual_adp *occupancies"  > ${pdb_name}_final_refine_noreduce.params
  echo "refinement.output.prefix=${pdb_name}"      >> ${pdb_name}_final_refine_noreduce.params
  echo "refinement.output.serial=5"                >> ${pdb_name}_final_refine_noreduce.params
  echo "refinement.main.number_of_macro_cycles=5"  >> ${pdb_name}_final_refine_noreduce.params
  echo "refinement.main.nqh_flips=False"           >> ${pdb_name}_final_refine_noreduce.params
  echo "refinement.refine.${adp}"                  >> ${pdb_name}_final_refine_noreduce.params
  echo "refinement.output.write_maps=False"        >> ${pdb_name}_final_refine_noreduce.params
  echo "refinement.hydrogens.refine=riding"        >> ${pdb_name}_final_refine_noreduce.params
  echo "refinement.main.ordered_solvent=True"      >> ${pdb_name}_final_refine_noreduce.params
  

  phenix.refine "${pdb_name}_001.pdb" "${pdb_name}_001.mtz" "${pdb_name}_final_refine_noreduce.params" --overwrite
fi

#__________________________________NAME FINAL FILES__________________________________
cp -v "${pdb_name}_005.pdb" "${pdb_name}_qFit.pdb"
cp -v "${pdb_name}_005.mtz" "${pdb_name}_qFit.mtz"
cp -v "${pdb_name}_005.log" "${pdb_name}_qFit.log"

#__________________________COMMENTARY FOR USER_______________________________________
if [ -f "${pdb_name}_005.pdb" ]; then 
   echo ""
   echo "[qfit_final_refine_xray] Refinement is complete."
   echo "                         Please be sure to INSPECT your refined structure, especially all new altconfs."
   echo "                         The output can be found at ${pdb_name}_qFit.(pdb|mtz|log) ."
else
   echo "Refinement did not complete."
   echo "Please check for failure reports by examining log files."
fi

if [ -f "reduce_failure.pdb" ]; then
  echo "Refinement was run without checking for flips in NQH residues due to memory constraints. Please inspect your structure."
fi

