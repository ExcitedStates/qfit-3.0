#!/bin/bash

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

#__________________________________REMOVE DUPLICATE HET ATOMS__________________________________
remove_duplicates "${multiconf}"

#________________________________REMOVE TRAILING HYDROGENS___________________________________
phenix.pdbtools remove="element H" "${multiconf}.fixed"

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
obstypes="FP FOBS F-obs IOBS"

# Get amplitude fields
ampfields=`grep "amplitude" <<< "${mtzmetadata}"`
ampfields=`echo "${ampfields}" | awk '{$1=$1};1' | cut -d " " -f 1`

# Clear xray_data_labels variable
xray_data_labels=""

# Is amplitude an Fo?
for field in ${ampfields}; do
  # Check field in obstypes
  if grep -F -q -w $field <<< "${obstypes}"; then
    # Check SIGFo is in the mtz too!
    if grep -F -q -w "SIG$field" <<< "${mtzmetadata}"; then
      xray_data_labels="${field},SIG${field}";
      break
    fi
  fi
done
if [ -z "${xray_data_labels}" ]; then
  echo >&2 "Could not determine Fo field name with corresponding SIGFo in .mtz.";
  echo >&2 "Was not among ${obstypes}. Please check .mtz file\!";
  exit 1;
else
  echo "data labels: ${xray_data_labels}"
fi

#_____________________________DETERMINE R FREE FLAGS______________________________
gen_Rfree=True
rfreetypes="FREE R-free-flags"
for field in ${rfreetypes}; do
  if grep -F -q -w $field <<< "${mtzmetadata}"; then
    gen_Rfree=False;
    echo "Rfree column: ${field}";
    break
  fi
done

#__________________________________GET CIF FILE__________________________________
phenix.ready_set pdb_file_name="${multiconf}.f_modified.pdb"

#__________________________________DETERMINE IF THERE ARE LIGANDS__________________________________
if [ -f "${multiconf}.f_modified.ligands.cif" ]; then
  phenix.refine ${multiconf}.f_modified.updated.pdb \
              ${pdb_name}.mtz \
              ${multiconf}.f_modified.ligands.cif \
              strategy=individual_sites \
              output.prefix=${pdb_name} \
              output.serial=2 \
              main.number_of_macro_cycles=5 \
              refinement.input.xray_data.r_free_flags.generate=True \
              refinement.input.xray_data.labels=$xray_data_labels \
              write_maps=false --overwrite
else
  phenix.refine ${multiconf}.f_modified.updated.pdb \
              ${pdb_name}.mtz \
              strategy=individual_sites \
              output.prefix=${pdb_name} \
              output.serial=2 \
              main.number_of_macro_cycles=5 \
              refinement.input.xray_data.r_free_flags.generate=True \
              refinement.input.xray_data.labels=$xray_data_labels \
              write_maps=false --overwrite
fi


#__________________________________REFINE UNTIL OCCUPANCIES CONVERGE__________________________________
zeroes=50
while [ $zeroes -gt 10 ]; do
  if [[ -e "${multiconf}.f_modified.ligands.cif" ]]; then
        phenix.refine ${pdb_name}_002.pdb ${pdb_name}.mtz \
              ${multiconf}.f_modified.ligands.cif \
              output.prefix=${pdb_name} \
              output.serial=3 \
              strategy="individual_sites" \
              main.number_of_macro_cycles=5 \
              #refinement.input.xray_data.r_free_flags.label='FREE' \
              #refinement.input.xray_data.labels=$xray_data_labels \
              write_maps=false --overwrite
  else
        phenix.refine ${pdb_name}_002.pdb \
              ${pdb_name}.mtz \
              strategy="individual_sites" \
              output.prefix=${pdb_name} \
              output.serial=3 \
              main.number_of_macro_cycles=5 \
              #refinement.input.xray_data.r_free_flags.label='FREE' \
              #refinement.input.xray_data.labels=$xray_data_labels \
              write_maps=false --overwrite
  
  fi
  zeroes=`normalize_occupancies -occ 0.09 ${pdb_name}_003.pdb`
  normalize_occupancies -occ 0.09 ${pdb_name}_003.pdb
  
  echo 'Post refinement Zeroes:'
  echo $zeroes
  
  if [ ! -f ${pdb_name}_003_norm.pdb ]; then
     echo 'normalize occupanies did not work!'
     exit
  fi
  remove_duplicates ${pdb_name}_003_norm.pdb
  mv ${pdb_name}_003_norm.pdb.fixed ${pdb_name}_002.pdb
done

#__________________________________ADD HYDROGENS__________________________________
phenix.reduce ${pdb_name}_002.pdb > ${pdb_name}_004.pdb

#__________________________________FINAL REFINEMENT__________________________________
if [[ -e "multiconformer_model2.ligands.cif" ]]; then
phenix.refine ${pdb_name}_004.pdb ${pdb_name}.mtz \
              ${multiconf}.f_modified.ligands.cif \
              "$adp" \
              output.prefix=${pdb_name} \
              output.serial=5 \
              strategy="*individual_sites *individual_adp" \
              main.number_of_macro_cycles=5 \
              #refinement.input.xray_data.r_free_flags.label='FREE' \
              #refinement.input.xray_data.labels=$xray_data_labels \
              write_maps=false \
              --overwrite
else
  phenix.refine ${pdb_name}_004.pdb ${pdb_name}.mtz \
              ${pdb_name}_004.pdb \
              "$adp" \
              output.prefix=${pdb_name} \
              output.serial=5 \
              strategy="*individual_sites *individual_adp" \
              main.number_of_macro_cycles=5 \
              #refinement.input.xray_data.r_free_flags.label='FREE' \
              #refinement.input.xray_data.labels=$xray_data_labels \
              write_maps=false \
              --overwrite
fi


cp ${pdb_name}_005.pdb ${pdb_name}_qFit.pdb
cp ${pdb_name}_005.mtz ${pdb_name}_qFit.mtz
cp ${pdb_name}_005.log ${pdb_name}_qFit.log
