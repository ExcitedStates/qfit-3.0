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
remove_duplicates multiconformer_model2.pdb

#________________________________REMOVE TRAILING HYDROGENS___________________________________
phenix.pdbtools remove="element H" multiconformer_model2.pdb.fixed

#__________________________________DETERMINE RESOLUTION AND (AN)ISOTROPIC REFINEMENT__________________________________
resrange= phenix.mtz.dump ${pdb_name}.mtz | grep "Resolution range:"

echo $resrange

res=`echo $resrange[4] | cut -c 1-5`
res1000=`echo $res | awk '{tot = $1*1000}{print tot }'`

if ($res1000<1550) then
  adp='adp.individual.anisotropic="not (water or element H)"'
else
  adp='adp.individual.isotropic=all'
fi


#__________________________________DETERMINE FOBS v SIGOBS__________________________________
if grep -F _refln.F_meas_au ${pdb_name}-sf.cif; then
       xray_data_labels="FOBS,SIGFOBS"
else
       xray_data_labels="IOBS,SIGIOBS"       
fi


#__________________________________GET CIF FILE__________________________________
phenix.ready_set pdb_file_name=multiconformer_model2.pdb.f_modified.pdb

#_____________________________DETERMINE R FREE FLAGS______________________________
phenix.mtz.dump ${pdb_name}.mtz > ${pdb_name}_mtzdump.out

if grep -q FREE ${pdb_name}_mtzdump.out; then
  Rfree_flags=False
else
  Rfree_flags=True
fi

#__________________________________DETERMINE IF THERE ARE LIGANDS__________________________________
if [ -f "multiconformer_model2.pdb.f_modified.ligands.cif" ]; then
  phenix.refine multiconformer_model2.pdb.f_modified.updated.pdb \
              ${pdb_name}.mtz \
              multiconformer_model2.pdb.f_modified.ligands.cif \
              strategy=individual_sites \
              output.prefix=${pdb_name} \
              output.serial=2 \
              main.number_of_macro_cycles=5 \
              refinement.input.xray_data.r_free_flags.generate=True \
              refinement.input.xray_data.labels=$xray_data_labels \
              write_maps=false --overwrite
else
  phenix.refine multiconformer_model2.pdb.f_modified.updated.pdb \
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
  if [[ -e "multiconformer_model2.pdb.f_modified.ligands.cif" ]]; then
        phenix.refine ${pdb_name}_002.pdb ${pdb_name}.mtz \
              multiconformer_model2.pdb.f_modified.ligands.cif \
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
              multiconformer_model2.pdb.f_modified.ligands.cif \
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
