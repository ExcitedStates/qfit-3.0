#!/bin/bash

'''
This script works with Phenix version 1.18.
'''

qfit_usage() {
  echo >&2 "Usage:";
  echo >&2 "  $0 mapfile.ccp4 originalmodel.pdb [multiconformer_model2.pdb]";
  echo >&2 "";
  echo >&2 "mapfile.ccp4, originalmodel.pdb, multiconformer_model2.pdb MUST exist in this directory.";
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
org_model=$2
multiconf=${3:-multiconformer_model2.pdb}

echo "mapfile              : ${mapfile} $([[ -f ${mapfile} ]] || echo '[NOT FOUND]')";
echo "original model : ${org_model} $([[ -f ${org_model} ]] || echo '[NOT FOUND]')";
echo "qfit unrefined model : ${multiconf} $([[ -f ${multiconf} ]] || echo '[NOT FOUND]')";
echo "";
if [[ ! -f "${mapfile}" ]] || [[ ! -f "${multiconf}" ]] || [[ -f ${org_model} ]]; then
  qfit_usage;
fi

#CCP4 or MAP

pdb_name="${mapfile%.ccp4}"
echo $pdb_name
#__________________________________DETERMINE RESOLUTION AND (AN)ISOTROPIC REFINEMENT__________________________________
resolution=$(grep 'REMARK   3   RESOLUTION RANGE HIGH (ANGSTROMS) :' $org_model)
echo $resolution
res=`echo "${resolution}" | cut -d " " -f 14 | cut -c 1-5`
echo "Resolution: ${res}"

#__________________________________REMOVE DUPLICATE HET ATOMS__________________________________
remove_duplicates "${multiconf}"

#________________________________REMOVE TRAILING HYDROGENS___________________________________
phenix.pdbtools remove="element H" "${multiconf}.fixed"

#__________________________________GET CIF FILE__________________________________
phenix.ready_set hydrogens=false \
                 trust_residue_code_is_chemical_components_code=true \
                 pdb_file_name="${multiconf}.f_modified.pdb"

#__________________________________REFINEMENT WITHOUT HYDROGENS__________________________________
if [ -f "${multiconf}.f_modified.ligands.cif" ]; then
  echo "with ligand"
  phenix.real_space_refine "${multiconf}.f_modified.updated.pdb" \
                "${pdb_name}.ccp4" \
                "${multiconf}.f_modified.ligands.cif" \
                output.file_name_prefix="${pdb_name}2" \
                macro_cycles=5 \
                resolution=${res} \
                --overwrite
else
  phenix.real_space_refine "${multiconf}.f_modified.pdb" \
                "${pdb_name}.ccp4" \
                output.file_name_prefix="${pdb_name}2" \
                macro_cycles=5 \
                resolution=${res} \
                --overwrite
fi

#__________________________________ADD HYDROGENS__________________________________
# The first round of refinement regularizes geometry from qFit.
# Here we add H with phenix.reduce. Addition of H to the backbone is important
#   since it introduces planarity restraints to the peptide bond.
# This helps to prevent backbone conformers from being merged during
#   subsequent rounds of refinement.
cp "${pdb_name}2_real_space_refined.001.pdb" "${pdb_name}2.000.pdb"
phenix.reduce "${pdb_name}2.000.pdb" > "${pdb_name}2_real_space_refined.001.pdb"

#__________________________________REFINE UNTIL OCCUPANCIES CONVERGE__________________________________
# Write refinement parameters into parameters file
echo "output.prefix=${pdb_name}"                                  >> ${pdb_name}_occ_refine.params
echo "output.serial=3"                                            >> ${pdb_name}_occ_refine.params
echo "refinement.macro_cycles=5"                                  >> ${pdb_name}_occ_refine.params
echo "refinement.nqh_flips=False"                                  >> ${pdb_name}_occ_refine.params
echo "resolution=${res}"                                          >> ${pdb_name}_occ_refine.params

if [ -f "${multiconf}.f_modified.ligands.cif" ]; then
  echo "refinement.input.monomers.file_name='${multiconf}.f_modified.ligands.cif'" >> ${pdb_name}_occ_refine.params
fi


zeroes=50
i=1
while [ $zeroes -gt 1 ]; do
  cp "${pdb_name}2_real_space_refined.001.pdb" "${pdb_name}2_real_space_refined.001.$(printf '%03d' $i).pdb";
  ((i++));
  phenix.real_space_refine "${pdb_name}2_real_space_refined.001.pdb" "${pdb_name}.ccp4" ${pdb_name}_occ_refine.params

  zeroes=`redistribute_cull_low_occupancies -occ 0.09 "${pdb_name}3_real_space_refined.pdb" | tail -n 1`
  echo "Post refinement zeroes: ${zeroes}"

  if [ ! -f "${pdb_name}3_real_space_refined_norm.pdb" ]; then
     echo >&2 "Normalize occupancies did not work!";
     exit 1;
  else
     mv "${pdb_name}3_real_space_refined_norm.pdb" "${pdb_name}2_real_space_refined.001.pdb"
  fi
  
done

#__________________________________FINAL REFINEMENT__________________________________
mv "${pdb_name}2_real_space_refined.001.pdb" "${pdb_name}4_real_space_refined.001.pdb"

# Write refinement parameters into parameters file
echo "output.prefix=${pdb_name}"                 >> ${pdb_name}_final_refine.params
echo "output.serial=5"                           >> ${pdb_name}_final_refine.params
echo "refinement.macro_cycles=5"                 >> ${pdb_name}_final_refine.params
echo "refinement.nqh_flips=True"                 >> ${pdb_name}_final_refine.params
echo "resolution=${res}"                         >> ${pdb_name}_final_refine.params

if [ -f "${pdb_name}_002.ligands.cif" ]; then
  echo "refinement.input.monomers.file_name='${pdb_name}_002.ligands.cif'"  >> ${pdb_name}_final_refine.params
fi

phenix.real_space_refine "${pdb_name}4_real_space_refined.001.pdb" "${pdb_name}.ccp4" ${pdb_name}_final_refine.params

#__________________________________NAME FINAL FILES__________________________________
cp "${pdb_name}5_real_space_refined.pdb" "${pdb_name}_qFit.pdb"
cp "${pdb_name}.ccp4" "${pdb_name}_qFit.ccp4"
cp "${pdb_name}5_real_space_refined.log" "${pdb_name}_qFit.log"

