#!/bin/bash

qfit_usage() {
  echo >&2 "Usage:";
  echo >&2 "  $0 mapfile.mtz qFit_ligand.pdb";
  echo >&2 "";
  echo >&2 "mapfile.mtz and qFit_ligand.pdb MUST exist in this directory.";
  echo >&2 "Outputs will be written to mapfile_qFitligand.{pdb|mtz|log}.";
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


# Assert required files exist
mapfile=$1
multiconf=${2:-qFit_ligand.pdb}
echo "mapfile              : ${mapfile} $([[ -f ${mapfile} ]] || echo '[NOT FOUND]')";
echo "qfit unrefined model : ${multiconf} $([[ -f ${multiconf} ]] || echo '[NOT FOUND]')";
echo "";
if [[ ! -f "${mapfile}" ]] || [[ ! -f "${multiconf}" ]]; then
  qfit_usage;
fi
pdb_name="${mapfile%.mtz}"

#__________________________________DETERMINE FOBS v IOBS v FP__________________________________
mtzmetadata=`phenix.mtz.dump "${pdb_name}.mtz"`
# List of Fo types we will check for
obstypes="FP FOBS F-obs I IOBS I-obs"

# Get amplitude fields
ampfields=`grep -E "amplitude|intensity" <<< "${mtzmetadata}"`
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

#__________________________________REMOVE DUPLICATE HET ATOMS__________________________________
remove_duplicates "${multiconf}"

#__________________________________GET CIF FILE__________________________________
phenix.ready_set hydrogens=false pdb_file_name="${multiconf}.fixed"

head "${multiconf}.ligands.cif"
#__________________________________REFINEMENT__________________________________
phenix.refine "${multiconf}.updated.pdb" \
                "${pdb_name}.mtz" \
                "${multiconf}.ligands.cif" \
                output.prefix="${pdb_name}" \
                output.serial=1 \
                strategy="*individual_sites *individual_adp *occupancies" \
                main.number_of_macro_cycles=5 \
                write_maps=False

#__________________________________NAME FINAL FILES__________________________________
cp "${pdb_name}_001.pdb" "${pdb_name}_qFitligand.pdb"
cp "${pdb_name}_001.mtz" "${pdb_name}_qFitligand.mtz"
cp "${pdb_name}_001.log" "${pdb_name}_qFitligand.log"
