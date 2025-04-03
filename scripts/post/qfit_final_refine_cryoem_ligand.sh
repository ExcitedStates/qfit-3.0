#!/bin/bash

'''
This script works with Phenix version 1.21.2.
'''

qfit_usage() {
  echo >&2 "Usage:";
  echo >&2 "  $0 mapfile.map [multiconformer_ligand_bound_with_protein.pdb] [multiconformer_ligand_only.pdb] ";
  echo >&2 "";
  echo >&2 "mapfile.map, multiconformer_ligand_bound_with_protein.pdb, and multiconformer_ligand_only.pdb MUST exist in this directory.";
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
res=$2
multiconf="multiconformer_ligand_bound_with_protein.pdb"
multiconf_lig="multiconformer_ligand_only.pdb"


# Display relevant file information
echo "mapfile              : ${mapfile} $([[ -f ${mapfile} ]] || echo '[NOT FOUND]')";
echo "qfit unrefined model : ${multiconf} $([[ -f ${multiconf} ]] || echo '[NOT FOUND]')";
echo "qfit unrefined ligand model : ${multiconf_lig} $([[ -f ${multiconf_lig} ]] || echo '[NOT FOUND]')";


if [ -z "${res}" ]; then
  echo >&2 "Resolution not specified.";
  exit 1;
fi
echo "resolution : ${res}";

echo "";
if [[ ! -f "${mapfile}" ]] || [[ ! -f "${multiconf}" ]]; then
  qfit_usage;
fi
pdb_name="${mapfile%.map}"

echo $pdb_name

#__________________________________REMOVE DUPLICATE HET ATOMS__________________________________
remove_duplicates "${multiconf}" 

#________________________________REMOVE TRAILING HYDROGENS___________________________________
phenix.pdbtools remove="element H" "${multiconf}.fixed"

#__________________________________GET CIF FILE__________________________________
echo "Getting the cif file with ready_set/elbow" 
phenix.ready_set hydrogens=false \
                 trust_residue_code_is_chemical_components_code=true \
                 pdb_file_name="${multiconf}.f_modified.pdb"

# If ready_set doesn't generate a ligand cif file, use elbow
if [ ! -f "${multiconf}.f_modified.ligands.cif" ]; then
  phenix.elbow multiconformer_ligand_only.pdb --output ${multiconf}.f_modified.ligands
fi

if [ ! -f "${multiconf}.f_modified.ligands.cif" ]; then
  echo "Ligand CIF generation failed"
fi

cp ${multiconf}.f_modified.pdb ${pdb_name}2_real_space_refined.001.pdb ## error thrown if uses elbow? 
#__________________________________REAL SPACE REFINEMENT WITH PHENIX__________________________________
phenix.real_space_refine "${pdb_name}2_real_space_refined.001.pdb" \
                  "${pdb_name}.map" \
                  "${multiconf}.f_modified.ligands.cif" \
                  output.prefix="${pdb_name}3" \
                  macro_cycles=5 \
                  resolution=${res} \
                  --overwrite


zeroes=`redistribute_cull_low_occupancies -occ 0.09 "${pdb_name}3_real_space_refined_000.pdb" | tail -n 1`
echo "Post refinement zeroes: ${zeroes}"

if [ ! -f "${pdb_name}3_real_space_refined_000_norm.pdb" ]; then
    echo >&2 "Normalize occupancies did not work!";
    exit 1;
else
    mv "${pdb_name}3_real_space_refined_000_norm.pdb" "${pdb_name}2_real_space_refined.001.pdb"
fi



#__________________________________NAME FINAL FILES__________________________________
cp "${pdb_name}2_real_space_refined.001.pdb" "${pdb_name}_qFit_ligand.pdb"
cp "${pdb_name}.ccp4" "${pdb_name}_qFit_ligand.ccp4"
cp "${pdb_name}2_real_space_refined.001.log" "${pdb_name}_qFit_ligand.log"
