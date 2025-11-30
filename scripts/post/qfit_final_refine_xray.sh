#!/bin/bash
# This script works with Phenix version 2.0.

qfit_usage() {
  echo >&2 "Usage:";
  echo >&2 "  $0 mapfile.mtz [multiconformer_model2.pdb] [qFit_occupancy.params]";
  echo >&2 "";
  echo >&2 "mapfile.mtz, multiconformer_model2.pdb, and qFit_occupancy.params MUST exist in this directory.";
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
obstypes=("FP" "FOBS" "F-obs" "I" "IOBS" "I-obs" "F(+)" "I(+)" "F-obs-filtered(+)")

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
  echo "data_manager.miller_array.labels.name=$xray_data_labels" > ${pdb_name}_refine.params
fi

#_____________________________DETERMINE R FREE FLAGS______________________________
gen_Rfree=True
rfreetypes="FREE R-free-flags"
for field in ${rfreetypes}; do
  if grep -F -q -w $field <<< "${mtzmetadata}"; then
    gen_Rfree=False;
    echo "Rfree column: ${rfield}";
    echo "miller_array.labels.name=${rfield}" >> ${pdb_name}_refine.params
    break
  fi
done
echo "xray_data.r_free_flags.generate=${gen_Rfree}" >> ${pdb_name}_refine.params

#__________________________________REMOVE DUPLICATE HET ATOMS__________________________________
remove_duplicates "${multiconf}"

#__________________________________NORMALIZE OCCUPANCIES________________________________________
redistribute_cull_low_occupancies -occ 0.09 "${multiconf}.fixed"
mv -v "${multiconf}.f_norm.pdb" "${multiconf}.fixed"

#________________________________REMOVE TRAILING HYDROGENS___________________________________
phenix.pdbtools remove="element H" "${multiconf}.fixed"

#__________________________________GET CIF FILE__________________________________
phenix.ready_set hydrogens=false \
                 trust_residue_code_is_chemical_components_code=true \
                 pdb_file_name="${multiconf}.f_modified.pdb"

phenix.elbow ${multiconf}.f_modified.pdb --do_all

# If there are no unknown ligands, ready_set doesn't output a file. We have to do it.
if [ ! -f "${multiconf}.f_modified.updated.pdb" ]; then
  cp -v "${multiconf}.f_modified.pdb" "${multiconf}.f_modified.updated.pdb";
fi
if [ -f "elbow.${multiconf}_pdb.all.cif" ]; then
  echo "refinement.input.monomers.file_name='elbow.${multiconf}_pdb.all.cif'" >> ${pdb_name}_refine.params
fi

#__________________________________COORDINATE REFINEMENT ONLY__________________________________
# Write refinement parameters into parameters file (note, this is for record keeping, but is not accepted by Phenix)
echo "refinement.refine.strategy=*individual_sites"  >> ${pdb_name}_refine.params
echo "output.prefix=${pdb_name}"          >> ${pdb_name}_refine.params
echo "output.serial=2"                    >> ${pdb_name}_refine.params
echo "refinement.main.number_of_macro_cycles=5"      >> ${pdb_name}_refine.params
echo "refinement.main.nqh_flips=False"               >> ${pdb_name}_refine.params
echo "refinement.output.write_maps=False"            >> ${pdb_name}_refine.params

phenix.refine  "${multiconf}.f_modified.updated.pdb" \
               "${pdb_name}.mtz" \
	           "refinement.input.monomers.file_name=elbow.multiconformer_model2_pdb_f_modified_pdb.all.cif" \
               "refine.strategy=*individual_sites" \
               "output.prefix=${pdb_name}" \
               "output.serial=2" \
               "refinement.main.number_of_macro_cycles=5" \
               "refinement.main.nqh_flips=False" \
               "refinement.output.write_maps=False" \
               "data_manager.miller_array.labels.name=${xray_data_labels}" \
               "xray_data.r_free_flags.generate=True" \
               --overwrite

create_restraints_file.py "${pdb_name}_002.pdb"

#__________________________________REFINE UNTIL OCCUPANCIES CONVERGE__________________________________
# Write refinement parameters into parameters file (note, this is for record keeping, but is not accepted by Phenix)
echo "refine.strategy=*individual_sites *individual_adp *occupancies"  > ${pdb_name}_occ_refine.params
echo "output.prefix=${pdb_name}"                                      >> ${pdb_name}_occ_refine.params
echo "output.serial=3"                                                >> ${pdb_name}_occ_refine.params
echo "refinement.main.number_of_macro_cycles=5"                       >> ${pdb_name}_occ_refine.params
echo "refinement.main.nqh_flips=False"                                >> ${pdb_name}_occ_refine.params
echo "refinement.refine.${adp}"                                       >> ${pdb_name}_occ_refine.params
echo "refinement.output.write_maps=False"                             >> ${pdb_name}_occ_refine.params

if [ -f "elbow.${multiconf}_pdb.all.cif" ]; then
  echo "refinement.input.monomers.file_name='elbow.${multiconf}_pdb.all.cif'" >> ${pdb_name}_occ_refine.params
fi

zeroes=50
i=1
too_many_loops_flag=false
while [ $zeroes -gt 1 ]; do
  echo "qfit_final_refine_xray.sh:: Starting refinement round ${i}..."
  if [ -f "elbow.multiconformer_model2_pdb_f_modified_pdb.all.cif" ]; then
      phenix.refine "${pdb_name}_002.pdb" \
                    "${pdb_name}_002.mtz" \
		            "refinement.input.monomers.file_name=elbow.multiconformer_model2_pdb_f_modified_pdb.all.cif" \
                    "refine.strategy=*individual_sites *individual_adp *occupancies" \
                    "output.prefix=${pdb_name}" \
                    "output.serial=3" \
                    "refinement.main.number_of_macro_cycles=5" \
                    "refinement.main.nqh_flips=False" \
                    "refinement.refine.${adp}" \
                    qFit_occupancy.params \
                    --overwrite
    else
        phenix.refine "${pdb_name}_002.pdb" \
                    "${pdb_name}_002.mtz" \
                    "refine.strategy=*individual_sites *individual_adp *occupancies" \
                    "output.prefix=${pdb_name}" \
                    "output.serial=3" \
                    "refinement.main.number_of_macro_cycles=5" \
                    "refinement.main.nqh_flips=False" \
                    "refinement.refine.${adp}" \
                    qFit_occupancy.params \
                    --overwrite
    fi

  zeroes=`redistribute_cull_low_occupancies -occ 0.09 "${pdb_name}_003.pdb" | tail -n 1`
  echo "Post refinement zeroes: ${zeroes}"
  if [ ! -f "${pdb_name}_003_norm.pdb" ]; then
    echo >&2 "Normalize occupancies did not work!";
    exit 1;
  else
    mv -v "${pdb_name}_003_norm.pdb" "${pdb_name}_002.pdb";
    python /panfs/accrepfs.vampire/data/wankowicz_lab/stephanie/qfit-3.0/scripts/post/create_restraints_file.py "${pdb_name}_002.pdb"
  fi

  if [ $i -ge 50 ]; then
    too_many_loops_flag=true
    echo "[WARNING] qfit_final_refine_xray.sh:: Aborting refinement loop after ${i} rounds.";
    break;
  fi

  ((i++));
done

#________________________________CLEAN DUPLICATE___________________________________
remove_duplicates  ${pdb_name}_002.pdb
mv ${pdb_name}_002_cleaned.pdb ${pdb_name}_002.pdb

#__________________________________ADD HYDROGENS__________________________________
# The first round of refinement regularizes geometry from qFit.
# Here we add H with phenix.ready_set. Addition of H to the backbone is important
#   since it introduces planarity restraints to the peptide bond.
# We will also create a cif file for any ligands in the structure at this point.
phenix.ready_set hydrogens=true pdb_file_name="${pdb_name}_002.pdb"
mv "${pdb_name}_002.updated.pdb" "${pdb_name}_002.pdb"

#__________________________________FINAL REFINEMENT__________________________________
cp -v "${pdb_name}_002.pdb" "${pdb_name}_004.pdb"
phenix.elbow ${pdb_name}_004.pdb --do_all

# Write refinement parameters into parameters file (note, this is for record keeping, but is not accepted by Phenix)
echo "refine.strategy=*individual_sites *individual_adp *occupancies"  >> ${pdb_name}_final_refine.params
echo "output.prefix=${pdb_name}"      >> ${pdb_name}_final_refine.params
echo "output.serial=5"                >> ${pdb_name}_final_refine.params
echo "include_altlocs=True"           >> ${pdb_name}_final_refine.params
echo "ordered_solvent.dist_min = 2.0" >> ${pdb_name}_final_refine.params
echo "refinement.main.number_of_macro_cycles=5"  >> ${pdb_name}_final_refine.params
echo "refinement.main.nqh_flips=True"            >> ${pdb_name}_final_refine.params
echo "refinement.refine.${adp}"                  >> ${pdb_name}_final_refine.params
echo "refinement.output.write_maps=False"        >> ${pdb_name}_final_refine.params
echo "refinement.hydrogens.refine=riding"        >> ${pdb_name}_final_refine.params
echo "refinement.main.ordered_solvent=True"      >> ${pdb_name}_final_refine.params
echo "refinement.target_weights.optimize_xyz_weight=true"  >> ${pdb_name}_final_refine.params
echo "refinement.target_weights.optimize_adp_weight=true"  >> ${pdb_name}_final_refine.params
echo "refinement.main.ordered_solvent=True" >> ${pdb_name}_final_refine.params
echo "ordered_solvent.mode=every_macro_cycle" >> ${pdb_name}_final_refine.params
echo "include_altlocs=True" >> ${pdb_name}_final_refine.params

if [ -f "elbow.${pdb_name}_004_pdb.all.cif" ]; then
  echo "refinement.input.monomers.file_name='elbow.${pdb_name}_004_pdb.all.cif'"  >> ${pdb_name}_final_refine.params
fi

if [ -f "elbow.${pdb_name}_004_pdb.all.cif" ]; then
      phenix.refine "${pdb_name}_002.pdb" \
            "${pdb_name}_002.mtz" \
	        "refinement.input.monomers.file_name=elbow.${pdb_name}_004_pdb.all.cif" \
            "refine.strategy=*individual_sites *individual_adp *occupancies" \
            "output.prefix=${pdb_name}" \
            "output.serial=5" \
            "refinement.main.number_of_macro_cycles=5" \
            "refinement.main.nqh_flips=True" \
            "refinement.refine.${adp}" \
            "refinement.hydrogens.refine=riding" \
            "refinement.main.ordered_solvent=True" \
            "ordered_solvent.dist_min = 2.0" \
            "ordered_solvent.mode=every_macro_cycle" \
            "include_altlocs=True" \
            "refinement.target_weights.optimize_xyz_weight=true" \
            "refinement.target_weights.optimize_adp_weight=true" \
            qFit_occupancy.params \
            --overwrite
 else
      phenix.refine "${pdb_name}_002.pdb" \
            "${pdb_name}_002.mtz" \
            "refine.strategy=*individual_sites *individual_adp *occupancies" \
            "output.prefix=${pdb_name}" \
            "output.serial=5" \
            "refinement.main.number_of_macro_cycles=5" \
            "refinement.main.nqh_flips=True" \
            "refinement.refine.${adp}" \
            "refinement.hydrogens.refine=riding" \
            "refinement.main.ordered_solvent=True" \
            "ordered_solvent.dist_min = 2.0" \
            "ordered_solvent.mode=every_macro_cycle" \
            "include_altlocs=True" \
            "refinement.target_weights.optimize_xyz_weight=true" \
            "refinement.target_weights.optimize_adp_weight=true" \
            qFit_occupancy.params \
            --overwrite
fi


#________________________________CHECK FOR REDUCE ERRORS______________________________
if [ -f "reduce_failure.pdb" ]; then
  echo "refinement.refine.strategy=*individual_sites *individual_adp *occupancies"  > ${pdb_name}_final_refine_noreduce.params
  echo "output.prefix=${pdb_name}"      >> ${pdb_name}_final_refine_noreduce.params
  echo "output.serial=5"                >> ${pdb_name}_final_refine_noreduce.params
  echo "include_altlocs=True"           >> ${pdb_name}_final_refine_noreduce.params
  echo "ordered_solvent.dist_min = 2.0" >> ${pdb_name}_final_refine_noreduce.params
  echo "refinement.main.number_of_macro_cycles=5"  >> ${pdb_name}_final_refine_noreduce.params
  echo "refinement.main.nqh_flips=False"           >> ${pdb_name}_final_refine_noreduce.params
  echo "refinement.refine.${adp}"                  >> ${pdb_name}_final_refine_noreduce.params
  echo "refinement.output.write_maps=False"        >> ${pdb_name}_final_refine_noreduce.params
  echo "refinement.hydrogens.refine=riding"        >> ${pdb_name}_final_refine_noreduce.params
  echo "refinement.main.ordered_solvent=True"      >> ${pdb_name}_final_refine_noreduce.params
  echo "refinement.target_weights.optimize_xyz_weight=true"  >> ${pdb_name}_final_refine_noreduce.params
  echo "refinement.target_weights.optimize_adp_weight=true"  >> ${pdb_name}_final_refine_noreduce.params

  if [ -f "elbow.${pdb_name}_004_pdb.all.cif" ]; then
      phenix.refine "${pdb_name}_002.pdb" \
            "${pdb_name}_002.mtz" \
	        "refinement.input.monomers.file_name=elbow.${pdb_name}_004_pdb.all.cif"
            "refine.strategy=*individual_sites *individual_adp *occupancies" \
            "output.prefix=${pdb_name}" \
            "output.serial=5" \
            "refinement.main.number_of_macro_cycles=5" \
            "refinement.main.nqh_flips=False" \
            "refinement.refine.${adp}" \
            "refinement.hydrogens.refine=riding" \
            "refinement.main.ordered_solvent=True" \
            "ordered_solvent.dist_min = 2.0" \ 
            "ordered_solvent.mode=every_macro_cycle" \
            "include_altlocs=True" \
            "refinement.target_weights.optimize_xyz_weight=true" \
            "refinement.target_weights.optimize_adp_weight=true" \
            qFit_occupancy.params \
            --overwrite
 else
      phenix.refine "${pdb_name}_002.pdb" \
            "${pdb_name}_002.mtz" \
            "refine.strategy=*individual_sites *individual_adp *occupancies" \
            "output.prefix=${pdb_name}" \
            "output.serial=5" \
            "refinement.main.number_of_macro_cycles=5" \
            "refinement.main.nqh_flips=False" \
            "refinement.refine.${adp}" \
            "refinement.hydrogens.refine=riding" \
            "refinement.main.ordered_solvent=True" \
            "ordered_solvent.mode=every_macro_cycle" \
            "include_altlocs=True" \
            "ordered_solvent.dist_min = 2.0" \
            "refinement.target_weights.optimize_xyz_weight=true" \
            "refinement.target_weights.optimize_adp_weight=true" \
            qFit_occupancy.params \
            --overwrite
fi
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

if [ "${too_many_loops_flag}" = true ]; then
  echo "[qfit_final_refine_xray] WARNING: Refinement and low-occupancy rotamer culling was taking too long (${i} rounds).";
  echo "                         Some low-occupancy rotamers may remain. Please inspect your structure.";
fi

if [ -f "reduce_failure.pdb" ]; then
  echo "Refinement was run without checking for flips in NQH residues due to memory constraints. Please inspect your structure."
fi

