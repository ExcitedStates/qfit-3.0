#!/bin/bash -f

qfit_usage() {
  echo >&2 "Usage:";
  echo >&2 "  $0 mapfile.mtz [multiconformer_model2.pdb] [qFit_occupancy.params] ligand.cif ";
  echo >&2 "";
  echo >&2 "mapfile.mtz, multiconformer_model2.pdb, and qFit_occupancy.params MUST exist in this directory.";
  echo >&2 "if the structure contains a ligand, please provide a ligand.cif";
  echo >&2 "Outputs will be written to mapfile_qFit.{pdb|mtz}.";
  exit 1;
}

#___________________________SOURCE__________________________________

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
ligand=$4
echo "mapfile              : ${mapfile} $([[ -f ${mapfile} ]] || echo '[NOT FOUND]')";
echo "qfit unrefined model : ${multiconf} $([[ -f ${multiconf} ]] || echo '[NOT FOUND]')";
echo "ligand info          : ${ligand} $([[ -f ${ligand} ]] || echo '[NOT FOUND]')";

echo "";
if [[ ! -f "${mapfile}" ]] || [[ ! -f "${multiconf}" ]]; then
  qfit_usage;
fi
pdb_name="${mapfile%.mtz}"

#define number of cycles
ncyc='10'

#__________________________________DETERMINE RESOLUTION AND (AN)ISOTROPIC REFINEMENT__________________________________
mtzmetadata=`echo "RUN" | mtzdump hklin "${pdb_name}.mtz"` 
line=`grep -n  "Resolution Range :" <<< "${mtzmetadata}"`
linenumber=`echo "${line}" | cut -d ":" -f 1`
resline=`echo "$(($linenumber+2))"`
resrange=`echo "${mtzmetadata}" | sed -n "${resline} p"`

res=`echo "${resrange}" | cut -d "-" -f 2 | cut -c 7-11` #not sure if this will always work, mtzmetadata spacing $
res1000=`echo $res | awk '{tot = $1*1000}{print tot }'`

if (( $res1000 < 1550 )); then
  adp='REFI BREF ANISotropic'
else
  adp='REFI BREF ISOTropic'
fi
#_____________________________DETERMINE R FREE FLAGS______________________________
gen_Rfree=true
rfreetypes="FREE R-free-flags FreeR_Flag"
for field in ${rfreetypes}; do
  if grep -F -q -w $field <<< "${mtzmetadata}"; then
    rfreeflag=`echo "${field}"`
    gen_Rfree=false;
    break
  fi
done
if [ "$gen_Rfree" = true ] ; then
  echo "Rfree Flags not found. Generating Rfree flags."
  freerflag HKLIN "${pdb_name}.mtz" HKLOUT "${pdb_name}_ready.mtz" << eor
END
eor
  rfreeflag="FreeR_flag"
else
  #Rename file if flags weren't added 
  cp "${pdb_name}.mtz" "${pdb_name}_ready.mtz"
fi  

#__________________________________DETERMINE If F(+)/I(+)__________________________________
#Refmac can find all input labels except F(+)/I(+). They must be labelled manually

#get column labels
line=`grep -n  "Column Labels :" <<< "${mtzmetadata}"`
linenumber=`echo "${line}" | cut -d ":" -f 1`
columnline=`echo "$(($linenumber+2))"`
columnlabels=`echo "${mtzmetadata}" | sed -n "${columnline} p"`
IFS=' ' read -r -a labels_array <<< "$columnlabels"

# List of Fo types we will check for
obstypes=("FP" "FOBS" "F-obs" "I" "IOBS" "I-obs")

label_flag=false
# Is amplitude an Fo?
for label in ${labels_array[*]}; do
  # Check field in obstypes
  if [[ " ${obstypes[*]} " =~ " ${label} " ]]; then
    # Check SIGFo is in the mtz too!
    if grep -F -q -w "SIG$label" <<< "${mtzmetadata}"; then
      label_flag=true; #if any of these labels are found, we don't need to feed Refmac any info
      break
    fi
  fi
done

Fplus_flag=false
Iplus_flag=false
for label in ${labels_array[*]}; do
  if [ ${label} = "F(+)" ]; then
    if grep -F -q -w "SIGF(+)" <<< "${mtzmetadata}"; then
      Fplus_flag=true
    fi
  fi
    if [ ${label} = "I(+)" ]; then
    if grep -F -q -w "SIGI(+)" <<< "${mtzmetadata}"; then
      Iplus_flag=true
    fi
  fi
done

#make labels
labels_in=""
if [ "$Fplus_flag" = true ] || [ "$Iplus_flag" = true ]; then #only needs to be added if relying on F+/I+
  labels_in="LABIN FREE=${rfreeflag}" 
fi

if [ "$Fplus_flag" = true ]; then
  labels_in="${labels_in} F+=F(+) SIGF+=SIGF(+)"
fi

if [ "$Iplus_flag" = true ]; then
  labels_in="${labels_in} I+=I(+) SIGI+=SIGI(+)"
fi

if [ "$label_flag" = true ]; then 
  labels_in="" #since we have labels Refmac can read automatically, it is best to give Refmac nothing
fi

#__________________________________REMOVE DUPLICATE HET ATOMS__________________________________
remove_duplicates "${multiconf}"

#__________________________________NORMALIZE OCCUPANCIES________________________________________
redistribute_cull_low_occupancies -occ 0.09 "${multiconf}.fixed"
mv -v "${multiconf}.f_norm.pdb" "${multiconf}.fixed"

#__________________________________First Refinement Preprocessing_______________________
#Add cryst data back to pdb
CRYST=`grep -e CRYST "${multiconf}"`
SCALE=`grep -e SCALE "${multiconf}"`
echo "${CRYST}
${SCALE}
$(cat ${multiconf}.fixed)" > "${multiconf%.pdb}_ready.pdb"

#make occupancy restraints file
create_restraints_file.py "${multiconf%.pdb}_ready.pdb"
python generateRefmacOccupancyParams.py qFit_occupancy.params
refinement_params=`cat Refmac_occupancy.txt`

#__________________________________First Refinement __________________________________
#Write refinement parameters into parameters file
echo "MONItor FEW" > ${pdb_name}_first_refine.params
echo "MAKE_restraints HYDRogens No" >> ${pdb_name}_first_refine.params
echo "DAMP 0.1 0 25" >> ${pdb_name}_first_refine.params
echo "NCYC=${ncyc}"  >> ${pdb_name}_first_refine.params
echo "${labels_in}"  >> ${pdb_name}_first_refine.params
echo "${refinement_params}" >> ${pdb_name}_first_refine.params


first_refi_log=`refmac5 \
HKLIN ${pdb_name}_ready.mtz \
HKLOUT ${pdb_name}_post_first_refi.mtz \
XYZIN ${multiconf%.pdb}_ready.pdb \
XYZOUT ${multiconf%.pdb}_post_first_refi.pdb << eor

MONItor FEW
MAKE_restraints HYDRogens No
DAMP 0.1 0 25

$refinement_params
$labels_in
NCYC=$ncyc

END
eor`

echo "${first_refi_log}"

#__________________________________Check for bad Rfree labels and rerun coord refine__________________

if grep "Warning ==> Switching off use of free R" <<< "${first_refi_log}"; then
  echo "Rfree Flags are unusable. Generating usable Rfree flags."
  freerflag HKLIN "${pdb_name}.mtz" HKLOUT "${pdb_name}_ready.mtz" << eor
END
eor
  rfreeflag="FreeR_flag"

#since Rfree label has been changed, input labels must be remade
  labels_in=""
  if [ "$Fplus_flag" = true ] || [ "$Iplus_flag" = true ]; then
    labels_in="LABIN FREE=${rfreeflag}" #only needs to be initialized if relying on F+/I+
  fi

  if [ "$Fplus_flag" = true ]; then
    labels_in="${labels_in} F+=F(+) SIGF+=SIGF(+)"
  fi

  if [ "$Iplus_flag" = true ]; then
    labels_in="${labels_in} I+=I(+) SIGI+=SIGI(+)"
  fi

  if [ "$label_flag" = true ]; then 
    labels_in="" #since we have labels Refmac can read automatically, it is best to give Refmac nothing
  fi

first_refi_log2=`refmac5 \
HKLIN ${pdb_name}_ready.mtz \
HKLOUT ${pdb_name}_post_first_refi.mtz \
XYZIN ${multiconf%.pdb}_ready.pdb \
XYZOUT ${multiconf%.pdb}_post_first_refi.pdb << eor

MONItor FEW
MAKE_restraints HYDRogens No
DAMP 0.1 0 25

$refinement_params
$labels_in
NCYC=$ncyc

END
eor`

echo "${first_refi_log2}"
fi

#__________________________________Occupancy Refinement Preprocessing_____________________________________
#cull occupancies < 0.1
zeroes=`redistribute_cull_low_occupancies -occ 0.09 "${multiconf%.pdb}_post_first_refi.pdb" | tail -n 1`
echo "zeroes:"
echo "${zeroes}"
if [ ! -f "${multiconf%.pdb}_post_first_refi_norm.pdb" ]; then
  echo >&2 "Normalize occupancies did not work!";
  exit 1;
else
  mv -v "${multiconf%.pdb}_post_first_refi_norm.pdb" "${multiconf%.pdb}_post_first_refi.pdb";
  echo "${CRYST}
${SCALE}
$(cat ${multiconf%.pdb}_post_first_refi.pdb)" > "${multiconf%.pdb}_post_first_refi.pdb"
fi

#make occupancy restraints file
create_restraints_file.py "${multiconf%.pdb}_post_first_refi.pdb"
python generateRefmacOccupancyParams.py qFit_occupancy.params
refinement_params=`cat Refmac_occupancy.txt`

#__________________________________REFINE UNTIL OCCUPANCIES CONVERGE__________________________________
echo "MONItor FEW" > ${pdb_name}_occ_refine.params
echo "DAMP 0.5 0.1 25" >> ${pdb_name}_occ_refine.params
echo "NCYC=${ncyc}"  >> ${pdb_name}_occ_refine.params
echo "${adp}" >> ${pdb_name}_occ_refine.params
echo "${refinement_params}" >> ${pdb_name}_occ_refine.params

zeroes=50
i=1
too_many_loops_flag=false

while [ $zeroes -gt 1 ]; do

  echo "qfit_final_refine_xray.sh:: Starting refinement round ${i}..."
  if [ -f "${ligand}" ]; then
  
    refmac5 \
HKLIN ${pdb_name}_post_first_refi.mtz \
HKLOUT ${pdb_name}_post_occupancy_refi.mtz \
XYZIN ${multiconf%.pdb}_post_first_refi.pdb \
XYZOUT ${multiconf%.pdb}_post_occupancy_refi.pdb \
LIBIN ${ligand} << eor

MONItor FEW
DAMP 0.5 0.1 25

NCYC=$ncyc
$adp
$refinement_params

END
eor
  else

      refmac5 \
HKLIN ${pdb_name}_post_first_refi.mtz \
HKLOUT ${pdb_name}_post_occupancy_refi.mtz \
XYZIN ${multiconf%.pdb}_post_first_refi.pdb \
XYZOUT ${multiconf%.pdb}_post_occupancy_refi.pdb << eor

MONItor FEW
DAMP 0.5 0.1 25

NCYC=$ncyc
$adp
$refinement_params

END
eor
  fi

  zeroes=`redistribute_cull_low_occupancies -occ 0.09 "${multiconf%.pdb}_post_occupancy_refi.pdb" | tail -n 1`
  echo "Post refinement zeroes: ${zeroes}"
  if [ ! -f "${multiconf%.pdb}_post_occupancy_refi_norm.pdb" ]; then
    echo >&2 "Normalize occupancies did not work!";
    exit 1;
  else
    mv -v "${multiconf%.pdb}_post_occupancy_refi_norm.pdb" "${multiconf%.pdb}_post_first_refi.pdb";
    echo "${CRYST}
${SCALE}
$(cat ${multiconf%.pdb}_post_first_refi.pdb)" > "${multiconf%.pdb}_post_first_refi.pdb"

    create_restraints_file.py "${multiconf%.pdb}_post_first_refi.pdb"
    python generateRefmacOccupancyParams.py qFit_occupancy.params
    refinement_params=`cat Refmac_occupancy.txt`
  fi

  if [ $i -ge 50 ]; then
    too_many_loops_flag=true
    echo "[WARNING] qfit_final_refine_xray.sh:: Aborting refinement loop after ${i} rounds.";
    break;
  fi

  ((i++));
done

#_______________________________Penultimate Refinement______________________________
echo "MONItor FEW" > ${pdb_name}_penultimate_refine.params
echo "NCYC=${ncyc}"  >> ${pdb_name}_penultimate_refine.params
echo "${adp}" >> ${pdb_name}_penultimate_refine.params
echo "${refinement_params}" >> ${pdb_name}_penultimate_refine.params

if [ -f "${ligand}" ]; then
  refmac5 \
HKLIN ${pdb_name}_post_first_refi.mtz \
HKLOUT ${pdb_name}_post_penultimate_refi.mtz \
XYZIN ${multiconf%.pdb}_post_first_refi.pdb \
XYZOUT ${multiconf%.pdb}_post_penultimate_refi.pdb \
LIBIN ${ligand} << eor

MONItor FEW

NCYC=$ncyc
$adp
$refinement_params

END
eor
else

  refmac5 \
HKLIN ${pdb_name}_post_first_refi.mtz \
HKLOUT ${pdb_name}_post_penultimate_refi.mtz \
XYZIN ${multiconf%.pdb}_post_first_refi.pdb \
XYZOUT ${multiconf%.pdb}_post_penultimate_refi.pdb << eor

MONItor FEW

NCYC=$ncyc
$adp
$refinement_params

END
eor
fi

zeroes=`redistribute_cull_low_occupancies -occ 0.09 "${multiconf%.pdb}_post_penultimate_refi.pdb" | tail -n 1`
  echo "Post refinement zeroes: ${zeroes}"
  if [ ! -f "${multiconf%.pdb}_post_penultimate_refi_norm.pdb" ]; then
    echo >&2 "Normalize occupancies did not work!";
    exit 1;
  else
    mv -v "${multiconf%.pdb}_post_penultimate_refi_norm.pdb" "${multiconf%.pdb}_post_penultimate_refi.pdb";
    echo "${CRYST}
${SCALE}
$(cat ${multiconf%.pdb}_post_penultimate_refi.pdb)" > "${multiconf%.pdb}_post_penultimate_refi.pdb"

    create_restraints_file.py "${multiconf%.pdb}_post_penultimate_refi.pdb"
    python generateRefmacOccupancyParams.py qFit_occupancy.params
    refinement_params=`cat Refmac_occupancy.txt`
  fi

#__________________________________FINAL REFINEMENT__________________________________

# Write refinement parameters into parameters file
echo "MONItor FEW" > ${pdb_name}_final_refine.params
echo "NCYC=${ncyc}"  >> ${pdb_name}_final_refine.params
echo "${adp}" >> ${pdb_name}_final_refine.params
echo "${refinement_params}" >> ${pdb_name}_final_refine.params

if [ -f "${ligand}" ]; then

  refmac5 \
HKLIN ${pdb_name}_post_penultimate_refi.mtz \
HKLOUT ${pdb_name}_post_final_refi.mtz \
XYZIN ${multiconf%.pdb}_post_penultimate_refi.pdb \
XYZOUT ${multiconf%.pdb}_post_final_refi.pdb \
LIBIN ${ligand} << eor

MONItor FEW

NCYC=$ncyc
$adp
$refinement_params

END
eor

  else

    refmac5 \
HKLIN ${pdb_name}_post_penultimate_refi.mtz \
HKLOUT ${pdb_name}_post_final_refi.mtz \
XYZIN ${multiconf%.pdb}_post_penultimate_refi.pdb \
XYZOUT ${multiconf%.pdb}_post_final_refi.pdb << eor

MONItor FEW

NCYC=$ncyc
$adp
$refinement_params

END
eor
  fi

#__________________________________NAME FINAL FILES__________________________________
cp "${multiconf%.pdb}_post_final_refi.pdb" "${pdb_name}_qFit.pdb"
cp "${pdb_name}_post_final_refi.mtz" "${pdb_name}_qFit.mtz"

#__________________________COMMENTARY FOR USER_______________________________________
if [ -f "${pdb_name}_qFit.pdb" ]; then 
   echo ""
   echo "[qfit_final_refine_xray] Refinement is complete."
   echo "                         Please be sure to INSPECT your refined structure, especially all new altconfs."
   echo "                         The output can be found at ${pdb_name}_qFit.(pdb|mtz)."
else
   echo "Refinement did not complete."
fi

if [ "${too_many_loops_flag}" = true ]; then
  echo "[qfit_final_refine_xray] WARNING: Refinement and low-occupancy rotamer culling was taking too long (${i} rounds).";
  echo "                         Some low-occupancy rotamers may remain. Please inspect your structure.";
fi
