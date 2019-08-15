#!/bin/bash

source /home/wankowicz/phenix-installer-1.16-3546-intel-linux-2.6-x86_64-centos6/phenix-1.16-3546/phenix_env.sh
#conda activate qfit
#OR MAKE SURE YOUR QFIT ENV IS ACTIVE

pdb_name=$1
echo $pdb_name

#REMOVE DUPLICATE HET ATOMS
remove_duplicates multiconformer_model2.pdb

# DETERMINE RESOLUTION AND (AN)ISOTROPIC REFINEMENT
resrange= phenix.mtz.dump ${pdb_name}.mtz | grep "Resolution range:"

echo $resrange

res=`echo $resrange[4] | cut -c 1-5`
res1000=`echo $res | awk '{tot = $1*1000}{print tot }'`

if ($res1000<1550) then
  adp='adp.individual.anisotropic="not (water or element H)"'
else
  adp='adp.individual.isotropic=all'
fi

#GET CIF FILE
phenix.ready_set pdb_file_name=${pdb_name}.pdb

#DETERMINE FOBS v SIGOBS
if grep -F _refln.F_meas_au ${pdb_name}-sf.cif; then
       xray_data_labels="FOBS,SIGFOBS"
else
       xray_data_labels="IOBS,SIGIOBS"       
fi

#DETERMINE IF THERE ARE LIGANDS
if [[ -e "${pdb_name}.ligands.cif" ]]; then
  phenix.refine multiconformer_model2.pdb.fixed \
              ${pdb_name}.mtz \
              ${pdb_name}.ligands.cif \
              strategy=individual_sites \
              output.prefix=${pdb_name} \
              output.serial=2 \
              main.number_of_macro_cycles=5 \
              refinement.input.xray_data.r_free_flags.generate=True \
              refinement.input.xray_data.labels=$xray_data_labels \
              write_maps=false --overwrite
else
  phenix.refine multiconformer_model2.pdb.fixed \
              ${pdb_name}.mtz \
              strategy=individual_sites \
              output.prefix=${pdb_name} \
              output.serial=2 \
              main.number_of_macro_cycles=5 \
              refinement.input.xray_data.r_free_flags.generate=True \
              refinement.input.xray_data.labels=$xray_data_labels \
              write_maps=false --overwrite
fi


# REFINE UNTIL OCCUPANCIES CONVERGE
zeroes=50

while [ $zeroes > 10 ]
do
  if [[ -e "${pdb_name}.ligands.cif" ]]; then
  phenix.refine ${pdb_name}_002.pdb ${pdb_name}.mtz \
              ${pdb_name}.ligands.cif \
              output.prefix=${pdb_name} \
              output.serial=3 \
              strategy="*individual_sites *individual_adp *occupancies" \
              main.number_of_macro_cycles=5 \
              refinement.input.xray_data.labels=$xray_data_labels\
              write_maps=false --overwrite
  else
  phenix.refine ${pdb_name}_002.pdb ${pdb_name}.mtz \
              output.prefix=${pdb_name} \
              output.serial=3 \
              strategy="*individual_sites *individual_adp *occupancies" \
              main.number_of_macro_cycles=5 \
              refinement.input.xray_data.labels=$xray_data_labels \
              write_maps=false --overwrite
  fi

zeroes=`normalize_occupancies -occ 0.09 ${pdb_name}_003.pdb`
   #z=`awk '{ if(ubstr($0,1,6)=="CRYST1"||(substr($0,1,4)=="ATOM"||substr($0,1,6)=="HETATM")&&substr($0,57,4)+0<0.09) print $0}' ${pdb_name}_003.pdb | wc`
   #zeroes=$z[1]

normalize_occupancies -occ 0.09 ${pdb_name}_003.pdb
mv ${pdb_name}_003_norm.pdb ${pdb_name}_002.pdb
done

# ADD HYDROGENS
phenix.reduce ${pdb_name}_002.pdb > ${pdb_name}_004.pdb

# FINAL REFINEMENT
if [[ -e "${pdb_name}.ligands.cif" ]]; then
phenix.refine ${pdb_name}_004.pdb ${pdb_name}.mtz \
              ${pdb_name}.ligands.cif \
              ${pdb_name}_004.pdb \
              "$adp" \
              output.prefix=${pdb_name} \
              output.serial=5 \
              strategy="*individual_sites *individual_adp *occupancies" \
              main.number_of_macro_cycles=5 \
              refinement.input.xray_data.labels=$xray_data_labels \
              write_maps=false \
              --overwrite
else
  phenix.refine ${pdb_name}_004.pdb ${pdb_name}.mtz \
              ${pdb_name}.ligands.cif \
              ${pdb_name}_004.pdb \
              "$adp" \
              output.prefix=${pdb_name} \
              output.serial=5 \
              strategy="*individual_sites *individual_adp *occupancies" \
              main.number_of_macro_cycles=5 \
              refinement.input.xray_data.labels=$xray_data_labels \
              write_maps=false \
              --overwrite
fi


cp ${pdb_name}_005.pdb ${pdb_name}_qFit.pdb
cp ${pdb_name}_005.mtz ${pdb_name}_qFit.mtz
cp ${pdb_name}_005.log ${pdb_name}_qFit.log
