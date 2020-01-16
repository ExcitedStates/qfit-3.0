#!/bin/bash
#___________________________SOURCE__________________________________
source /home/wankowicz/phenix-installer-1.16-3546-intel-linux-2.6-x86_64-centos6/phenix-1.16-3546/phenix_env.sh
export PHENIX_OVERWRITE_ALL=true
source /home/wankowicz/anaconda3/etc/profile.d/conda.sh
conda activate 'qfit2.1'
pdb_name=$1
echo $pdb_name


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


#GET CIF FILE
phenix.ready_set pdb_file_name=${pdb_name}.pdb

#DETERMINE FOBS v SIGOBS
if grep -F _refln.F_meas_au ${pdb_name}-sf.cif; then
       xray_data_labels="FOBS,SIGFOBS"
else
       xray_data_labels="IOBS,SIGIOBS"       
fi


#__________________________________GET CIF FILE__________________________________
phenix.ready_set pdb_file_name=multiconformer_model2.pdb

#_____________________________DETERMINE R FREE FLAGS______________________________
phenix.mtz.dump ${pdb_name}.mtz > ${pdb_name}_mtzdump.out

if grep -q FREE ${pdb_name}_mtzdump.out; then
  Rfree_flags=False
else
  Rfree_flags=True
fi

#__________________________________DETERMINE IF THERE ARE LIGANDS__________________________________
if [ -f "multiconformer_model2.pdb.cif" ]; then
  phenix.refine multiconformer_model2.pdb.fixed_modified.pdb \
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
  phenix.refine multiconformer_model2.pdb.fixed_modified.pdb \
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
  if [[ -e "multiconformer_model2.ligands.cif" ]]; then
        phenix.refine ${pdb_name}_002.pdb ${pdb_name}.mtz \
              multiconformer_model2.ligands.cif \
              output.prefix=${pdb_name} \
              output.serial=3 \
              strategy="*individual_sites *individual_adp *occupancies" \
              main.number_of_macro_cycles=5 \
              refinement.input.xray_data.labels=$xray_data_labels \
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
              multiconformer_model2.ligands.cif \
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
