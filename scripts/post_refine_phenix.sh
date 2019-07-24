#!/bin/bash

source /home/wankowicz/phenix-installer-1.16-3546-intel-linux-2.6-x86_64-centos6/phenix-1.16-3546/phenix_env.sh

#pdb=$1
pdb_name=$1
F=$2
SF=$3
echo $pdb_name
#set elbow = $5


#bspdb='%s\n' "${pdb//.pdb/}" #`basename $pdb .pdb`
#echo $bspdb
#bsmtz='%s\n' "${mtz//.mtz/}" #`basename $mtz .mtz`

#echo $bspdb

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
phenix.ready_set pdb_file_name=multiconformer_model2.pdb

#CHANGE TO IOBS/SIGOBS
#phenix.reflection_file_converter


# REFINE COORDINATES ONLY
phenix.refine multiconformer_model2.pdb \
              ${pdb_name}.mtz \
              multiconformer_model2.ligands.cif \
              strategy=individual_sites \
              output.prefix=${pdb_name} \
              output.serial=2 \
              main.number_of_macro_cycles=5 \
              #refinement.input.xray_data.labels=$F,$SF\
              write_maps=false --overwrite

# REFINE UNTIL OCCUPANCIES CONVERGE
zeroes=50

while [ $zeroes > 10 ]
do
phenix.refine ${pdb_name}_002.pdb ${pdb_name}.mtz \
              multiconformer_model2.ligands.cif \
              output.prefix=${pdb_name} \
              output.serial=3 \
              strategy="*individual_sites *individual_adp *occupancies" \
              main.number_of_macro_cycles=5 \
              #refinement.input.xray_data.labels=$F,$SF\
              write_maps=false --overwrite

zeroes=`normalize_occupancies -occ 0.09 ${bspdb}_003.pdb`
   #z=`awk '{ if(ubstr($0,1,6)=="CRYST1"||(substr($0,1,4)=="ATOM"||substr($0,1,6)=="HETATM")&&substr($0,57,4)+0<0.09) print $0}' ${pdb_name}_003.pdb | wc`
   #zeroes=$z[1]

normalize_occupancies -occ 0.09 ${pdb_name}_003.pdb
mv ${pdb_name}_003_norm.pdb ${pdb_name}_002.pdb
done

# ADD HYDROGENS
phenix.reduce ${pdb_name}_002.pdb > ${pdb_name}_004.pdb

# FINAL REFINEMENT
phenix.refine ${pdb_name}_004.pdb ${pdb_name}.mtz \
              multiconformer_model2.ligands.cif \
              ${pdb_name}_004.pdb \
              "$adp" \
              output.prefix=${pdb_name} \
              output.serial=5 \
              strategy="*individual_sites *individual_adp *occupancies" \
              main.number_of_macro_cycles=5 \
              #refinement.input.xray_data.labels=$F,$SF\
              multiconformer_model2.ligands.cif \
              write_maps=false \
              --overwrite

cp ${pdb_name}_005.pdb ${pdb_name}_qFit.pdb
cp ${pdb_name}_005.mtz ${pdb_name}_qFit.mtz
cp ${pdb_name}_005.log ${pdb_name}_qFit.log
