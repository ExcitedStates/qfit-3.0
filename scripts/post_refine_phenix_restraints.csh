#!/bin/tcsh

source /home/sw/rhel6/x86_64/phenix/phenix-1.14-3260/phenix_env.csh

set pdb = $1
set mtz = $2
set F = $3
set SF = $4
set elbow = $5
# set restraints = $6
set bspdb = `basename $pdb .pdb`
set bsres = `basename $6 .eff`

fix_restraints $1 $6 > ${bsres}.out

set restraints = ${bsres}.out

# DETERMINE RESOLUTION AND (AN)ISOTROPIC REFINEMENT
set resrange = `phenix.mtz.dump $mtz | grep "Resolution range:"`
set res = `echo $resrange[4] | cut -c 1-5`
set res1000 = `echo $res | awk '{tot = $1*1000}{print tot }'`

if ( $res1000 < 1550 ) then
  set adp = 'adp.individual.anisotropic="not (water or element H)"'
else
  set adp = 'adp.individual.isotropic=all'
endif

# REFINE COORDINATES ONLY
phenix.refine $mtz\
             $pdb\
             strategy=individual_sites\
             output.prefix=${bspdb}\
             output.serial=2\
             main.number_of_macro_cycles=5\
             refinement.input.xray_data.labels=$F,$SF\
             $elbow\
             $restraints\
             write_maps=false\
             --overwrite > phenix.log

# REFINE UNTIL OCCUPANCIES CONVERGE
set zeroes = 50

while ($zeroes > 1 )
   phenix.refine $mtz\
              ${bspdb}_002.pdb\
              output.prefix=${bspdb}\
              output.serial=3\
              strategy="*individual_sites *individual_adp *occupancies"\
              main.number_of_macro_cycles=5\
              refinement.input.xray_data.labels=$F,$SF\
              $elbow\
              $restraints\
              write_maps=false\
              --overwrite >> phenix.log

   set zeroes = `normalize_occupancies -occ 0.09 ${bspdb}_003.pdb`
   mv ${bspdb}_003_norm.pdb ${bspdb}_002.pdb
   fix_restraints ${bspdb}_002.pdb $6 > $restraints;
end

# ADD HYDROGENS
phenix.reduce ${bspdb}_002.pdb > ${bspdb}_004.pdb

# FINAL REFINEMENT
phenix.refine $mtz\
              ${bspdb}_004.pdb\
              "$adp"\
              output.prefix=${bspdb}\
              output.serial=5\
              strategy="*individual_sites *individual_adp *occupancies"\
              main.number_of_macro_cycles=5\
              refinement.input.xray_data.labels=$F,$SF\
              $elbow\
              $restraints\
              write_maps=false\
              --overwrite >> phenix.log

cp multiconformer_model2_005.pdb qFit.pdb
cp multiconformer_model2_005.mtz qFit.mtz
cp multiconformer_model2_005.log qFit.log

exit
