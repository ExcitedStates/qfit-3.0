#!/bin/bash
#Stephanie Wankowicz
#07/31/2019

#this must be done before you submit to SGE since SGE cannot connect to the internet!
source /home/wankowicz/phenix-installer-1.16-3546-intel-linux-2.6-x86_64-centos6/phenix-1.16-3546/phenix_env.sh
#________________________________________________INPUTS________________________________________________#
#base_folder='/mnt/home1/wankowicz/190709_qfit/' #base folder (where you want to put folders/pdb files)
base_folder='/data/wankowicz/190719_PDBs/'
pdb_filelist=/mnt/home1/wankowicz/scripts/apo_190903_PDBs.txt  #=/mnt/home1/wankowicz/190709_qfit/PDBs_holo.txt #list of pdb files
while read -r line; do
  PDB=$line
  echo $PDB
  cd $base_folder
  sh /home/wankowicz/scripts/qfit_complete.sh $PDB $base_folder
done < $pdb_filelist
