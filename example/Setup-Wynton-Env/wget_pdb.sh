#!/bin/bash
#Jessica Flowers
#09/14/2023

#this must be done before you submit to SGE since SGE cannot connect to the internet!
source /wynton/home/fraserlab/jessicaflowers/phenix/phenix-1.20.1-4487/phenix_env.sh
#________________________________________________INPUTS________________________________________________#
base_folder='/wynton/home/fraserlab/jessicaflowers/tp_qfit_ligands'

pdb_filelist=/wynton/home/fraserlab/jessicaflowers/tp_qfit_ligands/tp_ligands.txt
while read -r line; do
  PDB_ID=$line
  PDB=$(echo "$PDB_ID" | awk -F, '{print $1}')
  echo ${PDB}
  cd $base_folder
  if [ -d "/$PDB" ]; then
    echo "Folder exists."
  else
    mkdir $PDB
  fi
  #mkdir $PDB
  cd $PDB
  phenix.fetch_pdb $PDB
  phenix.fetch_pdb $PDB --mtz
done < $pdb_filelist

