#!/bin/bash
#$ -l h_vmem=40G
#$ -l mem_free=40G
#$ -t 1-135
#$ -l h_rt=48:00:00
#$ -pe smp 1
#$ -R yes
#$ -V

#________________________________________________INPUTS________________________________________________#
PDB_file=/wynton/home/fraserlab/jessicaflowers/tp_qfit_ligands/tp_ligands.txt
base_dir='/wynton/home/fraserlab/jessicaflowers/tp_qfit_ligands'

category='qfit_run_1'

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

#________________________________________________SET PATHS________________________________________________#
export PATH="/wynton/home/fraserlab/jessicaflowers/miniconda3/bin:$PATH"
source activate qfit_ligand
which python
export PHENIX_OVERWRITE_ALL=true
source /wynton/home/fraserlab/jessicaflowers/phenix/phenix-1.20.1-4487/phenix_env.sh

#________________________________________________RUN QFIT________________________________________________#
PDB=$(cat $PDB_file | head -n $SGE_TASK_ID | tail -n 1)

# Extract and format the required information
pdb_id=$(echo "$PDB" | awk -F, '{print $1}')
smiles=$(echo "$PDB" | awk -F, '{print $4}')
chain_res=$(echo "$PDB" | awk -F, '{print $(NF-3) "," $(NF-2)}')

echo "smiles = ${smiles}"
echo "chain_res = ${chain_res}"

# Construct the command
command="qfit_ligand composite_omit_map.mtz -sm '${smiles}' -l 2FOFCWT,PH2FOFCWT ${pdb_id}_001.pdb ${chain_res}"

# Go to folder containing your data
cd ${base_dir}/${pdb_id}
echo "Working on pdb: ${pdb_id}"

# Make catagory folder
mkdir ${category}
cd ${category}

cp ../${pdb_id}.mtz .
cp ../composite_omit_map.mtz .
cp ../${pdb_id}_001.pdb .
cp ../${pdb_id}.pdb .

eval $command
qfit_final_refine_ligand.sh ${pdb_id}.mtz
