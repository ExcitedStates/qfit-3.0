#!/bin/bash
#$ -l h_vmem=40G
#$ -l mem_free=40G
#$ -t 1-length of text_file_of_pdbs.txt
#$ -l h_rt=48:00:00
#$ -pe smp 1
#$ -R yes
#$ -V

#________________________________________________INPUTS________________________________________________#
PDB_file=/path/to/dir/containing/pdb/files/text_file_of_pdbs.txt
base_dir='/path/to/dir/containing/pdb/files'

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

#________________________________________________SET PATHS________________________________________________#
export PATH="/wynton/home/fraserlab/jessicaflowers/miniconda3/bin:$PATH"
source activate qfit-conda-environment
which python
source /wynton/home/fraserlab/jessicaflowers/phenix/phenix-1.20.1-4487/phenix_env.sh

#________________________________________________PROCESS INPUT________________________________________________#
PDB_LINE=$(sed -n "${SGE_TASK_ID}p" ${PDB_file})

# Extract fields
pdb_id=$(echo "$PDB_LINE" | awk -F, '{print $1}')
chain_id=$(echo "$PDB_LINE" | awk -F, '{print $2}')
res_num=$(echo "$PDB_LINE" | awk -F, '{print $3}')
smiles=$(echo "$PDB_LINE" | awk -F, '{print $4}')
lig_name=$(echo "$PDB_LINE" | awk -F, '{print $5}')

echo "Processing PDB: ${pdb_id}, Chain: ${chain_id}, Residue: ${res_num}, Ligand: ${lig_name}"

# cd into the pdb folder 
cd ${base_dir}/${pdb_id}
echo "Working directory: $(pwd)"

# Run the Python script
python ${base_dir}/altloc_clash.py "$pdb_id" "$chain_id" "$res_num" "$lig_name" "$base_dir" >> ${base_dir}/altloc_clash_results.txt
