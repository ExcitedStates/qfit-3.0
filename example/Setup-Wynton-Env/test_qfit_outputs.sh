#!/bin/bash
#$ -l h_vmem=40G
#$ -l mem_free=40G
#$ -t 1-135
#$ -l h_rt=48:00:00
#$ -pe smp 1
#$ -R yes
#$ -V

#________________________________________SOURCE PHENIX/QFIT_____________________________________________#
source /wynton/home/fraserlab/jessicaflowers/phenix/phenix-1.20.1-4487/phenix_env.sh
export PATH="/wynton/home/fraserlab/jessicaflowers/miniconda3/bin:$PATH"

source activate qfit_ligand
which python
export PHENIX_OVERWRITE_ALL=true

#________________________________________________PDB INFO________________________________________________#

PDB_file=/wynton/home/fraserlab/jessicaflowers/tp_qfit_ligands/tp_ligands.txt
PDB_dir='/wynton/home/fraserlab/jessicaflowers/tp_qfit_ligands'
output_dir='/wynton/home/fraserlab/jessicaflowers/tp_qfit_ligands/output_data/results_from_qfit_run_1'

PDB=$(cat $PDB_file | head -n $SGE_TASK_ID | tail -n 1)
category='qfit_run_1'

# Extract and format the required information
pdb_id=$(echo "$PDB" | awk -F, '{print $1}')
path=${PDB_dir}/${pdb_id}/${category}
chain_res=$(echo "$PDB" | awk -F, '{print $(NF-3) "," $(NF-2)}')

#________________________________________________SPLIT MULTICONF________________________________________________
cd ${path}

# Split the refined qFit-Ligand output multiconformer model into seperate PDBs (necessary for the torsion strain calculation)

# this should save qfit_ligand_A.pdb, qfit_ligand_B.pdb, ...
qfit_split_conf=$(split_multiconformer_ligand.py ${pdb_id}_qFit_ligand.pdb  --residue=${chain_res} --directory=${path} --output_name qfit)

# this should save depo_ligand_A.pdb and depo_ligand_B.pdb
depo_split_conf=$(split_multiconformer_ligand.py ${pdb_id}.pdb  --residue=${chain_res} --directory=${path} --output_name depo)

#________________________________________________CONVERT REFINED QFIT MTZ INTO CCP4____________________________
phenix.mtz2map ${pdb_id}_qFit_ligand.mtz ${pdb_id}_qFit_ligand.pdb

#________________________________________________METRICS_______________________________________________________
cd ${output_dir}

# RSCC
rscc_output_csv="$output_dir/${pdb_id}_rscc.csv"
rscc=$(calc_rscc.py ${path}/${pdb_id}_qFit_ligand.mtz ${path}/${pdb_id}_qFit_ligand.pdb ${chain_res} -l 2FOFCWT,PH2FOFCWT -comp ${path}/${pdb_id}.pdb)
echo $rscc
qfit_rscc=$(echo "$rscc" | sed -n 's/.*RSCC for model of interest: \([0-9.]*\).*/\1/p')
depo_rscc=$(echo "$rscc" | sed -n 's/.*RSCC for comparision model: \([0-9.]*\)/\1/p') # this part is optional, and only should be done IF you include the -comp flag in the RSCC calculation
echo "PDB,qfit_rscc,depo_rscc" > "$rscc_output_csv"
echo "${pdb_id},${qfit_rscc},${depo_rscc}" >> "$rscc_output_csv"

# Parse the log file for total run time, and number of output conformers
python ${PDB_dir}/parse_log_file.py --pdb ${pdb_id} --path ${path} --output_dir ${output_dir}

# Torsion strain of qfit conformers and deposited conformers
python ${PDB_dir}/calc_torsion_strain.py --pdb ${pdb_id}_qfit --path ${path} --output_dir ${output_dir} --conf_type qfit_ligand
python ${PDB_dir}/calc_torsion_strain.py --pdb ${pdb_id}_depo --path ${path} --output_dir ${output_dir} --conf_type depo_ligand

# EDIAm
map=${path}/composite_omit_map.mtz
min_resolution=$(phenix.mtz.dump "$map" | grep "Resolution range:" | awk '{print $NF}')
echo "Minimum resolution: $min_resolution"

model=${path}/${pdb_id}_qFit_ligand.pdb
edia ${path}/${pdb_id}_qFit_ligand_2mFo-DFc_no_fill_no_fill.ccp4 ${min_resolution} ${model} --selection ${chain_res} -d ${output_dir} --pbd_name ${pdb_id}_qfit


