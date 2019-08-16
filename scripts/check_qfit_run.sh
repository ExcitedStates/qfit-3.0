#!/bin/bash

# This program expects the input pdb used for qFit as the first argument:
input_pdb=$1;

# This program expects the output directory of qFit as the second argument:
qfit_output_dir=$2;

# Extract the number of residues and HETATM molecules from the input PDB:
NUM_RESIDUES=$(grep "^ATOM" $input_pdb | cut -c 22-26 | sort | uniq | wc -l);
HET_RESIDUES=$(grep "^HETATM" $input_pdb | cut -c 22-26 | sort | uniq | wc -l);

if [ -f $qfit_output_dir/multiconformer_model.pdb ]
then
    if [ -f $qfit_output_dir/multiconformer_model2.pdb ]
    then
        echo "[STATUS] qFit run has finished.";
        NUM_RESIDUES2=$(grep "^ATOM" $qfit_output_dir/multiconformer_model2.pdb | cut -c 22-26  | sort | uniq | wc -l);
        HET_RESIDUES2=$(grep "^HETATM" $qfit_output_dir/multiconformer_model2.pdb | cut -c 22-26 | sort | uniq | wc -l);
    else
        echo -e "[STATUS] qfit has completed modeling on a residue level. Running qFit_segment.\nPlease, make sure the qFit executable is still running (e.g. ps ux OR pgrep qfit)";
        NUM_RESIDUES2=$(grep "^ATOM" $qfit_output_dir/multiconformer_model.pdb | cut -c 22-26 | sort | uniq | wc -l);
        HET_RESIDUES2=$(grep "^HETATM" $qfit_output_dir/multiconformer_model.pdb | cut -c 22-26 | sort | uniq | wc -l);
    fi
    if [ ! $NUM_RESIDUES -eq $NUM_RESIDUES2 ] || [ ! $HET_RESIDUES -eq $HET_RESIDUES2 ]
    then
        echo "[ERROR] qFit has produced models for ($NUM_RESIDUES2/$NUM_RESIDUES) residues and ($HET_RESIDUES2/$HET_RESIDUES) hetatom entries.";
    else
        numAs=$(cut -c 17,21-26 $qfit_output_dir/multiconformer_model2.pdb | sed "s/^ /_/" | sort -k 3,3n -k 2,2 -k 1,1 | uniq | grep "^A" | wc -l)
        numBs=$(cut -c 17,21-26 $qfit_output_dir/multiconformer_model2.pdb | sed "s/^ /_/" | sort -k 3,3n -k 2,2 -k 1,1 | uniq | grep "^B" | wc -l)
        if [ ! $numAs -eq $numBs ]
        then
          echo "[ERROR] There are single conformer residues with an 'A' altloc identifier."
          echo "Please, fix this before running refinement."
        else
          echo "[SUCCESS] qFit has been run successfully.";
        fi
    fi
else
    echo -e "[STATUS] qFit should still be running.\nPlease, make sure the qFit executable is still running (e.g. ps ux OR pgrep qfit).";
    NUM_RESIDUES2=$(ls $qfit_output_dir/*/multiconformer_residue.pdb | wc -l);
    PROGRESS=$(echo "$NUM_RESIDUES2*100/$NUM_RESIDUES" | bc);
    echo "[RUNNING] qFit run has modeled $PROGRESS % ($NUM_RESIDUES2/$NUM_RESIDUES) of residues.";
fi
