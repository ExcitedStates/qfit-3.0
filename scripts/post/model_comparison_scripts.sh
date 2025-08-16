#!/usr/bin/env bash

usage() {
    echo "Usage: $0 -f <pdb_list_file> [-b <base_suffix>] [-c <comp_suffix>] [-l <labels>] [-m <mtz_suffix>] [-o <output_folder>]"
    echo "  -f <pdb_list_file> : File containing list of PDBs, one per line."
    echo "  -b <base_suffix>   : Suffix for base PDB files (optional)."
    echo "  -c <comp_suffix>   : Suffix for comparison PDB files (optional)."
    echo "  -l <labels>        : Labels for MTZ (optional)."
    echo "  -m <mtz_suffix>    : Suffix for MTZ files (optional)."
    echo "  -o <output_folder> : Directory to store output files (optional)."
    exit 1
}

# Defaults
base_suffix=""
comp_suffix=""
mtz_suffix=""
labels=""
output_folder="./output"

# Parse command line arguments
while getopts ":f:b:c:l:m:o:" opt; do
    case $opt in
        f) pdb_list_file="$OPTARG" ;;
        b) base_suffix="$OPTARG" ;;
        c) comp_suffix="$OPTARG" ;;
        l) labels="$OPTARG" ;;
        m) mtz_suffix="$OPTARG" ;;
        o) output_folder="$OPTARG" ;;
        *) usage ;;
    esac
done

# Check if pdb_list_file is provided
if [ -z "$pdb_list_file" ]; then
    usage
fi

# Create output directory if it doesn't exist
mkdir -p "$output_folder"

# Check that pdb_list_file exists
if [ ! -f "$pdb_list_file" ]; then
    echo "Error: PDB list file '$pdb_list_file' not found."
    exit 1
fi

# Iterate over each PDB in the list
while IFS= read -r pdb; do
    [ -z "$pdb" ] && continue  # skip empty lines

    # Construct file names
    base_pdb="${pdb}/${pdb}${base_suffix}.pdb"
    comp_pdb="${pdb}/${pdb}${comp_suffix}.pdb"
    mtz_name="${pdb}/${pdb}${mtz_suffix}.mtz"
    echo $output_folder
    echo "Running compare_rscc_voxel for $pdb..."
    compare_rscc_voxel.py "$base_pdb" "$mtz_name" \
        --comp_pdb "$comp_pdb" \
        -l "$labels" \
        --directory "$output_folder" \
        --pdb "$pdb"

    # Run compare_edia
    #echo "Running compare_edia for $pdb..."
    #python compare_edia.py "$base_pdb" "$mtz_name" --comp_pdb "$comp_pdb" --directory "$output_folder"

    # Run compare_rotamer (assuming a similar script exists)
    echo "Running compare_rotamer for $pdb..."
    compare_rotamer_altlocs.py "$base_pdb" "$comp_pdb" --base_pdb_id "${pdb}${base_suffix}" --comp_pdb_id "${pdb}${comp_suffix}" --directory "$output_folder"

done < "$pdb_list_file"

