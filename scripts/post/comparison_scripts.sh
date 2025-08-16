#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 -f <pdb_list_file> [-b <base_prefix>] [-c <comp_prefix>] [-s <suffix>] [-m <mtz_suffix>] [-o <output_folder>]"
    echo "  -f <pdb_list_file> : File containing list of PDBs, one per line."
    echo "  -b <base_prefix>   : Prefix for base PDB files (optional)."
    echo "  -c <comp_prefix>   : Prefix for comparison PDB files (optional)."
    echo "  -s <suffix>        : Suffix for PDB files (optional)."
    echo "  -m <mtz_suffix>    : Suffix for MTZ files (optional)."
    echo "  -o <output_folder> : Directory to store output files (optional)."
    exit 1
}

# Parse command line arguments
while getopts ":f:b:c:s:m:o:" opt; do
    case $opt in
        f) pdb_list_file="$OPTARG" ;;
        b) base_prefix="$OPTARG" ;;
        c) comp_prefix="$OPTARG" ;;
        s) suffix="$OPTARG" ;;
        m) mtz_suffix="$OPTARG" ;;
        o) output_folder="$OPTARG" ;;
        *) usage ;;
    esac
done

# Check if pdb_list_file is provided
if [ -z "$pdb_list_file" ]; then
    usage
fi

# Check if output_folder is provided, if not set default
if [ -z "$output_folder" ]; then
    output_folder="./output"
fi

# Create output directory if it doesn't exist
mkdir -p "$output_folder"

# Read the PDB list file
if [ ! -f "$pdb_list_file" ]; then
    echo "Error: PDB list file '$pdb_list_file' not found."
    exit 1
fi

# Iterate over each PDB in the list
while IFS= read -r pdb; do
    # Construct file names
    base_pdb="${base_prefix}${pdb}${suffix}.pdb"
    comp_pdb="${comp_prefix}${pdb}${suffix}.pdb"
    mtz_name="${pdb}${mtz_suffix}.mtz"

    # Run compare_rscc_voxel
    echo "Running compare_rscc_voxel for $pdb..."
    python compare_rscc_voxel.py "$base_pdb" "$mtz_name" --comp_pdb "$comp_pdb" --directory "$output_folder"

    # Run compare_edia
    echo "Running compare_edia for $pdb..."
    python compare_edia.py "$base_pdb" "$mtz_name" --comp_pdb "$comp_pdb" --directory "$output_folder"

    # Run compare_rotamer (assuming a similar script exists)
    echo "Running compare_rotamer for $pdb..."
    python compare_rotamer.py "$base_pdb" "$mtz_name" --comp_pdb "$comp_pdb" --directory "$output_folder"

done < "$pdb_list_file"
