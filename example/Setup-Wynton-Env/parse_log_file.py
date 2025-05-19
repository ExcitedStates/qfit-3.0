import os
import csv
import argparse


# Set up argument parser
parser = argparse.ArgumentParser(description='Parse qFit-ligand log file to read total run time and number of output conformers.')
parser.add_argument('--pdb', required=True, help='PDB identifier for labeling in CSV')
parser.add_argument('--path', required=True, help='Path to data')
parser.add_argument('--output_dir', required=True, help='Directory to save the output CSV file')
args = parser.parse_args()

# FIND TOTAL RUN TIME
# Initialize an empty list to store the results
results = []
log_file = os.path.join(args.path, 'qfit_ligand.log')

# Check if the log file exists
if os.path.isfile(log_file):
    found = False
    with open(log_file, 'r') as f:
        for line in f:
            if 'Total time:' in line:
                # Extract the time from the line
                time_str = line.split('Total time: ')[1].strip()
                # Add the time to the results list
                results.append(time_str)
                found = True
                break
    if not found:
        # If 'Total time:' not found, add 'NA' to results
        results.append("NA")
        print(f"Total time not found in log file for {args.pdb}")
else:
    # If the log file does not exist
    results.append("NA")
    print(f"Log file not found in path: {args.path}")

# Write to CSV
csv_filename = os.path.join(args.output_dir, f"{args.pdb}_qfit_runtime.csv")
with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['PDBID', 'Time'])
    writer.writerow([args.pdb] + results)

print("Runtime results have been written to", csv_filename)


# FIND NUMBER OF FINAL CONFORMERS
# Initialize an empty list to store the results
conformer_results = []
log_file = os.path.join(args.path, 'qfit_ligand.log')

# Check if the log file exists
if os.path.isfile(log_file):
    found = False
    with open(log_file, 'r') as f:
        for line in f:
            if 'Number of final conformers:' in line:
                # Extract the number from the line
                num_confs = line.split('Number of final conformers: ')[1].strip()
                # Add the number to the conformer_results list
                conformer_results.append(num_confs)
                found = True
                break
    if not found:
        # If 'Number of final conformers:' not found, add 'NA' to conformer_results
        conformer_results.append("NA")
        print(f"Number of final conformers not found in log file for {args.pdb}")
else:
    # If the log file does not exist
    conformer_results.append("NA")
    print(f"Log file not found in path: {args.path}")

# Write to CSV
csv_filename_confs = os.path.join(args.output_dir, f"{args.pdb}_num_confs.csv")
with open(csv_filename_confs, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['PDBID', 'Number of Final Conformers'])
    writer.writerow([args.pdb] + conformer_results)

print("Number of final conformers results have been written to", csv_filename_confs)
