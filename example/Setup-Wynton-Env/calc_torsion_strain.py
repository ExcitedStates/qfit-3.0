import os
import subprocess
import glob
import pandas as pd
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Calculate torsion strain and output to CSV.')
parser.add_argument('--pdb', required=True, help='PDB identifier for labeling in CSV')
parser.add_argument('--path', required=True, help='Path to data')
parser.add_argument('--output_dir', required=True, help='Directory to save the output CSV file')
parser.add_argument(
    "--conf_type",
    type=str,
    required=True,
    help="This will affect the output file naming pattern."
)
args = parser.parse_args()

# The directory where the Torsion_Strain.py script and related files are located
script_dir = '/wynton/home/fraserlab/jessicaflowers/STRAIN_FILTER'

# Function to extract occupancy from a PDB file
def extract_occupancy(pdb_file):
    occupancy = 1.0  # Default value if not found
    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    occupancy = float(line[54:60].strip())
                    break  # Assume all ATOM/HETATM lines have the same occupancy
                except ValueError:
                    continue
    return occupancy

energy_data = []

# Find all pdb files based on the conf_type argument
pdb_files = glob.glob(os.path.join(args.path, f'{args.conf_type}_*.pdb'))

# Debug: print found conf_type_ligand_*.pdb files
print(f"Found {args.conf_type}_*.pdb files: {pdb_files}")

# Loop through all found conformers
for pdb_file in pdb_files:
    ligand_name = os.path.splitext(os.path.basename(pdb_file))[0]
    print("\n")
    print("ligand name = ", ligand_name)
    mol2_file = pdb_file.replace('.pdb', '.mol2')
    csv_file = pdb_file.replace('.pdb', '_Torsion_Strain.csv')

    try:
        # Convert PDB to MOL2
        obabel_path = '/wynton/home/fraserlab/jessicaflowers/Tools/openbabel-install/bin/obabel'
        subprocess.run([obabel_path, pdb_file, '-O', mol2_file, '-d'])

        # Run the strain calculation
        subprocess.run(['python', os.path.join(script_dir, 'Torsion_Strain.py'), mol2_file], cwd=script_dir)

        # Extract occupancy
        occupancy = extract_occupancy(pdb_file)
        print(f"occupancy {occupancy}")

        # Extract the first numerical value (strain) from the CSV file
        df = pd.read_csv(csv_file)
        cols = df.columns
        try:
            energy = float(cols[1])  # Attempt to convert the string to float
            energy_data.append((energy, occupancy))  # Append valid energy data
            print(f"energy {energy}")
        except ValueError:
            print(f"Skipping {ligand_name} due to non-numeric energy value: {cols[1]}")
            continue  # Skip this ligand if the energy value is not numeric

        # Cleanup: Remove the intermediate mol2 and csv file if desired
        os.remove(mol2_file)
        os.remove(csv_file)

    except subprocess.CalledProcessError as e:
        print(f"Error processing {ligand_name}: {e}")
        continue
    except pd.errors.EmptyDataError:
        print(f"Skipping {ligand_name} due to empty CSV file.")
        continue
    except Exception as e:
        print(f"Unexpected error processing {ligand_name}: {e}")
        continue

if energy_data:
    # Calculate weighted average energy
    total_weighted_energy = sum(energy * occupancy for energy, occupancy in energy_data)
    total_occupancy = sum(occupancy for _, occupancy in energy_data)
    weighted_average_energy = total_weighted_energy / total_occupancy

    # Save to a new CSV file
    final_csv_path = os.path.join(args.output_dir, f'{args.pdb}_weighted_energy.csv')
    pd.DataFrame({'Weighted_Average_Energy': [weighted_average_energy]}).to_csv(final_csv_path, index=False)

print("Processing complete.")