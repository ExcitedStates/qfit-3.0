import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np

# Argument parser for output_folder, base_suffix, and comp_suffix
parser = argparse.ArgumentParser(description="Concatenate RSCC CSVs and plot RSCC comparison.")
parser.add_argument("--output_folder", type=str, required=True, help="Directory containing RSCC CSV files.")
parser.add_argument("--base_suffix", type=str, required=True, help="Suffix for base PDB files.")
parser.add_argument("--comp_suffix", type=str, required=True, help="Suffix for comparison PDB files.")
args = parser.parse_args()

# Define the directory containing the CSV files
csv_directory = args.output_folder

# Initialize an empty DataFrame to store concatenated data
all_rscc_data = pd.DataFrame()

# Iterate over all files in the directory
for filename in os.listdir(csv_directory):
    if filename.endswith("_rscc.csv"):
        file_path = os.path.join(csv_directory, filename)
        # Read each CSV file and append it to the DataFrame
        df = pd.read_csv(file_path)
        all_rscc_data = pd.concat([all_rscc_data, df], ignore_index=True)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(all_rscc_data['Base_RSCC'], all_rscc_data['Comparison_RSCC'], alpha=0.5)
plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, color='red', linestyle='--')  # 1 to 1 line
plt.xlabel('Base RSCC')
plt.ylabel('Comparison RSCC')

# Save the plot
plot_filename = os.path.join(csv_directory, f"rscc_comparison_{args.base_suffix}_{args.comp_suffix}.png")
plt.savefig(plot_filename)
plt.close()

# Create a grid of small plots for RSCC by each residue name
unique_residues = all_rscc_data['Residue_Name'].unique()
num_plots = min(20, len(unique_residues))  # Limit to 20 plots

# Determine grid size
grid_size = int(np.ceil(np.sqrt(num_plots)))

# Create a figure with subplots
fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
axes = axes.flatten()

# Plot RSCC for each unique residue name
for i, residue in enumerate(unique_residues[:num_plots]):
    ax = axes[i]
    residue_data = all_rscc_data[all_rscc_data['Residue_Name'] == residue]
    ax.scatter(residue_data['Base_RSCC'], residue_data['Comparison_RSCC'], alpha=0.5)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='red', linestyle='--')  # 1 to 1 line
    ax.set_title(f'Residue: {residue}')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Base RSCC')
    ax.set_ylabel('Comparison RSCC')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()

# Save the grid plot
grid_plot_filename = os.path.join(csv_directory, f"rscc_residue_grid_{args.base_suffix}_{args.comp_suffix}.png")
plt.savefig(grid_plot_filename)
plt.close()


# Read in altloc_differences.csv files and plot distribution of differences of alt locs
altloc_data = pd.DataFrame()

# Iterate over all files in the directory
for filename in os.listdir(csv_directory):
    if filename.endswith("_altloc_differences.csv"):
        file_path = os.path.join(csv_directory, filename)
        # Read each CSV file and append it to the DataFrame
        df = pd.read_csv(file_path)
        altloc_data = pd.concat([altloc_data, df], ignore_index=True)

base_col = [col for col in altloc_data.columns if f'{args.base_suffix}' in col]
comp_col = [col for col in altloc_data.columns if f'{args.comp_suffix}' in col]

if base_col and comp_col:
    altloc_data['Altloc_Difference'] = altloc_data.apply(
        lambda row: abs(row[base_col[0]] - row[comp_col[0]]), axis=1
    )

# Plotting the distribution of alt loc differences
plt.figure(figsize=(10, 6))
# Create a new column for categorical differences from -5 to 5
altloc_data['Categorical_Difference'] = pd.cut(
    altloc_data['Altloc_Difference'], 
    bins=[-2, -1, 0, 1, 2, 3, 4, 5], 
    labels=[-2, -1, 0, 1, 2, 3, 4],
    include_lowest=True
)
# Plot the histogram of the categorical differences
plt.hist(altloc_data['Categorical_Difference'].dropna(), bins=11, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Categorical Difference in Alt Locs (-5 to 5)')
plt.ylabel('Frequency')
plt.title('Distribution of Categorical Differences in Alt Locs')

# Save the plot
altloc_plot_filename = os.path.join(csv_directory, f"altloc_difference_distribution_{args.base_suffix}_{args.comp_suffix}.png")
plt.savefig(altloc_plot_filename)
plt.close()

# Read in rotamer_difference.csv files and plot distribution of % of same/diff/shared
rotamer_data = pd.DataFrame()

# Iterate over all files in the directory
for filename in os.listdir(csv_directory):
    if filename.endswith("_rotamer_difference.csv"):
        file_path = os.path.join(csv_directory, filename)
        # Read each CSV file and append it to the DataFrame
        df = pd.read_csv(file_path)
        rotamer_data = pd.concat([rotamer_data, df], ignore_index=True)

# Calculate the percentage of each classification
classification_counts = rotamer_data['classification'].value_counts(normalize=True) * 100

# Plotting the distribution of rotamer classifications
plt.figure(figsize=(10, 6))
classification_counts.plot(kind='bar', color=['green', 'orange', 'blue'], edgecolor='black')
plt.xlabel('Rotamer Classification')
plt.ylabel('Percentage')
plt.title('Distribution of Rotamer Classifications (% Same/Different/Shared)')

# Save the plot
rotamer_plot_filename = os.path.join(csv_directory, f"rotamer_classification_distribution_{args.base_suffix}_{args.comp_suffix}.png")
plt.savefig(rotamer_plot_filename)
plt.close()


