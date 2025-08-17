import argparse
import pandas as pd
from qfit.structure import Structure

def build_argparser():
    parser = argparse.ArgumentParser(description="Compare alt locs and rotamer states between two PDB files.")
    parser.add_argument("pdb1", type=str, help="First PDB file.")
    parser.add_argument("pdb2", type=str, help="Second PDB file.")
    parser.add_argument("--output", type=str, default="comparison_results.csv", help="Output CSV file.")
    return parser

def count_alt_locs(structure):
    alt_loc_counts = {}
    for residue in structure.residues:
        alt_locs = set(atom.altloc for atom in residue.atoms if atom.altloc)
        alt_loc_counts[residue.id] = len(alt_locs)
    return alt_loc_counts

def compare_rotamers(structure1, structure2):
    rotamer_differences = []
    for res1, res2 in zip(structure1.residues, structure2.residues):
        if res1.id != res2.id:
            continue
        chi_angles1 = res1.chi_angles()
        chi_angles2 = res2.chi_angles()
        for chi1, chi2 in zip(chi_angles1, chi_angles2):
            if abs(chi1 - chi2) > 15:
                rotamer_differences.append((res1.id, chi1, chi2))
    return rotamer_differences

def main():
    parser = build_argparser()
    options = parser.parse_args()

    # Load structures
    structure1 = Structure.fromfile(options.pdb1)
    structure2 = Structure.fromfile(options.pdb2)

    # Count alt locs
    alt_loc_counts1 = count_alt_locs(structure1)
    alt_loc_counts2 = count_alt_locs(structure2)

    # Compare rotamers
    rotamer_differences = compare_rotamers(structure1, structure2)


    # Prepare data for DataFrame
    data = []
    for res_id in alt_loc_counts1.keys():
        alt_locs1 = alt_loc_counts1.get(res_id, 0)
        alt_locs2 = alt_loc_counts2.get(res_id, 0)
        chi1, chi2 = None, None
        for diff in rotamer_differences:
            if diff[0] == res_id:
                chi1, chi2 = diff[1], diff[2]
                break
        data.append([res_id, alt_locs1, alt_locs2, chi1, chi2])

    # Create DataFrame
    df = pd.DataFrame(data, columns=["Residue", "AltLocs_PDB1", "AltLocs_PDB2", "Chi1", "Chi2"])

    # Write DataFrame to CSV
    df.to_csv(options.output, index=False)

if __name__ == "__main__":
    main()
