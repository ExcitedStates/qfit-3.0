import argparse
import requests

"""
This script fetches the SMILES string for a given ligand ID using the RCSB PDB REST API.

Input:
    LIGAND_ID: The ligand ID to fetch the SMILES string for.

<<<<<<< HEAD
Output: 
=======
Output:
>>>>>>> origin/dev
    SMILES string printed to your console.

Example:
    python get_smiles.py ATP
"""

<<<<<<< HEAD
# Function to perform the REST API call
def fetch_smiles_for_ligand(ligand_id):
    # Construct the URL for the ligand information
    url = f'https://data.rcsb.org/rest/v1/core/chemcomp/{ligand_id}'
    
    # Make the GET request to fetch the ligand information
    response = requests.get(url)
    
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        
        # Retrieve the SMILES string using the specified JSON path
        smiles = data.get('rcsb_chem_comp_descriptor', {}).get('smiles')
        return smiles
    else:
        print(f"Failed to fetch data for ligand '{ligand_id}' with status code: {response.status_code}")
        return None

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Fetch SMILES string for a given ligand.')
    parser.add_argument('ligand_id', type=str, help='The ligand ID to fetch the SMILES string for.')
=======

# Function to perform the REST API call
def fetch_smiles_for_ligand(ligand_id):
    # Construct the URL for the ligand information
    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{ligand_id}"

    # Make the GET request to fetch the ligand information
    response = requests.get(url)

    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

        # Retrieve the SMILES string using the specified JSON path
        smiles = data.get("rcsb_chem_comp_descriptor", {}).get("smiles")
        return smiles
    else:
        print(
            f"Failed to fetch data for ligand '{ligand_id}' with status code: {response.status_code}"
        )
        return None


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Fetch SMILES string for a given ligand."
    )
    parser.add_argument(
        "ligand_id", type=str, help="The ligand ID to fetch the SMILES string for."
    )
>>>>>>> origin/dev
    args = parser.parse_args()

    # Fetch the SMILES string for the provided ligand ID
    smiles = fetch_smiles_for_ligand(args.ligand_id)
<<<<<<< HEAD
    
=======

>>>>>>> origin/dev
    # Print the SMILES string to the console
    if smiles:
        print(smiles)
    else:
        print(f"No SMILES string found for ligand '{args.ligand_id}'.")
