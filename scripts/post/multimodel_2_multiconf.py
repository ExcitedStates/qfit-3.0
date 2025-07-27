#!/usr/bin/env python3
import argparse
import numpy as np
import string
from copy import deepcopy
from qfit.structure.math import calc_rmsd

from qfit.structure import Structure
from qfit import relabel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a multi-model PDB into a multi-conformer PDB with RMSD-based clustering."
    )
    parser.add_argument("pdb", help="Input multi-model PDB file.")
    parser.add_argument(
        "-o", "--output", default="multiconformer.pdb",
        help="Output multi-conformer PDB file."
    )
    parser.add_argument(
        "--rmsd", type=float, default=0.5,
        help="RMSD cutoff (in Å) to cluster residue conformations."
    )
    return parser.parse_args()



def cluster_residues(residues, rmsd_cutoff):
    """
    Cluster residue conformations from different models using RMSD.
    Returns a list of representative residues (one per cluster).
    """
    clusters = []
    for res in residues:
        coords = np.array([atom.coor for atom in res.atoms])
        added = False
        for cluster in clusters:
            ref_coords = np.array([atom.coor for atom in cluster[0].atoms])
            if calc_rmsd(coords, ref_coords) < rmsd_cutoff:
                cluster.append(res)
                added = True
                break
        if not added:
            clusters.append([res])
    return [c[0] for c in clusters]  # return one representative per cluster


def build_multiconformer(models, rmsd_cutoff):
    """
    Build a multi-conformer structure by clustering residues across models.
    """
    if len(models) <= 1:
        return models[0]
    
    # Start with the first model as the base structure
    base_model = deepcopy(models[0])
    
    # Set altloc to 'A' for all residues in the base model
    base_model.set_altloc('A')
    
    combined_structure = deepcopy(base_model)
    
    # Iterate over each residue in the base model
    for base_residue in base_model.residues:
        print(f"Processing residue: {base_residue}")
        base_coords = base_residue.coor
        
        # Initialize altloc label iterator starting from 'B'
        alt_labels = iter(string.ascii_uppercase[1:])
        
        # Iterate over each subsequent model
        for model in models[1:]:
            # Find the corresponding residue in the current model
            model_residue = next((res for res in model.residues if res.id == base_residue.id), None)
            if model_residue is None:
                print(f"  No corresponding residue found in model")
                continue
            
            print(f"  Comparing with residue: {model_residue}")
            model_coords = model_residue.coor
            rmsd = calc_rmsd(base_coords, model_coords)
            print(f"  RMSD: {rmsd}")
            
            if rmsd > rmsd_cutoff:
                print(f"  Adding alternate conformation (RMSD {rmsd} > {rmsd_cutoff})")
                # Create a copy of the residue with modified altloc
                residue_copy = deepcopy(model_residue)
                
                # Get the next available altloc label
                altloc = next(alt_labels, None)
                if altloc is None:
                    print("Warning: Ran out of altloc labels.")
                    break
                
                residue_copy.set_altloc(altloc)
                
                # Set occupancy to 1.0 for this residue
                residue_copy.set_occupancies(1.0)
                
                # Combine the residue with the combined structure
                combined_structure = combined_structure.combine(residue_copy)
    
    return combined_structure


def main():
    args = parse_args()
    structure = Structure.fromfile(args.pdb)

    # Split multi-model PDB into separate single-model structures
    models = structure.split_models()

    print(f"Loaded {len(models)} models from {args.pdb}")

    # Build multi-conformer structure
    multiconf = build_multiconformer(models, args.rmsd)

    # Reorder atoms within residues according to rotamer library
    multiconf = multiconf.reorder()

    # Write to file
    multiconf.tofile(args.output)


if __name__ == "__main__":
    main()

