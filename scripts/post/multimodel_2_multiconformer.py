#!/usr/bin/env python3
import argparse
import numpy as np
from copy import deepcopy

from qfit.structure.math import calc_rmsd
from qfit.structure import Structure
from qfit.relabel import Relabeller, RelabellerOptions



def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a multi-model PDB into a multi-conformer PDB with RMSD-based clustering."
    )
    parser.add_argument("pdb", help="Input multi-model PDB file.")
    parser.add_argument(
        "-o", "--output", default="multiconformer_ensemble.pdb",
        help="Output multi-conformer PDB file."
    )
    parser.add_argument(
        "--rmsd", type=float, default=1.0,
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


def cluster_conformations(conformations, rmsd_cutoff):
    """
    Cluster conformations by RMSD and select representatives.
    
    Args:
        conformations: List of residue conformations
        rmsd_cutoff: RMSD threshold for clustering
    
    Returns:
        List of representative conformations (one per cluster)
    """
    if len(conformations) <= 1:
        return conformations
    
    # Initialize clusters with first conformation
    clusters = [[0]]  # Each cluster is a list of indices into conformations
    cluster_representatives = [0]  # Index of representative for each cluster
    
    # Process each remaining conformation
    for i in range(1, len(conformations)):
        assigned = False
        
        # Try to assign to existing cluster
        for cluster_idx, cluster in enumerate(clusters):
            rep_idx = cluster_representatives[cluster_idx]
            
            # Calculate RMSD to cluster representative
            rmsd = calc_rmsd(conformations[i].coor, conformations[rep_idx].coor)
            
            if rmsd <= rmsd_cutoff:
                # Add to existing cluster
                cluster.append(i)
                assigned = True
                print(f"    Assigned conformation {i} to cluster {cluster_idx} (RMSD {rmsd:.3f} <= {rmsd_cutoff})")
                break
        
        if not assigned:
            # Create new cluster
            clusters.append([i])
            cluster_representatives.append(i)
            print(f"    Created new cluster {len(clusters)-1} for conformation {i}")
    
    # Return representatives
    representatives = [conformations[idx] for idx in cluster_representatives]
    return representatives


def build_multiconformer(models, rmsd_cutoff):
    """
    Build a multi-conformer structure using efficient clustering approach.
    
    For each residue, collect all conformations across models, cluster them by RMSD,
    and select representatives from each cluster.
    """
    if len(models) <= 1:
        return models[0]
    
    # Start with the first model as the base structure
    base_model = deepcopy(models[0])
    
    # Set altloc to 'A' for all residues in the base model
    base_model.set_altloc('A')
    
    # Collect all residues that need alternate conformations
    residues_to_add = []
    
    # Iterate over each unique residue in the base model
    for base_residue in base_model.single_conformer_residues:
        chain_id = base_residue.chain[0]
        print(f"Processing residue: {base_residue} (chain {chain_id})")
        
        # Collect all conformations for this residue from all models
        all_conformations = []
        
        # Add base model conformation
        all_conformations.append(base_residue)
        
        # Add conformations from other models (same chain)
        for model in models[1:]:
            # Find the corresponding chain in the current model
            model_chain = next((ch for ch in model.chains if ch.id == chain_id), None)
            if model_chain is None:
                continue
            
            # Find the corresponding residue in the current model's chain
            # We need to search through all conformers in the chain
            model_residue = None
            for conformer in model_chain.conformers:
                for residue in conformer.residues:
                    if residue.id == base_residue.id:
                        model_residue = residue
                        break
                if model_residue:
                    break
            
            if model_residue is None:
                continue
            
            model_copy = deepcopy(model_residue)
            model_copy.set_occupancies(0.5)
            all_conformations.append(model_copy)

        
        # Cluster conformations and get representatives
        representatives = cluster_conformations(all_conformations, rmsd_cutoff)
        
        # Store representatives that need to be added (skip first as it's already in base_model)
        if len(representatives) > 1:
            Altlocs = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
            alt_labels = iter(Altlocs)
            
            for i, rep in enumerate(representatives[1:], 1):
                altloc = next(alt_labels, None)
                if altloc is None:
                    print("Warning: Ran out of altloc labels.")
                    break
                
                rep.set_altloc(altloc)
                residues_to_add.append(rep)
    
    # Combine all residues at once to minimize memory usage
    if residues_to_add:
        # Combine in batches to avoid memory issues
        batch_size = 50  
        combined_structure = base_model
        
        for i in range(0, len(residues_to_add), batch_size):
            batch = residues_to_add[i:i + batch_size]

            # Combine batch with current structure
            for residue in batch:
                combined_structure = combined_structure.combine(residue)
        
        return combined_structure
    else:
        return base_model


def main():
    args = parse_args()
    structure = Structure.fromfile(args.pdb)

    # Split multi-model PDB into separate single-model structures
    models = structure.split_models()

    print(f"Loaded {len(models)} models from {args.pdb}")

    # Build multi-conformer structure
    multiconf = build_multiconformer(models, args.rmsd)
    # Relabel alternate conformers
    options = RelabellerOptions()
    relabeller = Relabeller(multiconf, options)
    multiconf = relabeller.run()

    # Write to file
    multiconf.reorder().tofile(args.output)


if __name__ == "__main__":
    main()
