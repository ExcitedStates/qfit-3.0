# TODO unit tests

import itertools
from collections import defaultdict

import numpy as np


class ClashDetector:
    """Detect clashes between ligand and receptor using spatial hashing.
    
    Optimized with pre-computed grid and efficient numpy operations.
    """

    def __init__(self, ligand, receptor, scaling_factor=0.85, exclude=None):
        self.ligand = ligand
        self.scaling_factor = scaling_factor
        self.receptor = receptor
        receptor_radius = receptor.vdw_radius
        self.ligand_radius = np.asarray(self.ligand.vdw_radius)
        max_receptor_radius = 0 if receptor_radius.size == 0 else receptor_radius.max()
        self.voxelspacing = self.scaling_factor * (
            max_receptor_radius + self.ligand_radius.max()
        )
        # Pre-compute inverse for faster lookups
        self._inv_voxelspacing = 1.0 / self.voxelspacing
        self._half_sf = self.scaling_factor / 2.0
        self.exclude = exclude

        # Build spatial hash grid
        self.grid = defaultdict(list)
        self.radius = defaultdict(list)
        receptor_coor = receptor.coor
        keys = (receptor_coor * self._inv_voxelspacing + 0.5).astype(np.int32)
        
        # Pre-compute translation offsets
        translations = list(itertools.product([-1, 0, 1], repeat=3))
        
        for key, coor, radius in zip(keys, receptor_coor, receptor_radius):
            key = tuple(key)
            for trans in translations:
                new_key = (key[0] + trans[0], key[1] + trans[1], key[2] + trans[2])
                self.grid[new_key].append(coor)
                self.radius[new_key].append(radius)
        
        # Convert lists to numpy arrays for vectorized operations
        for key, value in self.grid.items():
            self.grid[key] = np.asarray(value, dtype=np.float64)
        for key, value in self.radius.items():
            self.radius[key] = np.asarray(value, dtype=np.float64)

    def __call__(self):
        """Check for clashes. Returns True if any clash detected.
        
        Optimized to return early on first clash detection.
        """
        ligand_coor = self.ligand.coor
        active = self.ligand.active
        inv_vs = self._inv_voxelspacing
        half_sf = self._half_sf
        ligand_radius = self.ligand_radius
        grid = self.grid
        radius_dict = self.radius
        exclude = self.exclude
        
        # Get indices of active atoms only
        active_indices = np.nonzero(active)[0] if hasattr(active, '__len__') else range(len(ligand_coor))
        
        for idx in active_indices:
            coor = ligand_coor[idx]
            radius = ligand_radius[idx]
            
            # Compute grid key
            key = (int(coor[0] * inv_vs + 0.5),
                   int(coor[1] * inv_vs + 0.5),
                   int(coor[2] * inv_vs + 0.5))
            
            neighbors = grid.get(key)
            if neighbors is None or len(neighbors) == 0:
                continue
            
            # Vectorized distance calculation
            diff = coor - neighbors
            distance_sq = np.einsum('ij,ij->i', diff, diff)  # Fast squared norm
            cutoff = half_sf * (radius + radius_dict[key])
            cutoff_sq = cutoff * cutoff

            clash_mask = distance_sq < cutoff_sq
            nclashes = clash_mask.sum()
            
            # Check if certain clashes need to be excluded
            if nclashes > 0 and exclude is not None:
                for ligand_ind, rcoor in exclude:
                    # Check if this is the coordinate we are interested in
                    if np.allclose(coor, self.ligand.get_xyz(ligand_ind)):
                        # Now check if it is clashing with the excluded receptor atom
                        clashing_neighbors = neighbors[clash_mask]
                        for cn in clashing_neighbors:
                            if np.allclose(cn, rcoor):
                                nclashes -= 1
                                break
            
            if nclashes > 0:
                return True

        return False
