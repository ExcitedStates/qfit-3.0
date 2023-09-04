# TODO unit tests

import itertools
from collections import defaultdict

import numpy as np


class ClashDetector:

    """Detect clashes between ligand and receptor using spatial hashing."""

    def __init__(self, ligand, receptor, scaling_factor=0.85, exclude=None):
        self.ligand = ligand
        self.scaling_factor = scaling_factor
        receptor_radius = receptor.vdw_radius
        self.ligand_radius = self.ligand.vdw_radius
        self.voxelspacing = self.scaling_factor * (
            receptor_radius.max() + self.ligand_radius.max()
        )
        self.exclude = exclude

        self.grid = defaultdict(list)
        self.radius = defaultdict(list)
        receptor_coor = receptor.coor
        ligand_coor = ligand.coor
        keys = (receptor.coor / self.voxelspacing + 0.5).astype(int)
        translations = list(itertools.product([-1, 0, 1], repeat=3))
        for key, coor, radius in zip(keys, receptor_coor, receptor_radius):
            key = tuple(key)
            for trans in translations:
                new_key = tuple(x + tx for x, tx in zip(key, trans))
                self.grid[new_key].append(coor)
                self.radius[new_key].append(radius)
        for key, value in self.grid.items():
            self.grid[key] = np.asarray(value)
        for key, value in self.radius.items():
            self.radius[key] = np.asarray(value)
        self.receptor = receptor

    def __call__(self):
        inv_voxelspacing = 1 / self.voxelspacing
        ligand_coor = self.ligand.coor
        active = self.ligand.active
        half_sf = self.scaling_factor / 2.0
        nclashes = 0
        for is_active, coor, radius in zip(active, ligand_coor, self.ligand_radius):
            if not is_active:
                continue
            key = tuple(int(x * inv_voxelspacing + 0.5) for x in coor)
            neighbors = self.grid[key]
            if len(neighbors) == 0:
                continue
            diff = coor - neighbors
            distance_sq = (diff * diff).sum(axis=1)
            cutoff = half_sf * (radius + self.radius[key])
            cutoff_sq = cutoff * cutoff

            clash_mask = distance_sq < cutoff_sq
            nclashes += clash_mask.sum()
            # Check if certain clashes need to be excluded
            if self.exclude is not None:
                for ligand_ind, rcoor in self.exclude:
                    # Check if this is the coordinate we are interested in
                    if np.allclose(coor, self.ligand._coor[ligand_ind]):
                        # Now check if it is clashing with the excluded receptor atom
                        if rcoor in neighbors[clash_mask]:
                            nclashes -= 1
            if nclashes > 0:
                break

        return nclashes > 0
