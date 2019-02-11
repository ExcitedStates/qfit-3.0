'''
Excited States software: qFit 3.0

Contributors: Saulo H. P. de Oliveira, Gydo van Zundert, and Henry van den Bedem.
Contact: vdbedem@stanford.edu

Copyright (C) 2009-2019 Stanford University
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

This entire text, including the above copyright notice and this permission notice
shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
'''

from collections import defaultdict
from itertools import product

import numpy as np
from scipy.spatial.distance import pdist, squareform

from .base_structure import _BaseStructure
from .residue import residue_type

from .math import aa_to_rotmat


class _Ligand(_BaseStructure):

    """Ligand class automatically generates a topology on the structure."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._get_connectivity()
        self.id = (kwargs['resi'], kwargs['icode'])
        self.type = kwargs["type"]

    #def __init__(self, structure_ligand):
    #    super().__init__(structure_ligand.data,structure_ligand._selection,
    #                     structure_ligand.parent)
    #    self._get_connectivity()
    #    resi, icode = structure_ligand.resi[0], structure_ligand.icode[0]
    #    if icode != '':
    #        self.id = (int(resi), icode)
    #    else:
    #        self.id = int(resi)
    #    self.type = residue_type(structure_ligand)

    def __repr__(self):
        string = 'Ligand: {}. Number of atoms: {}.'.format(self.resn[0], self.natoms)
        return string

    def _get_connectivity(self):
        """Determine connectivity matrix of ligand and associated distance
        cutoff matrix for later clash detection.
        """
        coor = self.coor
        #dm_size = self.natoms * (self.natoms - 1) // 2
        #dm = np.zeros(dm_size, np.float64)
        #k = 0
        #for i in range(0, self.natoms - 1):
        #    u = coor[i]
        #    for j in range(i + 1, self.natoms):
        #        u_v = u - coor[j]
        #        dm[k] = np.dot(u_v, u_v)
        #        k += 1
        dist_matrix = squareform(pdist(coor))
        covrad = self.covalent_radius
        natoms = self.natoms
        cutoff_matrix = np.repeat(covrad, natoms).reshape(natoms, natoms)
        # Add 0.5 A to give covalently bound atoms more room
        cutoff_matrix = cutoff_matrix + cutoff_matrix.T + 0.5
        connectivity_matrix = (dist_matrix < cutoff_matrix)
        # Atoms are not connected to themselves
        np.fill_diagonal(connectivity_matrix, False)
        self.connectivity = connectivity_matrix
        self._cutoff_matrix = cutoff_matrix

    def clashes(self):
        """Checks if there are any internal clashes."""
        dist_matrix = squareform(pdist(self.coor))
        mask = np.logical_not(self.connectivity)
        active_matrix = (self.active.reshape(1, -1) * self.active.reshape(-1, 1)) > 0
        mask &= active_matrix
        np.fill_diagonal(mask, False)
        clash_matrix = dist_matrix < self._cutoff_matrix
        if np.any(np.logical_and(clash_matrix, mask)):
            return True
        return False

    def bonds(self):
        """Print bonds"""
        indices = np.nonzero(self.connectivity)
        for a, b in zip(*indices):
            print(self.name[a], self.name[b])

    def ring_paths(self):
        def ring_path(T, v1, v2):
            v1path = []
            v = v1
            while v is not None:
                v1path.append(v)
                v = T[v]
            v = v2
            v2path = []
            while v not in v1path:
                v2path.append(v)
                v = T[v]
            ring = v1path[0:v1path.index(v) + 1] + v2path
            return ring
        ring_paths = []
        T = {}
        conn = self.connectivity
        for root in range(self.natoms):
            if root in T:
                continue
            T[root] = None
            fringe = [root]
            while fringe:
                a = fringe[0]
                del fringe[0]
                # Scan the neighbors of a
                for n in np.flatnonzero(conn[a]):
                    if n in T and n == T[a]:
                        continue
                    elif n in T and (n not in fringe):
                        ring_paths.append(ring_path(T, a, n))
                    elif n not in fringe:
                        T[n] = a
                        fringe.append(n)
        return ring_paths

    def rotatable_bonds(self):

        """Determine all rotatable bonds.

        A rotatable bond is currently described as two neighboring atoms with
        more than 1 neighbor and which are not part of the same ring.
        """

        conn = self.connectivity
        rotatable_bonds = []
        rings = self.ring_paths()
        for atom in range(self.natoms):
            neighbors = np.flatnonzero(conn[atom])
            if len(neighbors) == 1:
                continue
            for neighbor in neighbors:
                neighbor_neighbors = np.flatnonzero(conn[neighbor])
                new_bond = False
                if len(neighbor_neighbors) == 1:
                    continue
                # Check whether the two atoms are part of the same ring.
                same_ring = False
                for ring in rings:
                    if atom in ring and neighbor in ring:
                        same_ring = True
                        break
                if not same_ring:
                    new_bond = True
                    for b in rotatable_bonds:
                        # Check if we already found this bond.
                        if atom in b and neighbor in b:
                            new_bond = False
                            break
                if new_bond:
                    rotatable_bonds.append((atom, neighbor))
        return rotatable_bonds

    def rigid_clusters(self):

        """Find rigid clusters / seeds in the molecule.

        Currently seeds are either rings or terminal ends of the molecule, i.e.
        the last two atoms.
        """

        conn = self.connectivity
        rings = self.ring_paths()
        clusters = []
        for root in range(self.natoms):
            # Check if root is Hydrogen
            element = self.e[root]
            if element == 'H':
                continue
            # Check if root has already been clustered
            clustered = False
            for cluster in clusters:
                if root in cluster:
                    clustered = True
                    break
            if clustered:
                continue
            # If not, start new cluster
            cluster = [root]
            # Check if atom is part of a ring, if so add all atoms. This
            # step combines multi-ring systems.
            ring_atom = False
            for atom, ring in product(cluster, rings):
                if atom in ring:
                    ring_atom = True
                    for a in ring:
                        if a not in cluster:
                            cluster.append(a)

            # If root is not part of a ring, check if it is connected to a
            # terminal heavy atom.
            if not ring_atom:
                neighbors = np.flatnonzero(conn[root])
                for n in neighbors:
                    if self.e[n] == 'H':
                        continue
                    neighbor_neighbors = np.flatnonzero(conn[n])
                    # Hydrogen neighbors don't count
                    hydrogen_neighbors = (self.e[neighbor_neighbors] == 'H').sum()
                    if len(neighbor_neighbors) - hydrogen_neighbors == 1:
                        cluster.append(n)

            if len(cluster) > 1:
                clusters.append(cluster)
        # Add all left-over single unclustered atoms
        for atom in range(self.natoms):
            found = False
            for cluster in clusters:
                if atom in cluster:
                    found = True
                    break
            if not found:
                clusters.append([atom])
        return clusters

    def atoms_to_rotate(self, bond_or_root, neighbor=None):
        """Return indices of atoms to rotate given a bond."""

        if neighbor is None:
            root, neighbor = bond_or_root
        else:
            root = bond_or_root

        neighbors = [root]
        atoms_to_rotate = self._find_neighbors_recursively(neighbor, neighbors)
        atoms_to_rotate.remove(root)
        return atoms_to_rotate

    def _find_neighbors_recursively(self, neighbor, neighbors):
        neighbors.append(neighbor)
        local_neighbors = np.flatnonzero(self.connectivity[neighbor])
        for ln in local_neighbors:
            if ln not in neighbors:
                self._find_neighbors_recursively(ln, neighbors, conn)
        return neighbors

    def rotate_along_bond(self, bond, angle):
        coor = self.coor
        atoms_to_rotate = self.atoms_to_rotate(bond)
        origin = coor[bond[0]]
        end = coor[bond[1]]
        axis = end - origin
        axis /= np.linalg.norm(axis)

        coor = coor[atoms_to_rotate]
        coor -= origin
        rotmat = aa_to_rotmat(axis, angle)
        selection = self._selection[atoms_to_rotate]
        self._coor[selection] = coor.dot(rotmat.T) + origin

    def rotation_order(self, root):

        def _rotation_order(clusters, checked_clusters, atom, bonds, checked_bonds, tree):
            # Find cluster to which atom belongs to
            for cluster in clusters:
                if atom in cluster:
                    break
            if cluster in checked_clusters:
                return
            checked_clusters.append(cluster)
            # Get all neighboring atoms of the cluster
            neighbors = []
            for atom in cluster:
                neighbors.extend(np.flatnonzero(self.connectivity[atom]))

            for n in neighbors:
                # Find the cluster to which the neighbor belongs to
                for ncluster in clusters:
                    if n in ncluster:
                        break
                if ncluster == cluster:
                    continue
                for b in bonds:
                    # Check if bond is between the current and neighboring cluster
                    if b[0] in cluster and b[1] in ncluster:
                        bond = tuple(b)
                    elif b[1] in cluster and b[0] in ncluster:
                        bond = b[::-1]
                    else:
                        continue
                    # We dont want to go back, so make sure the backward bond
                    # is not already checked.
                    reversed_bond = bond[::-1]
                    if reversed_bond in checked_bonds:
                        continue
                    tree[bond] = {}
                    checked_bonds.append(bond)
                    _rotation_order(clusters, checked_clusters, bond[1],
                                    bonds, checked_bonds, tree[bond])

        rotation_tree = {}
        clusters = self.rigid_clusters()
        bonds = self.rotatable_bonds()
        checked_clusters = []
        checked_bonds = []
        _rotation_order(clusters, checked_clusters, root, bonds, checked_bonds, rotation_tree)
        return rotation_tree
