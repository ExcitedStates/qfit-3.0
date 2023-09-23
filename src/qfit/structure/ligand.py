from itertools import product

import numpy as np
from scipy.spatial.distance import pdist, squareform

from .base_structure import BaseStructure
from .mmCIF import mmCIFDictionary


class Ligand(BaseStructure):

    """Ligand class automatically generates a topology on the structure."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.id = (kwargs["resi"], kwargs["icode"])
        except KeyError:
            self.id = (args[0]["resi"], args[0]["icode"])
            self.ligand_name = self.resn[0]
        self.nbonds = None

        try:
            self.type = kwargs["type"]
        except:
            pass

        if "cif_file" in kwargs:
            self._get_connectivity_from_cif(kwargs["cif_file"])
        else:
            self._get_connectivity()

        # self.root = np.argwhere(self.name == self.link_data['name1'][i])
        # self.order = self.rotation_order(self.root)
        # self.bond_list = self.convert_rotation_tree_to_list(self.order)

    def __repr__(self):
        string = "Ligand: {}. Number of atoms: {}.".format(self.resn[0], self.natoms)
        return string

    @property
    def shortcode(self):
        resi, icode = self.id
        shortcode = f"{resi}"
        if icode:
            shortcode += f"_{icode}"

        return shortcode

    def _get_connectivity(self):
        """Determine connectivity matrix of ligand and associated distance
        cutoff matrix for later clash detection.
        """
        coor = self.coor
        # dm_size = self.natoms * (self.natoms - 1) // 2
        # dm = np.zeros(dm_size, np.float64)
        # k = 0
        # for i in range(0, self.natoms - 1):
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
        connectivity_matrix = dist_matrix < cutoff_matrix
        # Atoms are not connected to themselves
        np.fill_diagonal(connectivity_matrix, False)
        self.connectivity = connectivity_matrix
        self._cutoff_matrix = cutoff_matrix

    def _get_connectivity_from_cif(self, cif_file):
        """Determine connectivity matrix of ligand and associated distance
        cutoff matrix for later clash detection.
        """
        coor = self.coor
        self.bond_types = {}
        dist_matrix = squareform(pdist(coor))
        covrad = self.covalent_radius
        natoms = self.natoms
        cutoff_matrix = np.repeat(covrad, natoms).reshape(natoms, natoms)
        connectivity_matrix = np.zeros_like(dist_matrix, dtype=bool)
        cif = mmCIFDictionary()
        cif.load_file(cif_file)
        for cif_data in cif:
            if cif_data.name == f"comp_{self.ligand_name}":
                for cif_table in cif_data:
                    if cif_table.name == "chem_comp_bond":
                        for cif_row in cif_table:
                            a1 = cif_row["atom_id_1"]
                            a2 = cif_row["atom_id_2"]
                            index1 = np.argwhere(self.name == a1)
                            index2 = np.argwhere(self.name == a2)
                            try:
                                connectivity_matrix[index1, index2] = True
                                connectivity_matrix[index2, index1] = True
                            except:
                                pass
                            else:
                                try:
                                    index1 = index1[0, 0]
                                    index2 = index2[0, 0]
                                except:
                                    continue
                                if index1 not in self.bond_types:
                                    self.bond_types[index1] = {}
                                if index2 not in self.bond_types:
                                    self.bond_types[index2] = {}
                                self.bond_types[index1][index2] = cif_row["type"]
                                self.bond_types[index2][index1] = cif_row["type"]

        self._cutoff_matrix = cutoff_matrix
        self.connectivity = connectivity_matrix

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

    def get_bonds(self):
        bonds = []
        indices = np.nonzero(self.connectivity)
        for a, b in zip(*indices):
            bonds.append([self.name[a], self.name[b]])
        return bonds

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
            ring = v1path[0 : v1path.index(v) + 1] + v2path
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
            if element == "H":
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
                    if self.e[n] == "H":
                        continue
                    neighbor_neighbors = np.flatnonzero(conn[n])
                    # Hydrogen neighbors don't count
                    hydrogen_neighbors = (self.e[neighbor_neighbors] == "H").sum()
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
                self._find_neighbors_recursively(ln, neighbors)
        return neighbors

    def rotation_order(self, root):
        def _rotation_order(
            clusters, checked_clusters, atom, bonds, checked_bonds, tree
        ):
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
                    _rotation_order(
                        clusters,
                        checked_clusters,
                        bond[1],
                        bonds,
                        checked_bonds,
                        tree[bond],
                    )

        rotation_tree = {}
        clusters = self.rigid_clusters()
        bonds = self.rotatable_bonds()
        checked_clusters = []
        checked_bonds = []
        _rotation_order(
            clusters, checked_clusters, root, bonds, checked_bonds, rotation_tree
        )
        return rotation_tree

    def convert_rotation_tree_to_list(self, parent_tree):
        bond_list = []
        for bond, child_trees in parent_tree.items():
            bond_list += [bond]
            if child_trees:
                bond_list += self.convert_rotation_tree_to_list(child_trees)
        return bond_list


class BondOrder(object):

    """Determine bond rotation order given a ligand and root."""

    def __init__(self, ligand, atom):
        self.ligand = ligand
        self._conn = self.ligand.connectivity
        self.clusters = self.ligand.rigid_clusters()
        self.bonds = self.ligand.rotatable_bonds()
        self._checked_clusters = []
        self.order = []
        self.depth = []
        self._bondorder(atom)

    def _bondorder(self, atom, depth=0):
        for cluster in self.clusters:
            if atom in cluster:
                break
        if cluster in self._checked_clusters:
            return
        depth += 1
        self._checked_clusters.append(cluster)
        neighbors = []
        for atom in cluster:
            neighbors += np.flatnonzero(self._conn[atom]).tolist()

        for n in neighbors:
            for ncluster in self.clusters:
                if n in ncluster:
                    break
            if ncluster == cluster:
                continue
            for b in self.bonds:
                if b[0] in cluster and b[1] in ncluster:
                    bond = (b[0], b[1])
                elif b[1] in cluster and b[0] in ncluster:
                    bond = (b[1], b[0])
                try:
                    if (bond[1], bond[0]) not in self.order and bond not in self.order:
                        self.order.append(bond)
                        self.depth.append(depth)
                except UnboundLocalError:
                    pass
            self._bondorder(n, depth)


class CovalentLigand(BaseStructure):
    """Covalent Ligand class"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = (args[0]["resi"], args[0]["icode"])
        self.ligand_name = self.resn[0]
        self.nbonds = None
        self.covalent_bonds = 0
        self.covalent_partners = []
        self.covalent_atoms = []
        self.bond_types = {}

        if "cif_file" in kwargs:
            self._get_connectivity_from_cif(kwargs["cif_file"])
        else:
            self._get_connectivity()

        for i, res1 in enumerate(self.link_data["resn1"]):
            if (
                res1 == self.ligand_name
                and self.chain[0] == self.link_data["chain1"][i]
            ):
                self.covalent_bonds += 1
                self.covalent_partners.append(
                    [
                        self.link_data["chain2"][i],
                        self.link_data["resi2"][i],
                        self.link_data["icode2"][i],
                        self.link_data["name2"][i],
                    ]
                )
                self.covalent_atoms.append(
                    [
                        self.link_data["chain1"][i],
                        self.link_data["resi1"][i],
                        self.link_data["icode1"][i],
                        self.link_data["name1"][i],
                    ]
                )
                self.root = np.argwhere(self.name == self.link_data["name1"][i])
                self.order = self.rotation_order(self.root)
                self.bond_list = self.convert_rotation_tree_to_list(self.order)
        # self.type = args[0].data["type"]

    def __repr__(self):
        string = f"Covalent Ligand: {self.resn[0]}." f" Number of atoms: {self.natoms}."
        return string

    @property
    def _identifier_tuple(self):
        """Returns (chain, resi, icode) to identify this covalent ligand."""
        chainid = self.chain[0]
        resi, icode = self.id

        return (chainid, resi, icode)

    @property
    def shortcode(self):
        (chainid, resi, icode) = self._identifier_tuple
        shortcode = f"{chainid}_{resi}"
        if icode:
            shortcode += f"_{icode}"

        return shortcode

    def _get_connectivity_from_cif(self, cif_file):
        """Determine connectivity matrix of ligand and associated distance
        cutoff matrix for later clash detection.
        """
        coor = self.coor
        self.bond_types = {}
        dist_matrix = squareform(pdist(coor))
        covrad = self.covalent_radius
        natoms = self.natoms
        cutoff_matrix = np.repeat(covrad, natoms).reshape(natoms, natoms)
        connectivity_matrix = np.zeros_like(dist_matrix, dtype=bool)
        cif = mmCIFDictionary()
        cif.load_file(cif_file)
        for cif_data in cif:
            if cif_data.name == f"comp_{self.ligand_name}":
                for cif_table in cif_data:
                    if cif_table.name == "chem_comp_bond":
                        for cif_row in cif_table:
                            a1 = cif_row["atom_id_1"]
                            a2 = cif_row["atom_id_2"]
                            index1 = np.argwhere(self.name == a1)
                            index2 = np.argwhere(self.name == a2)
                            try:
                                connectivity_matrix[index1, index2] = True
                                connectivity_matrix[index2, index1] = True
                            except:
                                pass
                            else:
                                try:
                                    index1 = index1[0, 0]
                                    index2 = index2[0, 0]
                                except:
                                    continue
                                if index1 not in self.bond_types:
                                    self.bond_types[index1] = {}
                                if index2 not in self.bond_types:
                                    self.bond_types[index2] = {}
                                self.bond_types[index1][index2] = cif_row["type"]
                                self.bond_types[index2][index1] = cif_row["type"]

        self._cutoff_matrix = cutoff_matrix
        self.connectivity = connectivity_matrix

    def _get_connectivity(self):
        coor = self.coor
        dist_matrix = squareform(pdist(coor))
        covrad = self.covalent_radius
        natoms = self.natoms
        cutoff_matrix = np.repeat(covrad, natoms).reshape(natoms, natoms)
        # Add 0.5 A to give covalently bound atoms more room
        cutoff_matrix = cutoff_matrix + cutoff_matrix.T + 0.5
        connectivity_matrix = dist_matrix < cutoff_matrix
        # Atoms are not connected to themselves
        np.fill_diagonal(connectivity_matrix, False)
        self.connectivity = connectivity_matrix
        self._cutoff_matrix = cutoff_matrix

    def clashes(self):
        """Checks if there are any internal clashes."""
        """ dist_matrix = squareform(pdist(self.coor))
        mask = np.logical_not(self.connectivity)
        active_matrix = (self.active.reshape(1, -1) * self.active.reshape(-1, 1)) > 0
        mask &= active_matrix
        np.fill_diagonal(mask, False)
        clash_matrix = dist_matrix < self._cutoff_matrix
        if np.any(np.logical_and(clash_matrix, mask)):
            return True
        return False"""
        pass

    def bonds(self):
        """Print bonds"""
        indices = np.nonzero(self.connectivity)
        for a, b in zip(*indices):
            print(self.name[a], self.name[b])

    def get_bonds(self):
        bonds = []
        indices = np.nonzero(self.connectivity)
        for a, b in zip(*indices):
            bonds.append([self.name[a], self.name[b]])
        return bonds

    def rigid_clusters(self):
        """
        Find rigid clusters / seeds in the molecule.
        Currently seeds are either rings or terminal ends of the molecule,
        i.e. the last two atoms.
        """

        conn = self.connectivity
        rings = self.ring_paths()
        clusters = []
        clustered = np.zeros(self.natoms, dtype=int)

        for root in range(self.natoms):
            # Ignore root if it is a Hydrogen
            if self.e[root] == "H":
                continue

            # Check if root has already been clustered
            if clustered[root] == 2:
                continue
            elif clustered[root] == 1:
                for cluster in clusters:
                    if root in cluster:
                        break
            else:
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
                            clustered[a] = 2

            # If root is not part of a ring, check if it is connected to a
            # terminal heavy atom.
            if not ring_atom:
                neighbors = np.flatnonzero(conn[root])
                for n in neighbors:
                    if self.e[n] == "H":
                        continue
                    neighbor_neighbors = np.flatnonzero(conn[n])
                    # Ignore hydrogen neighbors:
                    hydrogen_neighbors = (self.e[neighbor_neighbors] == "H").sum()
                    if len(neighbor_neighbors) - hydrogen_neighbors == 1:
                        if clustered[n] == 0:
                            cluster.append(n)
                            clustered[n] = 2

                    # If bond type was provided via CIF file, check for
                    # double and aromatic bonds:
                    if self.bond_types:
                        if self.bond_types[root][n] != "single" and n not in cluster:
                            cluster.append(n)
                            clustered[n] = 1

            if len(cluster) > 1 and cluster not in clusters:
                clusters.append(cluster)

            clustered[root] = 2

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

    # This method aims to identify atoms that are involved in a ring.
    def ring_paths(self):
        # Call to the BFS:
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
            ring = v1path[0 : v1path.index(v) + 1] + v2path
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
            if len(neighbors) == 1 and atom != self.root:
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
        conn = self.connectivity
        neighbors.append(neighbor)
        local_neighbors = np.flatnonzero(conn[neighbor])
        for ln in local_neighbors:
            if ln not in neighbors:
                self._find_neighbors_recursively(ln, neighbors)
        return neighbors

    def rotation_order(self, root):
        def _rotation_order(
            clusters, checked_clusters, atom, bonds, checked_bonds, tree
        ):
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
                    _rotation_order(
                        clusters,
                        checked_clusters,
                        bond[1],
                        bonds,
                        checked_bonds,
                        tree[bond],
                    )

        rotation_tree = {}
        clusters = self.rigid_clusters()
        bonds = self.rotatable_bonds()
        checked_clusters = []
        checked_bonds = []
        _rotation_order(
            clusters, checked_clusters, root, bonds, checked_bonds, rotation_tree
        )
        return rotation_tree

    def convert_rotation_tree_to_list(self, parent_tree):
        bond_list = []
        for bond, child_trees in parent_tree.items():
            bond_list += [bond]
            if child_trees:
                bond_list += self.convert_rotation_tree_to_list(child_trees)
        return bond_list
