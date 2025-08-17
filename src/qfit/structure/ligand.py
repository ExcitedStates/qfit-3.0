from abc import abstractmethod
from itertools import product
import logging

import numpy as np
from scipy.spatial.distance import pdist, squareform
from mmtbx.chemical_components import cif_parser

from .base_structure import BaseMonomer

CIF_KEY_BOND = "_chem_comp_bond"
BOND_TYPE_SINGLE = {"SING", "SINGLE"}


class CIFParserError(ValueError):
    ...


class BaseLigand(BaseMonomer):
    """
    Base functions shared by all ligand classes
    """

    def __init__(self, *args, **kwargs):
        self._connectivity = []
        super().__init__(*args, **kwargs)
        self.id = self.resi[0], self.icode[0]

    @property
    def ligand_name(self):
        return self.resname

    @staticmethod
    @abstractmethod
    def from_structure(structure_ligand, cif_file=None):
        ...

    @property
    @abstractmethod
    def shortcode(self) -> str:
        ...

    @property
    def connectivity(self):
        return self._connectivity

    @abstractmethod
    def clashes(self) -> bool:
        ...

    def get_bonds(self):
        bonds = []
        indices = np.nonzero(self.connectivity)
        for a, b in zip(*indices):
            bonds.append([self.name[a], self.name[b]])
        return bonds

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.resname}. Number of atoms: {self.natoms}."

    def _convert_rotation_tree_to_list(self, parent_tree):
        bond_list = []
        for bond, child_trees in parent_tree.items():
            bond_list += [bond]
            if child_trees:
                bond_list += self._convert_rotation_tree_to_list(child_trees)
        return bond_list


class Ligand(BaseLigand):

    """Ligand class automatically generates a topology on the structure."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nbonds = None
        if kwargs.get("cif_file"):
            self._get_connectivity_from_cif(kwargs["cif_file"])
        else:
            self._get_connectivity()

    @staticmethod
    def from_structure(structure_ligand, cif_file=None):
        return Ligand(
            structure_ligand._pdb_hierarchy,  # pylint: disable=protected-access
            selection=structure_ligand.selection,
            link_data=structure_ligand.link_data,
            cif_file=cif_file,
            crystal_symmetry=structure_ligand.crystal_symmetry,
            atoms=structure_ligand._atoms,  # pylint: disable=protected-access
        )

    @property
    def shortcode(self):
        resi, icode = self.id
        shortcode = f"{resi}"
        if icode:
            shortcode += f"_{icode}"
        return shortcode

    def _initialize_connectivity_matrices(self):
        dist_matrix = squareform(pdist(self.coor))
        covrad = self.covalent_radius
        natoms = self.natoms
        cutoff_matrix = np.repeat(covrad, natoms).reshape(natoms, natoms)
        return (dist_matrix, cutoff_matrix)

    def _get_connectivity(self):
        """Determine connectivity matrix of ligand and associated distance
        cutoff matrix for later clash detection.
        """
        dist_matrix, cutoff_matrix = self._initialize_connectivity_matrices()
        # Add 0.5 A to give covalently bound atoms more room
        cutoff_matrix = cutoff_matrix + cutoff_matrix.T + 0.5
        connectivity_matrix = dist_matrix < cutoff_matrix
        # Atoms are not connected to themselves
        np.fill_diagonal(connectivity_matrix, False)
        self._connectivity = connectivity_matrix
        self._cutoff_matrix = cutoff_matrix

    def _get_connectivity_from_cif(self, cif_file):
        """
        Determine connectivity matrix of ligand and associated distance
        cutoff matrix for later clash detection.
        """
        coor = self.coor
        dist_matrix = squareform(pdist(coor))
        covrad = self.covalent_radius
        natoms = self.natoms
        cutoff_matrix = np.repeat(covrad, natoms).reshape(natoms, natoms)
        connectivity_matrix = np.zeros_like(dist_matrix, dtype=bool)
        self.bond_types = {}
        cif = cif_parser.run(cif_file)
        if not CIF_KEY_BOND in cif.keys():
            raise CIFParserError(f"Can't find {CIF_KEY_BOND} in CIF file {cif_file}; only Chemical Components and/or Refmac Monomer Library CIF files are accepted")
        for cif_bond in cif[CIF_KEY_BOND]:
            if cif_bond.comp_id == self.ligand_name:
                a1 = cif_bond.atom_id_1
                a2 = cif_bond.atom_id_2
                index1 = np.argwhere(self.name == a1)
                index2 = np.argwhere(self.name == a2)
                try:
                    connectivity_matrix[index1, index2] = True
                    connectivity_matrix[index2, index1] = True
                except IndexError as e:
                    if not (a1.startswith("H") or a2.startswith("H")):
                        logging.warning(f"Can't find atoms for {cif_bond}: {e}")
                    continue
                else:
                    try:
                        index1 = index1[0, 0]
                        index2 = index2[0, 0]
                    except IndexError as e:
                        logging.warning(f"Can't find atoms for {cif_bond}: {e}")
                        continue
                    if index1 not in self.bond_types:
                        self.bond_types[index1] = {}
                    if index2 not in self.bond_types:
                        self.bond_types[index2] = {}
                    # chemical components
                    if "value_order" in cif_bond.__dict__.keys():
                        bond_type = cif_bond.value_order.upper()
                    # refmac monomer library
                    elif "type" in cif_bond.__dict__.keys():
                        bond_type = cif_bond.type.upper()
                    # TODO what about Phenix?
                    else:
                        logging.warning(f"Can't determine bond type for {cif_bond}")
                        bond_type = "SINGLE"
                    self.bond_types[index1][index2] = bond_type
                    self.bond_types[index2][index1] = bond_type

        self._cutoff_matrix = cutoff_matrix
        self._connectivity = connectivity_matrix

    def clashes(self) -> bool:
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

    # XXX unused here, can we delete it?
    def _rotation_order(self, root):
        def _rotation_order_recursive(
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
                    _rotation_order_recursive(
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
        _rotation_order_recursive(
            clusters, checked_clusters, root, bonds, checked_bonds, rotation_tree
        )
        return rotation_tree

    def _convert_rotation_tree_to_list(self, parent_tree):
        bond_list = []
        for bond, child_trees in parent_tree.items():
            bond_list += [bond]
            if child_trees:
                bond_list += self._convert_rotation_tree_to_list(child_trees)
        return bond_list

    # Internal method, preserving some logic from the old CovalentLigand
    # class that is only partially implemented here
    def is_single_bond(self, id1, id2):
        try:
            return self.bond_types[id1][id2] in BOND_TYPE_SINGLE
        except KeyError:
            return True


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
                bond = (None, None)
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
