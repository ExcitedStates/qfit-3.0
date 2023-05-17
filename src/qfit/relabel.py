import argparse
import sys
import numpy as np
import itertools as itl
import copy
import tqdm
import logging
from .vdw_radii import vdwRadiiTable, EpsilonTable, EpsilonIndex, EpsilonArray
from .structure import Structure


logger = logging.getLogger(__name__)


def cartesian_product(*arrays):
    la = len(arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=int)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


class RelabellerOptions:
    def __init__(self, nSims=10000, nChains=10):
        self.nSims = nSims
        self.nChains = nChains
        self.random_seed = None

    def apply_command_args(self, args):
        for key, value in vars(args).items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class Relabeller:
    "Relabel alternate conformers using SA"

    def __init__(self, structure, options):
        self.structure = structure
        self.nSims = options.nSims
        self.nChains = options.nChains

        self.prng = np.random.default_rng(options.random_seed)

        self.nodes = []
        self.permutation = []
        self.initNodes()

        self.metric = self.initMetric()

    def initNodes(self):
        node = 0
        segment = []
        for chain in self.structure:
            for residue in chain:
                resInd = []
                for altloc in list(set(residue.altloc)):
                    if altloc != "":
                        self.nodes.append(residue.extract("altloc", altloc))
                        resInd.append(node)
                        node += 1
                self.permutation.append(resInd)

    def initMetric(self):
        # How many calcs are needed?
        n = len(self.nodes)
        n_combos = n * (n + 1) // 2  # triangular number

        # Build metric array
        metric = np.zeros((len(self.nodes), len(self.nodes)))
        node_pairs = itl.combinations_with_replacement(enumerate(self.nodes), r=2)
        with tqdm.tqdm(
            node_pairs, total=n_combos, desc="Pairwise interactions", leave=True
        ) as pbar:
            for (i, node1), (j, node2) in pbar:
                if node1.resi[0] != node2.resi[0] or node1.chain[0] != node2.chain[0]:
                    metric[i, j] = self.pairwise_residue_energy(node1, node2)

        # We have filled in the upper triangle, now fill in the lower
        metric += np.tril(metric.T, k=-1)

        return metric

    @staticmethod
    def pairwise_residue_energy(node1, node2):
        INTERACTION_DISTANCE_CUTOFF = 16.0
        BACKBONE_ATOMS = ["N", "CA", "C", "O", "H", "HA"]

        # distance
        dist_ij_x = node1.coor[:, np.newaxis] - node2.coor[np.newaxis, :]
        dist_ij = np.linalg.norm(dist_ij_x, axis=-1)

        # Only proceed if at least one interatomic distance is within cutoff
        if np.all(dist_ij >= INTERACTION_DISTANCE_CUTOFF):
            return 0.0

        # epsilon
        atom1_epsilon_index = np.array([EpsilonIndex.index(e) for e in node1.e])[
            :, np.newaxis
        ]
        atom2_epsilon_index = np.array([EpsilonIndex.index(e) for e in node2.e])[
            np.newaxis, :
        ]
        epsilon_ij = np.array(EpsilonArray)[atom1_epsilon_index, atom2_epsilon_index]

        # radii
        atom1_radius = np.array([vdwRadiiTable[e] for e in node1.e])[:, np.newaxis]
        atom2_radius = np.array([vdwRadiiTable[e] for e in node2.e])[np.newaxis, :]
        sigma_ij = (atom1_radius + atom2_radius) / 1.122

        # energy
        sigma_dist_ij = sigma_ij / dist_ij
        energy_ij = (
            4 * epsilon_ij * (np.power(sigma_dist_ij, 12) - np.power(sigma_dist_ij, 6))
        )

        # We must ignore energies from two backbone atoms in neighbouring residues
        is_neighbouring_resi = abs(node1.resi[0] - node2.resi[0]) == 1
        atom1_in_backbone = np.isin(node1.name, BACKBONE_ATOMS, assume_unique=True)[
            :, np.newaxis
        ]
        atom2_in_backbone = np.isin(node2.name, BACKBONE_ATOMS, assume_unique=True)[
            np.newaxis, :
        ]

        energy_ij = np.where(
            is_neighbouring_resi & atom1_in_backbone & atom2_in_backbone, 0.0, energy_ij
        )

        return np.sum(energy_ij)

    def SimulatedAnnealing(self, permutation):
        energyList = []
        NumOfClusters = len(max(permutation, key=len))
        energies = np.zeros(NumOfClusters)
        clusters = [[] for x in range(NumOfClusters)]

        # Use the permutation to identify the clusters:
        for i, elem in enumerate(permutation):
            for j in range(len(elem)):
                clusters[j].append(elem[j])

        # Calculate the energy of each cluster:
        for i, cluster in enumerate(clusters):
            b = cartesian_product(cluster, cluster)
            energies[i] = np.sum(self.metric[b[:, 0], b[:, 1]]) / 2

        # Sum the energy of each cluster:
        energyList.append(np.sum(energies))
        logger.debug(f"Starting energy: {energyList[-1]}")

        for i in tqdm.trange(
            self.nSims,
            unit="sims",
            desc="Annealing progress",
            unit_scale=True,
            leave=False,
            miniters=1,
        ):
            temperature = 273 * (1 - i / self.nSims)
            # Perturb the current solution:
            tmpPerm = copy.deepcopy(permutation)
            # Choose five random indices of residues to swap:
            for x in self.prng.integers(len(tmpPerm), size=5):
                # If the residue has a single conformer:
                if not tmpPerm[x]:
                    continue
                # Identify the indexes of the segment the residue belongs to:
                l_index = x - 1
                while l_index >= 0:
                    if not tmpPerm[l_index]:
                        break
                    if len(tmpPerm[l_index]) != len(tmpPerm[x]):
                        break
                    occ1 = np.min(self.nodes[tmpPerm[l_index][0]].q)
                    occ2 = np.min(self.nodes[tmpPerm[x][0]].q)
                    if occ1 != occ2 or occ1 > 0.95:
                        break
                    l_index -= 1
                l_index += 1
                u_index = x + 1
                while u_index < len(tmpPerm):
                    if not tmpPerm[u_index]:
                        break
                    if len(tmpPerm[u_index]) != len(tmpPerm[x]):
                        break
                    occ1 = np.min(self.nodes[tmpPerm[u_index][-1]].q)
                    occ2 = np.min(self.nodes[tmpPerm[x][-1]].q)
                    if occ1 != occ2 or occ1 > 0.95:
                        break
                    u_index += 1
                u_index -= 1
                ordering = list(range(len(tmpPerm[x])))
                self.prng.shuffle(ordering)
                for counter in range(l_index, u_index + 1):
                    tmpPerm[counter] = [tmpPerm[counter][li] for li in ordering]
                # self.prng.shuffle(tmpPerm[x])

            # Calculate the new clusters:
            tmpCluster = [[] for x in clusters]
            for ii, value in enumerate(tmpPerm):
                for j in range(len(value)):
                    tmpCluster[j].append(value[j])

            # Calculate the energy across all clusters
            tmpEnergies = np.zeros_like(energies)
            for i, cluster in enumerate(tmpCluster):
                b = cartesian_product(cluster, cluster)
                tmpEnergies[i] = np.sum(self.metric[b[:, 0], b[:, 1]]) / 2

            # If the new energy is better than the old one:
            if (
                np.sum(energies) >= np.sum(tmpEnergies)
                or np.exp(-(np.sum(tmpEnergies) - np.sum(energies)) / temperature)
                >= self.prng.uniform()
            ):
                energies = tmpEnergies[:]
                clusters = tmpCluster[:]
                permutation = copy.deepcopy(tmpPerm)
                energyList.append(np.sum(energies))

        logger.debug(f"Final locally optimal energy: {energyList[-1]}")
        return energyList[-1], permutation

    def run(self):
        perm = []
        energyList = []

        for i in tqdm.trange(
            self.nChains, unit="runs", desc="SA macrocycle", unit_scale=True, leave=True
        ):
            energy, permutation = self.SimulatedAnnealing(self.permutation)
            energyList.append(energy)
            perm.append(permutation)

        minIdx = energyList.index(min(energyList))
        # Relabel the residues:
        Altlocs = ["A", "B", "C", "D", "E"]
        node = 0
        res = 0
        for chain in self.structure:
            for residue in chain:
                tmpAltlocs = copy.deepcopy(residue.altloc)
                for altloc in list(set(residue.altloc)):
                    if altloc == "":
                        continue
                    else:
                        Idx = perm[minIdx][res].index(node)
                        new_altloc = Altlocs[Idx]
                        mask = residue.altloc == altloc
                        tmpAltlocs[mask] = new_altloc
                    node += 1
                residue.altloc = copy.deepcopy(tmpAltlocs)
                res += 1
        self.structure.reorder()
        # self.structure.tofile("Test_relabel.pdb")
        return self.structure


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("structure", type=str, help="PDB-file containing structure.")

    # MCMC options
    p.add_argument(
        "-S",
        "--number-of-sims",
        type=int,
        dest="nSims",
        default=10000,
        metavar="<int>",
        help="Number of simulations for MCMC.",
    )
    p.add_argument(
        "-NC",
        "--number-of-chains",
        type=int,
        dest="nChains",
        default=10,
        metavar="<int>",
        help="Number of chains for MCMC.",
    )

    # Global options
    p.add_argument(
        "--random-seed",
        dest="random_seed",
        metavar="<int>",
        type=int,
        help="Seed value for PRNG",
    )

    args = p.parse_args()
    return args


def main():
    args = parse_args()

    # Extract residue and prepare it
    structure = Structure.fromfile(args.structure).reorder()

    options = RelabellerOptions()
    options.apply_command_args(args)

    relabeller = Relabeller(structure, options)
    relabeller.run()


if __name__ == "__main__":
    main()
