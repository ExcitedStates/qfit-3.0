import itertools
import logging
from pathlib import Path
logger = logging.getLogger(__name__)

import numpy as np

from .structure import Ligand, BondOrder
from .volume import Volume
from .samplers import GlobalRotator, Translator, BondRotator, RotationSets, ClashDetector
from .transformer import Transformer
from .solvers import QPSolver, MIQPSolver
from .helpers import DJoiner


class IterativeBuilderOptions:

    def __init__(self, args=None):

        self.no_local = False
        self.build_stepsize = 2
        self.stepsize = 6
        self.scaling_factor = 0.75
        # MIQP options
        self.threshold = 0.01
        self.cardinality = 5
        # General options
        self.directory = Path()
        self.debug = False


class IterativeBuilder:

    QP_CUTOFF = 0.02

    def __init__(self, xmap, ligand, receptor, options):

        self.conformers = []
        self.ligand = ligand
        self.options = options
        self.receptor = receptor
        self.xmap = xmap

        # Setup clash detection
        self._cd = ClashDetector(self.ligand, self.receptor,
                                 scaling_factor=self.options.scaling_factor)
        if self._cd():
            logger.warning("Initial ligand configuration is clashing.")

        # Setup density generation
        resolution = self.xmap.resolution
        if resolution < 3.0:
            self._radius_mask = resolution / 3.0 + 0.5
        else:
            self._radius_mask = resolution / 2.0
        smax = 1 / (2 * resolution)
        self._model_map = Volume.zeros_like(self.xmap)
        # Initialize density creation
        self._model_map.set_spacegroup("P1")
        self._transformer = Transformer(self.ligand, self._model_map,
                                        smax=smax, rmax=3)
        # Setup attributes and starting coordinates
        self._starting_coor_set = [self.ligand.coor.copy()]
        self._coor_set = []
        self._all_coor_set = []
        self._occupancies = []

    def __call__(self, root):

        # Get starting cluster
        self._cluster = self._get_starting_cluster(root)
        self._local_search()
        self._build()


    def _build(self):
        pass

    def _convert(self):
        logger.info('Converting structures to densities ({:})'.format(len(self._coor_set)))
        self._transformer.volume.array.fill(0)
        for coor in self._coor_set:
            self.ligand.coor[:] = coor
            self._transformer.mask(self._radius_mask)
        mask = self._transformer.volume.array > 0
        self._transformer.volume.array.fill(0)

        nvalues = mask.sum()
        self._target = self.xmap.array[mask]
        self._models = np.zeros((len(self._coor_set), nvalues), dtype=np.float64)
        for n, coor in enumerate(self._coor_set):
            self.ligand.coor[:] = coor
            self._transformer.density()
            self._models[n] = self._transformer.volume.array[mask]
            self._transformer.reset()

    def _get_starting_cluster(self):
        pass

    def _local_search(self):
        pass

    def miqp(self):
        self._convert()
        logger.info("Starting MIQP.")
        miqpsolver = MIQPSolver(self._target, self._models)
        logger.info("Initializing.")
        miqpsolver.initialize()
        logger.info("Solving")
        miqpsolver(maxfits=self.options.cardinality, threshold=self.options.threshold)
        self._occupancies = miqpsolver.occupancies
        self._update_conformers()

    def qp(self):
        self._convert()
        logger.info("Starting QP.")
        qpsolver = QPSolver(self._target, self._models)
        logger.info("Initializing.")
        qpsolver.initialize()
        logger.info("Solving")
        qpsolver()
        self._occupancies = qpsolver.occupancies
        self._update_conformers()

    def set_conformers(self, coor_set):
        pass

    def _update_conformers(self):
        logger.info("Updating conformer list.")
        logger.info("Old number of conformers: {:d}".format(len(self._coor_set)))
        new_coor_set = []
        for n, coor in enumerate(self._coor_set):
            if self._occupancies[n] >= self.QP_CUTOFF:
                new_coor_set.append(coor)
        if new_coor_set:
            self._coor_set = new_coor_set
            self._occupancies = self._occupancies[self._occupancies >= self.QP_CUTOFF]
        else:
            logger.warning("No conformer found with occupancy bigger than {:}. Taking conformers with highest occupancy".format(self.QP_CUTOFF))
            sorted_indices = np.argsort(self._occupancies)[::-1]
            indices = np.arange(len(self._occupancies))[sorted_indices]
            nbest = min(self.options.cardinality, len(self._occupancies))
            for i in indices[:nbest]:
                new_coor_set.append(self._coor_set[i])
            self._coor_set = new_coor_set
            self._occupancies = self._occupancies[indices[:nbest]]
        logger.info("New number of conformers: {:d}".format(len(self._coor_set)))


class HierarchicalBuilder:

    """Build a multi-conformer ligand hierarchically."""

    def __init__(self, ligand, xmap, resolution, receptor=None,
            local_search=True, build=True,
            stepsize=2, build_stepsize=1,
            threshold=None, cardinality=5,
            directory='.', roots=None, debug=False):
        self.ligand = ligand
        self.xmap = xmap
        self.resolution = resolution
        self.build = build
        self.stepsize= stepsize
        self.build_stepsize = build_stepsize
        self.directory = directory
        self.local_search = local_search
        self.receptor = receptor
        if self.resolution < 3.0:
            self._rmask = 0.5 + self.resolution / 3.0
        else:
            self._rmask = 0.5 * self.resolution
        self._debug = debug

        # For MIQP
        self.threshold = threshold
        self.cardinality = cardinality

        self._trans_box = [(-0.2, 0.21, 0.1)] * 3
        self._sampling_range = np.deg2rad(np.arange(0, 360, self.stepsize))
        self._djoiner = DJoiner(directory)

        if self.receptor is not None:
            self._cd = ClashDetector(self.ligand, self.receptor, scaling_factor=0.75)
            if self._cd():
                logger.warning("Initial ligand configuration is clashing!")

        self._rigid_clusters = self.ligand.rigid_clusters()
        # Determine which roots to start building from
        if roots is None:
            self._clusters_to_sample = []
            for cluster in self._rigid_clusters:
                nhydrogen = (self.ligand.e[cluster] == 'H').sum()
                if len(cluster) - nhydrogen > 1:
                    self._clusters_to_sample.append(cluster)
        else:
            self._clusters_to_sample = []
            for root in roots:
                for cluster in self._rigid_clusters:
                    if root in cluster:
                        self._clusters_to_sample.append(cluster)
        msg = "Number of clusters to sample: {:}".format(len(self._clusters_to_sample))
        logger.info(msg)
        self._starting_coor_set = [ligand.coor.copy()]
        self._coor_set = []
        self._all_coor_set = []
        self.conformers = []
        self._occupancies = []

        smax = 1 / (2 * self.resolution)
        self._model_map = Volume.zeros_like(self.xmap)
        # Initialize density creation
        # We can let go of the spacegroup now that we have prepared the map.
        self._model_map.set_spacegroup("P1")
        self._transformer = Transformer(self.ligand, self._model_map, smax=smax, rmax=3)

    def __call__(self):

        self._all_occupancies = []
        if self.build:
            for self._cluster_index, self._cluster in enumerate(self._clusters_to_sample):
                self._iteration = 0
                self._coor_set = list(self._starting_coor_set)
                logger.info("Cluster index: {:}".format(self._cluster_index))
                logger.info("Number of conformers: {:}".format(len(self._coor_set)))
                if self.local_search:
                    self._local_search()
                self._build_ligand()

                self._all_coor_set += self._coor_set
                self._all_occupancies += list(self._occupancies)
                logger.info("Number of conformers: {:}".format(len(self._coor_set)))
                logger.info("Number of final conformers: {:}".format(len(self._all_coor_set)))

        self._coor_set = self._all_coor_set
        self._occupancies = np.asarray(self._all_occupancies)
        self._convert()

    def _clashing(self):
        if self.receptor is None:
            return self.ligand.clashes()
        else:
            return self.ligand.clashes() or self._cd()

    def _local_search(self):
        """Perform a local rigid body search on the cluster."""

        logger.info("Performing local search.")

        # Set occupancies of rigid cluster and its direct neighboring atoms to
        # 1 for clash detection and MIQP
        self.ligand.q.fill(0)
        self.ligand.q[self._cluster] = 1
        for atom in self._cluster:
            self.ligand.q[self.ligand.connectivity[atom]] = 1
        center = self.ligand.coor[self._cluster].mean(axis=0)
        new_coor_set = []
        for coor in self._coor_set:
            self.ligand.coor[:] = coor
            rotator = GlobalRotator(self.ligand, center=center)
            for rotmat in RotationSets.get_local_set():
                rotator(rotmat)
                translator = Translator(self.ligand)
                iterator = itertools.product(*[
                    np.arange(*trans) for trans in self._trans_box])
                for translation in iterator:
                        translator(translation)
                        if not self._clashing():
                            new_coor_set.append(self.ligand.coor.copy())
        self._coor_set = new_coor_set
        # In case all conformers were clashing
        if not self._coor_set:
            return
        if self._debug:
            self._write_intermediate_structures(base='intermediate')
        self._convert()
        self._QP()
        self._update_conformers()
        if self._debug:
            self._write_intermediate_structures(base='qp')
        self._convert()
        self._MIQP()
        self._update_conformers()
        if self._debug:
            self._write_intermediate_structures(base='miqp')

    def _build_ligand(self):
        """Build up the ligand hierarchically."""

        logger.info("Building up ligand.")
        # Sampling order of bonds
        bond_order = BondOrder(self.ligand, self._cluster[0])
        bonds = bond_order.order
        depths = bond_order.depth
        nbonds = len(bonds)
        logger.info("Number of bonds to sample: {:d}".format(nbonds))
        starting_bond_index = 0
        finished_building = True if nbonds == 0 else False
        while not finished_building:
            end_bond_index = min(starting_bond_index + self.build_stepsize, nbonds)
            logger.info("Sampling iteration: {:d}".format(self._iteration))
            for bond_index in range(starting_bond_index, end_bond_index):
                # Set the occupancies of build clusters and their direct
                # neighbors to 1 for clash detection and MIQP.
                nbonds_sampled = bond_index + 1
                self.ligand.q.fill(0)
                for cluster in self._rigid_clusters:
                    for sampled_bond in bonds[:nbonds_sampled]:
                        if sampled_bond[0] in cluster or sampled_bond[1] in cluster:
                            self.ligand.q[cluster] = 1
                            for atom in cluster:
                                self.ligand.q[self.ligand.connectivity[atom]] = 1

                # Sample by rotating around a bond
                bond = bonds[bond_index]
                atoms = [self.ligand.atomname[bond[0]], self.ligand.atomname[bond[1]]]
                new_coor_set = []
                for coor in self._coor_set:
                    self.ligand.coor[:] = coor
                    rotator = BondRotator(self.ligand, *atoms)
                    for angle in self._sampling_range:
                        rotator(angle)
                        if not self._clashing():
                            new_coor_set.append(self.ligand.coor.copy())
                self._coor_set = new_coor_set
                # Check if any acceptable configurations have been created.
                if not self._coor_set:
                    msg = "No non-clashing conformers found. Stop building."
                    logger.warning(msg)
                    return

                # Perform an MIQP if either the end bond index has been reached
                # or if the end of a sidechain has been reached, i.e. if the
                # next depth level is equal or smaller than current depth. If
                # we step out of bonds in the depths list it means we are done.
                end_iteration = (nbonds_sampled == end_bond_index)
                try:
                    end_sidechain = depths[bond_index] >= depths[bond_index + 1]
                except IndexError:
                    finished_building = True
                    end_sidechain = True
                if end_iteration or end_sidechain:
                    self._iteration += 1
                    if self._debug:
                        self._write_intermediate_structures()
                    self._convert()
                    self._QP()
                    self._update_conformers()
                    if self._debug:
                        self._write_intermediate_structures('qp')
                    self._convert()
                    self._MIQP()
                    self._update_conformers()
                    if self._debug:
                        self._write_intermediate_structures('miqp')

                    # Stop this building iteration and move on to next
                    starting_bond_index += 1
                    if end_sidechain:
                        starting_bond_index = nbonds_sampled
                    break

    def _convert(self, tofile=False):

        logger.info('Converting structures to densities ({:})'.format(len(self._coor_set)))
        self._transformer.volume.array.fill(0)
        for coor in self._coor_set:
            self.ligand.coor[:] = coor
            self._transformer.mask(self._rmask)
        if tofile:
            self._transformer.volume.tofile('mask.ccp4')
        mask = self._transformer.volume.array > 0
        self._transformer.volume.array.fill(0)

        nvalues = mask.sum()
        self._target = self.xmap.array[mask]
        self._models = np.zeros((len(self._coor_set), nvalues), dtype=np.float64)
        for n, coor in enumerate(self._coor_set):
            self.ligand.coor[:] = coor
            self._transformer.density()
            self._models[n] = self._transformer.volume.array[mask]
            if tofile:
                self._transformer.volume.tofile('density_{}.ccp4'.format(n))
            self._transformer.reset()

    def _QP(self):
        logger.info("Starting QP.")
        qpsolver = QPSolver(self._target, self._models)
        logger.info("Initializing.")
        qpsolver.initialize()
        logger.info("Solving")
        qpsolver()
        self._occupancies = qpsolver.occupancies

    def _MIQP(self, maxfits=None, exact=False, threshold=None):
        logger.info("Starting MIQP.")
        miqpsolver = MIQPSolver(self._target, self._models)
        logger.info("Initializing.")
        miqpsolver.initialize()
        logger.info("Solving")
        if maxfits is None:
            cardinality = self.cardinality
        else:
            cardinality = maxfits
        if threshold is None:
            threshold = self.threshold
        else:
            threshold = threshold
        miqpsolver(maxfits=cardinality, exact=exact, threshold=threshold)
        self._occupancies = miqpsolver.occupancies

    def _update_conformers(self):
        logger.info("Updating conformer list.")
        logger.info("Old number of conformers: {:d}".format(len(self._coor_set)))
        new_coor_set = []
        cutoff = 0.002
        for n, coor in enumerate(self._coor_set):
            if self._occupancies[n] >= cutoff:
                new_coor_set.append(coor)
        if new_coor_set:
            self._coor_set = new_coor_set
            self._occupancies = self._occupancies[self._occupancies >= cutoff]
        else:
            logger.warning("No conformer found with occupancy bigger than {:}.".format(cutoff))
            sorted_indices = np.argsort(self._occupancies)[::-1]
            indices = np.arange(len(self._occupancies))[sorted_indices]
            nbest = min(5, len(self._occupancies))
            for i in indices[:nbest]:
                new_coor_set.append(self._coor_set[i])
            self._coor_set = new_coor_set
            self._occupancies = self._occupancies[indices[:nbest]]
        logger.info("New number of conformers: {:d}".format(len(self._coor_set)))

    def _write_intermediate_structures(self, base='intermediate'):
        logger.info("Writing intermediate structures to file.")
        fname_base = self._djoiner(base + '_{:d}_{:d}_{:d}.pdb')
        for n, coor in enumerate(self._coor_set):
            self.ligand.coor[:] = coor
            ligand = self.ligand.select('q', 0, '!=')
            fname = fname_base.format(self._cluster_index, self._iteration, n)
            ligand.tofile(fname)

    def get_conformers(self, cutoff=0.01):
        conformers = []
        iterator = zip(self._coor_set, self._occupancies)
        for coor, occ in iterator:
            if occ >= cutoff:
                ligand = Ligand(self.ligand.data.copy(), self.ligand.coor.copy())
                ligand.coor[:] = coor
                ligand.q.fill(occ)
                conformers.append(ligand)
        ## Sort conformers based on occupancy
        #conformers = sorted(conformers, key=lambda conformer: conformer.q[0], reverse=True)
        return conformers

    def write_results(self, base='conformer', cutoff=0.01):
        logger.info("Writing results to file.")
        fname_base = self._djoiner(base + '_{:d}.pdb')
        fnames = []
        iterator = zip(self._coor_set, self._occupancies)
        n = 1
        old_q = self.ligand.q.copy()
        for coor, occ in iterator:
            if occ >= cutoff:
                self.ligand.q.fill(occ)
                self.ligand.coor[:] = coor
                fname = fname_base.format(n)
                self.ligand.tofile(fname)
                n += 1
                fnames.append(fname)
        self.ligand.q[:] = old_q
        return fnames
