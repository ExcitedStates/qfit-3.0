from __future__ import division
from itertools import izip
import logging
logger = logging.getLogger(__name__)

import numpy as np

from .structure import Residue
from .volume import Volume
from .samplers import ChiRotator, ClashDetector
from .transformer import Transformer
from .solvers import QPSolver, MIQPSolver
from .helpers import DJoiner

class SideChainBuilder:

    def __init__(self, residue, xmap, resolution, receptor=None, stepsize=5,
                 build_stepsize=1, neighborhood=40, directory='.'):
        self.residue = residue
        self.xmap = xmap
        self.resolution = resolution
        self.receptor = receptor
        self.build_stepsize = build_stepsize
        self.stepsize = stepsize
        self.neighborhood = neighborhood
        self._mask_radius = 0.5 + self.resolution / 3.0
        self._mask = np.zeros_like(self.xmap.array, dtype=bool)
        self._coor_set = [self.residue.coor.copy()]

        self._djoiner = DJoiner(directory)

        # Initialize the clash detector
        # TODO make sure the bonded atoms are not taken into account
        #self._cd = ClashDetector(self.residue, self.receptor, scaling_factor=0.75)

        # Initialize density creation transformer
        self._model_map = Volume.zeros_like(self.xmap)
        self._model_map.set_spacegroup("P1")
        smax = 1 / (2 * self.resolution)
        self._transformer = Transformer(self.residue, self._model_map, smax=smax, rmax=3)
        self._transformer.initialize()

    def __call__(self):

        if self.residue.nchi < 1:
            return

        # Add current conformer as a rotamer
        rotamers = list(self.residue.rotamers)
        chis = [self.residue.get_chi(chi_index) for chi_index in xrange(1, self.residue.nchi + 1)]
        if chis not in rotamers:
            rotamers.append(chis)

        start_chi_index = 1
        sampling_window = np.arange(-self.neighborhood,
                                    self.neighborhood + 0.5, self.stepsize)
        self._iteration = 0
        # Sample each chi angle
        while True:
            logger.info("Sampling iteration: {}".format(self._iteration))
            # Sample conformations
            end_chi_index = min(start_chi_index + self.build_stepsize,
                                self.residue.nchi + 1)
            logger.info("Start chi index: {}".format(start_chi_index))
            logger.info("End chi index: {}".format(end_chi_index))
            for chi_index in xrange(start_chi_index, end_chi_index):
                logger.info("Chi index: {}".format(chi_index))
                # Set the active atoms
                self.residue.activate()
                atoms_to_deactivate = []
                for i in xrange(chi_index + 1, self.residue.nchi + 1):
                    atoms_to_deactivate += self.residue._residue_data['chi-rotate'][i]
                atoms_to_deactivate = list(set(atoms_to_deactivate))
                if atoms_to_deactivate:
                    selection = self.residue.select(
                        'atomname', atoms_to_deactivate, return_ind=True)
                    self.residue.deactivate(selection)

                logger.info("Number of active atoms: {}".format(self.residue.active.sum()))

                new_coor_set = []
                sampled_rotamers = []
                for coor in self._coor_set:
                    self.residue.coor[:] = coor
                    # Check if residue is part of a rotamer. if so sample its
                    # neighborhood
                    chis = [self.residue.get_chi(i) for i in xrange(1, chi_index)]
                    for rotamer in rotamers:

                        is_this_rotamer = True
                        for curr_chi, rot_chi  in izip(chis, rotamer):
                            if abs(curr_chi - rot_chi) > self.neighborhood:
                                is_this_rotamer = False
                                break
                        if not is_this_rotamer:
                            continue
                        self.residue.set_chi(chi_index, rotamer[chi_index - 1])

                        # Check if coordinates are already present in sampled_rotamer set
                        unique = True
                        residue_coor = self.residue.coor
                        for rotamer_coor in sampled_rotamers:
                            if np.allclose(rotamer_coor, residue_coor, atol=0.01, rtol=0):
                                unique = False
                                break
                        if not unique:
                            continue
                        sampled_rotamers.append(self.residue.coor.copy())

                        chirotator = ChiRotator(self.residue, chi_index)
                        for angle in sampling_window:
                            chirotator(angle)
                            nclashes = self.residue.clashes()
                            if nclashes > 0:
                                continue
                            new_coor_set.append(self.residue.coor.copy())

                self._coor_set = new_coor_set
                self._iteration += 1

            self.residue.q.fill(0)
            self.residue.q[self.residue.active] = 1

            #self._write_intermediate_structures()
            # Perform QP/MIQP
            self._convert()
            self._QP()
            self._update_conformers()

            self._convert()
            self._MIQP()
            self._update_conformers()

            if chi_index == self.residue.nchi:
                break

            start_chi_index += 1

    def _convert(self):
        """Convert structures into densities and extract masked values."""
        nconformers = len(self._coor_set)
        logger.info("Converting conformers to densities ({})".format(nconformers))
        transformer = self._transformer
        residue = self.residue
        mask = self._mask
        # Fill in mask
        transformer.volume.array.fill(0)
        for coor in self._coor_set:
            residue.coor[:] = coor
            transformer.mask(self._mask_radius)
        np.greater(transformer.volume.array, 0, mask)
        transformer.volume.array.fill(0)
        nvalues = mask.sum()
        self._target = self.xmap.array[mask]
        shape = (len(self._coor_set), nvalues)
        self._models = np.zeros(shape, float)
        for n, coor in enumerate(self._coor_set):
            residue.coor[:] = coor
            transformer.density()
            self._models[n] = self._transformer.volume.array[mask]
            transformer.reset()

    def _QP(self):
        qpsolver = QPSolver(self._target, self._models)
        qpsolver.initialize()
        qpsolver()
        self._occupancies = qpsolver.occupancies

    def _MIQP(self, cardinality=5, threshold=0.01):
        miqpsolver = MIQPSolver(self._target, self._models, threads=1)
        miqpsolver.initialize()
        miqpsolver(maxfits=cardinality, threshold=threshold)
        self._occupancies = miqpsolver.occupancies

    def _update_conformers(self):
        new_coor_set = []
        cutoff = 1 / 500.0
        for q, coor in izip(self._occupancies, self._coor_set):
            if q >= cutoff:
                new_coor_set.append(coor)
        self._coor_set = new_coor_set
        self._occupancies = self._occupancies[self._occupancies >= cutoff]

    def _write_intermediate_structures(self, base='intermediate'):
        logger.info("Writing intermediate structures to file.")
        fname_base = self._djoiner(base + '_{:d}_{:d}.pdb')
        for n, coor in enumerate(self._coor_set):
            self.residue.coor[:] = coor
            residue = self.residue.select('q', 0, '!=')
            fname = fname_base.format(self._iteration, n)
            residue.tofile(fname)

    def get_conformers(self, cutoff=0.01):
        conformers = []
        iterator = izip(self._coor_set, self._occupancies)
        for coor, occ in iterator:
            if occ >= cutoff:
                residue = Residue(self.residue.data.copy(), self.residue.coor.copy())
                residue.coor[:] = coor
                residue.q.fill(occ)
                conformers.append(residue)
        ## Sort conformers based on occupancy
        #conformers = sorted(conformers, key=lambda conformer: conformer.q[0], reverse=True)
        return conformers

    def write_results(self, base='conformer', cutoff=0.01):
        logger.info("Writing results to file.")
        fname_base = self._djoiner(base + '_{:d}.pdb')
        fnames = []
        iterator = izip(self._coor_set, self._occupancies)
        n = 1
        old_q = self.residue.q.copy()
        for coor, occ in iterator:
            if occ >= cutoff:
                self.residue.q.fill(occ)
                self.residue.coor[:] = coor
                fname = fname_base.format(n)
                self.residue.tofile(fname)
                n += 1
                fnames.append(fname)
        self.residue.q[:] = old_q
        return fnames
