import itertools
import logging
import os
from string import ascii_uppercase

import numpy as np

from .backbone import NullSpaceOptimizer, move_direction_adp
from .clash import ClashDetector
from .samplers import ChiRotator
from .solvers import MIQPSolver, QPSolver
from .structure import Structure
from .transformer import Transformer

logger = logging.getLogger(__name__)

class _BaseQFitOptions:

    def __init__(self):

        # General options
        self.directory = '.'
        self.debug = False

        # Density creation options
        self.map_type = None
        self.resolution = None
        self.resolution_min = None
        self.scattering = 'xray'

        # Sampling options
        self.clash_scaling_factor = 1
        self.dofs_per_iteration = 2
        self.dofs_stepsize = 8

        # MIQP options
        self.cardinality = 2
        self.threshold = 0.30

    def apply_command_args(self, args):

        for key, value in vars(args).items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class QFitRotamericResidueOptions(_BaseQFitOptions):

    def __init__(self):
        super().__init__()

        # Backbone sampling
        self.sample_backbone = False
        self.neighbor_residues_required = 2

        # Rotamer sampling
        self.sample_rotamers = True
        self.rotamer_neighborhood = 40
        #self.rotamer_neighborhood_first = None

        # General settings
        # Exclude certain atoms always during optimization, e.g. backbone
        self.exclude_atoms = None


class _BaseQFit:

    def __init__(self, conformer, structure, xmap, options):

        self.structure = structure
        self.conformer = conformer
        self.conformer.q = 1
        self.xmap = xmap
        self.options = options

        self._coor_set = [self.conformer.coor]
        self._occupancies = [1.0]

        if options.resolution is not None and self.xmap.resolution is None:
            self._smax = 1 / (2 * options.resolution)
            self._simple = False
        elif self.xmap.resolution is not None:
            self._smax = 1 / (2 * self.xmap.resolution)
            self._simple = False
        else:
            self._smax = None
            self._simple = True

        self._smin = 0
        self._rmask = 1.5
        if self.options.resolution_min is not None:
            self._smin = 1 / (2 * options.resolution_min)
        elif self.xmap.resolution_min is not None:
            self._smin = 1 / (2 * self.xmap.resolution_min)

        self._xmap_model = xmap.zeros_like(self.xmap)
        # To speed up the density creation steps, reduce space group symmetry to P1
        self._xmap_model.set_space_group("P1")
        self._voxel_volume = self.xmap.unit_cell.calc_volume() / self.xmap.array.size

    def get_conformers(self):
        conformers = []
        for q, coor in zip(self._occupancies, self._coor_set):
            conformer = self.conformer.copy()
            conformer.coor = coor
            conformer.q = q
            conformers.append(conformer)
        return conformers

    def _update_transformer(self, structure):
        self.conformer = structure
        self._transformer = Transformer(
            structure, self._xmap_model, smax=self._smax, smin=self._smin,
            simple=self._simple, scattering=self.options.scattering)
        logger.debug("Initializing radial density lookup table.")
        self._transformer.initialize()

    def _convert(self):

        """Convert structures to densities and extract relevant values for (MI)QP."""
        logger.info("Converting")

        logger.debug("Masking")
        for n, coor in enumerate(self._coor_set):
            self.conformer.coor = coor
            self._transformer.mask(1.5)
        mask = (self._transformer.xmap.array > 0)
        self._transformer.reset(full=True)

        nvalues = mask.sum()
        self._target = self.xmap.array[mask]

        logger.debug("Density")
        nmodels = len(self._coor_set)
        self._models = np.zeros((nmodels, nvalues), float)
        for n, coor in enumerate(self._coor_set):
            self.conformer.coor = coor
            self._transformer.density()
            self._models[n] = self._transformer.xmap.array[mask]
            self._transformer.reset(rmax=3)

    def _solve(self, cardinality=None, threshold=None):
        do_qp = cardinality == threshold == None
        if do_qp:
            solver = QPSolver(self._target, self._models)
            solver()
        else:
            solver = MIQPSolver(self._target, self._models)
            solver(cardinality=cardinality,
                   threshold=threshold)
        self._occupancies = solver.weights

        residual = np.sqrt(2 * solver.obj_value + np.inner(self._target, self._target)) * self._voxel_volume
        logger.info(f"Residual under footprint: {residual:.4f}")
        return residual

    def _update_conformers(self):
        new_coor_set = []
        new_occupancies = []
        for q, coor in zip(self._occupancies, self._coor_set):
            if q >= 0.002:
                new_coor_set.append(coor)
                new_occupancies.append(q)
        self._coor_set = new_coor_set
        self._occupancies = np.asarray(new_occupancies)

    def _write_intermediate_conformers(self, prefix="_conformer"):
        for n, coor in enumerate(self._coor_set):
            self.conformer.coor = coor
            fname = os.path.join(self.options.directory, f"{prefix}_{n}.pdb")
            self.conformer.tofile(fname)

    def _write_maps(self):
        """Write out model and difference map."""
        # Create maps
        for q, coor in zip(self._occupancies, self._coor_set):
            self.conformer.q = q
            self.conformer.coor = coor
            self._transformer.mask(self._rmask)
        fname = os.path.join(self.options.directory, 'mask.mrc')
        self._transformer.xmap.tofile(fname)
        mask = self._transformer.xmap.array > 0
        self._transformer.reset(full=True)

        for q, coor in zip(self._occupancies, self._coor_set):
            self.conformer.q = q
            self.conformer.coor = coor
            self._transformer.density()
        fname = os.path.join(self.options.directory, 'model.mrc')
        self._transformer.xmap.tofile(fname)
        values = self._transformer.xmap.array[mask]
        self._transformer.xmap.array -= self.xmap.array
        fname = os.path.join(self.options.directory, 'diff.mrc')
        self._transformer.xmap.tofile(fname)

        self._transformer.reset(full=True)
        self._transformer.xmap.array[mask] = values
        fname = os.path.join(self.options.directory, 'model_masked.mrc')
        self._transformer.xmap.tofile(fname)
        values = self.xmap.array[mask]
        self._transformer.xmap.array[mask] -= values
        fname = os.path.join(self.options.directory, 'diff_masked.mrc')
        self._transformer.xmap.tofile(fname)


class QFitRotamericResidue(_BaseQFit):

    def __init__(self, residue, structure, xmap, options):

        # Check if residue is complete, for now we cant handle incomplete
        # residues.
        atoms = residue.name
        for atom in residue._rotamers['atoms']:
            if atom not in atoms:
                msg = "Residue is incomplete. Build full sidechain for qfitting"
                raise RuntimeError(msg)

        super().__init__(residue, structure, xmap, options)
        self.residue = residue
        # Get the segment that the residue belongs to
        for segment in self.structure.segments:
            if self.residue in segment:
                self.segment = segment
                break

        # For Proline we do not want to sample the neighborhood.
        if self.residue.resn[0] == "PRO":
            self.options.rotamer_neighborhood = 0

        # Set up the clashdetector, exclude the bonded interaction of the N and
        # C atom of the residue
        self._setup_clash_detector()
        self._update_transformer(self.residue)

    def _setup_clash_detector(self):

        residue = self.residue
        segment = self.segment
        index = segment.find(residue.id)
        # Exclude peptide bonds from clash detector
        exclude = []
        if index > 0:
            N_index = residue.select('name', 'N')[0]
            N_neighbor = segment.residues[index - 1]
            neighbor_C_index = N_neighbor.select('name', 'C')[0]
            if np.linalg.norm(residue._coor[N_index] - segment._coor[neighbor_C_index]) < 2:
                coor = N_neighbor._coor[neighbor_C_index]
                exclude.append((N_index, coor))
        if index < len(segment.residues) - 1:
            C_index = residue.select('name', 'C')[0]
            C_neighbor = segment.residues[index + 1]
            neighbor_N_index = C_neighbor.select('name', 'N')[0]
            if np.linalg.norm(residue._coor[C_index] - segment._coor[neighbor_N_index]) < 2:
                coor = C_neighbor._coor[neighbor_N_index]
                exclude.append((C_index, coor))
        # Obtain atoms which the residue can clash
        resi, icode = residue.id
        chainid = self.segment.chain[0]
        if icode:
            selection_str = f'not (resi {resi} and icode {icode} and chain {chainid})'
            receptor = self.structure.extract(selection_str)
        else:
            receptor = self.structure.extract(f'not (resi {resi} and chain {chainid})').copy()
        # Find symmetry mates of the receptor
        starting_coor = self.structure.coor.copy()
        iterator = self.xmap.unit_cell.iter_struct_orth_symops
        for symop in iterator(self.structure, target=self.residue, cushion=5):
            if symop.is_identity():
                continue
            self.structure.rotate(symop.R)
            self.structure.translate(symop.t)
            receptor = receptor.combine(self.structure)
            self.structure.coor = starting_coor

        self._cd = ClashDetector(residue, receptor, exclude=exclude,
                                 scaling_factor=self.options.clash_scaling_factor)
        #receptor.tofile('clash_receptor.pdb')

    def run(self):

        if self.options.sample_backbone:
            self._sample_backbone()
        if self.residue.nchi >= 1 and self.options.sample_rotamers:
            self._sample_sidechain()
        #self._write_maps()

    def _sample_backbone(self):

        # Check if residue has enough neighboring residues
        index = self.segment.find(self.residue.id)
        nn = self.options.neighbor_residues_required
        if index < nn or index + nn > len(self.segment):
            return
        segment = self.segment[index - nn: index + nn + 1]

        cb_atom = self.residue.extract('name', 'CB')
        try:
            unit_cell = self.xmap.unit_cell
            u_matrix = [[cb_atom.u00[0], cb_atom.u01[0], cb_atom.u02[0]],
                        [cb_atom.u01[0], cb_atom.u11[0], cb_atom.u12[0]],
                        [cb_atom.u02[0], cb_atom.u12[0], cb_atom.u22[0]],
                       ]
            directions = move_direction_adp(u_matrix, unit_cell)
        except AttributeError:
            directions = np.identity(3)

        optimizer = NullSpaceOptimizer(segment)

        start_coor = cb_atom.coor[0]
        torsion_solutions = []
        amplitudes = np.linspace(0.10, 0.30, 3, endpoint=True)
        sigma = 0.125
        for amplitude, direction in itertools.product(amplitudes, directions):
            endpoint = start_coor + (amplitude + sigma * np.random.random()) * direction
            optimize_result = optimizer.optimize(endpoint)
            torsion_solutions.append(optimize_result['x'])

            endpoint = start_coor - (amplitude + sigma * np.random.random()) * direction
            optimize_result = optimizer.optimize(endpoint)
            torsion_solutions.append(optimize_result['x'])
        starting_coor = segment.coor
        for solution in torsion_solutions:
            optimizer.rotator(solution)
            self._coor_set.append(self.residue.coor)
            segment.coor = starting_coor

    def _sample_sidechain(self):

        start_chi_index = 1
        sampling_window = np.arange(
            -self.options.rotamer_neighborhood,
            self.options.rotamer_neighborhood + self.options.dofs_stepsize,
            self.options.dofs_stepsize)
        rotamers = self.residue.rotamers
        rotamers.append([self.residue.get_chi(i) for i in range(1, self.residue.nchi + 1)])

        iteration = 0
        while True:
            end_chi_index = min(start_chi_index + self.options.dofs_per_iteration,
                                self.residue.nchi + 1)
            for chi_index in range(start_chi_index, end_chi_index):

                # Set active and passive atoms, since we are iteratively
                # building up the sidechain. This updates the internal clash mask.
                self.residue.active = True
                if chi_index < self.residue.nchi:
                    deactivate = self.residue._rotamers['chi-rotate'][chi_index + 1]
                    selection = self.residue.select('name', deactivate)
                    self.residue._active[selection] = False
                self.residue.update_clash_mask()

                logger.info(f"Sampling chi: {chi_index} ({self.residue.nchi})")
                new_coor_set = []
                sampled_rotamers = []
                n = 0
                for coor in self._coor_set:
                    n += 1
                    self.residue.coor = coor
                    chis = [self.residue.get_chi(i) for i in range(1, chi_index)]
                    for rotamer in rotamers:
                        # Check if the residue configuration corresponds to the
                        # current rotamer
                        is_this_rotamer = True
                        for curr_chi, rotamer_chi in zip(chis, rotamer):
                            diff_chi = abs(curr_chi - rotamer_chi)
                            if 360 - self.options.rotamer_neighborhood > diff_chi > self.options.rotamer_neighborhood:
                                is_this_rotamer = False
                                break
                        if not is_this_rotamer:
                            continue
                        # Set the chi angle to the standard rotamer value.
                        self.residue.set_chi(chi_index, rotamer[chi_index - 1])

                        # The starting chi angles are similar for many
                        # rotamers, make sure we are not sampling double
                        unique = True
                        residue_coor = self.residue.coor
                        for rotamer_coor in sampled_rotamers:
                            if np.allclose(rotamer_coor, residue_coor, atol=0.01):
                                unique = False
                                break
                        if not unique:
                            continue
                        sampled_rotamers.append(residue_coor)

                        # Sample around the neighborhood of the rotamer
                        chi_rotator = ChiRotator(self.residue, chi_index)
                        for angle in sampling_window:
                            chi_rotator(angle)
                            if not self._cd() and self.residue.clashes() == 0:
                                new_coor_set.append(self.residue.coor)
                self._coor_set = new_coor_set
            #self._write_intermediate_conformers(f"conformer_{iteration}")

            logger.info("Nconf: {:d}".format(len(self._coor_set)))
            if not self._coor_set:
                msg = "No conformers could be generated. Check for initial clashes."
                raise RuntimeError(msg)
            if self.options.debug:
                prefix = os.path.join(self.options.directory, f'_conformer_{iteration}.pdb')
                self._write_intermediate_conformers(prefix=prefix)
            # QP
            logger.debug("Converting densities.")
            self._convert()
            logger.info("Solving QP.")
            self._solve()
            logger.debug("Updating conformers")
            self._update_conformers()
            #self._write_intermediate_conformers(f"qp_{iteration}")
            # MIQP
            self._convert()
            logger.info("Solving MIQP.")
            residual1 = self._solve(cardinality=1,
                        threshold=self.options.threshold)
            residual2 = self._solve(cardinality=2,
                        threshold=self.options.threshold)
            # residual2 is guaranteed to be lower
            diff = residual1 - residual2
            logger.info(f"Improvement in residual: {diff:.5f}")
            if diff < 0.0005:
                cardinality = 1
            else:
                cardinality = 2
            self._solve(cardinality=cardinality,
                        threshold=self.options.threshold)

            self._update_conformers()
            #self._write_intermediate_conformers(f"miqp_{iteration}")
            logger.info("Nconf after MIQP: {:d}".format(len(self._coor_set)))

            # Check if we are done
            if chi_index == self.residue.nchi:
                return
            iteration += 1
            start_chi_index += 1

    def tofile(self):

        conformers = self.get_conformers()
        for n, conformer in enumerate(conformers, start=1):
            fname = os.path.join(self.options.directory, f'conformer_{n}.pdb')
            conformer.tofile(fname)
        # Make a multiconformer residue
        nconformers = len(conformers)
        mc_residue = Structure.fromstructurelike(conformers[0])
        if nconformers == 1:
            mc_residue.altloc = ''
        else:
            mc_residue.altloc = 'A'
            for altloc, conformer in zip(ascii_uppercase[1:], conformers[1:]):
                conformer.altloc = altloc
                mc_residue = mc_residue.combine(conformer)
        mc_residue = mc_residue.reorder()
        fname = os.path.join(self.options.directory, f"multiconformer_residue.pdb")
        mc_residue.tofile(fname)


class QFitResidue(_BaseQFit):

    def __call__(self):
        pass

    def _sample_backbone(self):
        pass

    def _sample_sidechain(self):
        pass


class QFitSegmentOptions(_BaseQFitOptions):

    def __init__(self):
        super().__init__()
        self.fragment_length = 5


class QFitSegment(_BaseQFit):

    """Determines consistent protein segments based on occupancy / density fit"""

    def __init__(self, segment, xmap, options):
        super().__init__(segment, xmap, options)
        self.segment = segment
        self.segment.q = 1

    def __call__(self):

        # Build up initial elements
        multiconformers = []
        for rg in self.segment.residue_groups:
            altlocs = np.unique(rg.altloc)
            multiconformer = []
            for altloc in altlocs:
                if not altloc and naltlocs > 1:
                    continue
                conformer = Structure.fromstructurelike(rg.extract('altloc', (altloc, '')))
                multiconformer.append(conformer)
            multiconformers.append(multiconformer)

        fl = self.options.fragment_length
        while len(multiconformers) > 1:

            n = len(multiconformers)
            fragment_multiconformers = [multiconformers[i: i + fl] for i in range(0, n, fl)]
            multiconformers = []
            for fragment_multiconformer in fragment_multiconformers:
                # Create all combinations of alternate residue fragments
                fragments = []
                for fragment_conformer in itertools.product(*fragment_multiconformer):
                    # Build up fragment by combining conformers
                    for n, element in enumerate(fragment_conformer):
                        if n == 0:
                            fragment = element
                        else:
                            fragment = fragment.combine(element)
                    fragments.append(fragment)
                # We have the fragments, select consistent optimal set
                self._update_transformer(fragments[0])
                self._coor_set = [fragment.coor for fragment in fragments]
                self._solve()
                self._solve(cardinality=self.options.cardinality,
                            threshold=self.options.threshold)
                multiconformer = []
                for coor in self._coor_set:
                    fragment_conformer = fragments[0].copy()
                    fragment_conformer.coor = coor
                    multiconformer.append(fragment_conformer)
                multiconformers.append(multiconformer)
        return multiconformers[0]


class QFitLigand(_BaseQFit):

    def __init__(self, ligand, root, receptor, xmap, options):
        super().__init__(ligand, xmap, options)
        self.ligand = ligand
        self.receptor = receptor
        csf = self.options.clash_scaling_factor
        self._cd = ClashDetector(ligand, receptor, scaling_factor=csf)
        self._rigid_clusters = ligand.rigid_clusters()
        self._update_transformer(ligand)

    def run(self):

        if self.options.local_search:
            self._local_search()
        self._sample_internal_dofs()

    def _local_search(self):
        pass

    def _sample_internal_dofs(self):

        nbonds = None
        if nbonds == 0:
            return

        starting_bond_index = 0

        while True:
            end_bond_index = min(starting_bond_index + self.options.dofs_per_iteration, nbonds)
            for bond_index in range(starting_bond_index, end_bond_index):

                nbonds_sampled = bond_index + 1
                self.ligand.active = False
                for cluster in self._rigid_clusters:
                    for sampled_bond in bonds[:nbonds_sampled]:
                        if sampled_bond[0] in cluster or sampled_bond[1] in cluster:
                            self.ligand.active[cluster] = True
                            for atom in cluster:
                                self.ligand.active[self.ligand.connectivity[atom]] = True

                bond = bonds[bond_index]
                atoms = [self.ligand.name[bond[0]], self.ligand.name[bond[1]]]
                new_coor_set = []
                for coor in self._coor_set:
                    self.ligand.coor = coor
                    rotator = BondRotator(self.ligand, *atoms)
                    for angle in sampling_range:
                        rotator(angle)
                        if not self._cd() and self.ligand.clashes():
                            new_coor_set.append(self.ligand.coor)
                self._coor_set = new_coor_set

                if not self._coor_set:
                    return


class QFitCovalentLigand(_BaseQFit):
    pass

