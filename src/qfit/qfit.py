import itertools
import logging
import os
from sys import argv
import copy
from string import ascii_uppercase
import subprocess
import numpy as np
import tqdm

from .backbone import NullSpaceOptimizer, adp_ellipsoid_axes
from .clash import ClashDetector
from .samplers import ChiRotator, CBAngleRotator, BondRotator
from .samplers import CovalentBondRotator, GlobalRotator
from .samplers import RotationSets, Translator
from .solvers import QPSolver, MIQPSolver
from .structure import Structure, _Segment
from .structure.residue import residue_type
from .structure.ligand import BondOrder
from .transformer import Transformer
from .validator import Validator
from .volume import XMap
from .scaler import MapScaler
from .relabel import RelabellerOptions, Relabeller
from .structure.rotamers import ROTAMERS


logger = logging.getLogger(__name__)


class QFitOptions:
    def __init__(self):
        # General options
        self.directory = '.'
        self.verbose = False
        self.debug = False
        self.write_intermediate_conformers = False
        self.random_seed = None
        self.label = None
        self.map = None
        self.structure = None

        # Density preparation options
        self.density_cutoff = 0.3
        self.density_cutoff_value = -1
        self.subtract = True
        self.padding = 8.0
        self.waters_clash = True

        # Density creation options
        self.map_type = None
        self.resolution = None
        self.resolution_min = None
        self.scattering = 'xray'
        self.omit = False
        self.scale = True
        self.scale_rmask = 1.
        self.randomize_b = False
        self.bulk_solvent_level = 0.3

        # Sampling options
        self.clash_scaling_factor = 0.75
        self.external_clash = False
        self.dofs_per_iteration = 2
        self.dihedral_stepsize = 10
        self.hydro = False
        self.rmsd_cutoff = 0.01

        # MIQP options
        self.cplex = True
        self.cardinality = 5
        self.threshold = 0.20
        self.bic_threshold = True
        self.seg_bic_threshold = True

        ### From QFitRotamericResidueOptions, QFitCovalentLigandOptions
        # Backbone sampling
        self.sample_backbone = True
        self.neighbor_residues_required = 3
        self.sample_backbone_amplitude = 0.30
        self.sample_backbone_step = 0.1
        self.sample_backbone_sigma = 0.125

        # N-CA-CB angle sampling
        self.sample_angle = True
        self.sample_angle_range = 7.5
        self.sample_angle_step = 3.75

        # Rotamer sampling
        self.sample_rotamers = True
        self.rotamer_neighborhood = 60  # Was 80 in QFitCovalentLigandOptions
        self.remove_conformers_below_cutoff = False

        # Anisotropic refinement using phenix
        self.phenix_aniso = False

        # General settings
        # Exclude certain atoms always during density and mask creation to
        # influence QP / MIQP. Provide a list of atom names, e.g. ['N', 'CA']
        # TODO not implemented
        self.exclude_atoms = None

        ### From QFitLigandOptions
        # Ligand sampling
        self.local_search = True
        self.sample_ligand = True  # From QFitCovalentLigandOptions
        self.sample_ligand_stepsize = 10  # Was 8 in QFitCovalentLigandOptions
        self.selection = None
        self.cif_file = None

        ### From QFitSegmentOptions
        self.fragment_length = None

        ### From QFitProteinOptions
        self.nproc = 1
        self.pdb = None

    def apply_command_args(self, args):
        for key, value in vars(args).items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class _BaseQFit:
    def __init__(self, conformer, structure, xmap, options):
        self.structure = structure
        self.conformer = conformer
        self.conformer.q = 1
        self.xmap = xmap
        self.options = options
        self.BIC = np.inf
        self.prng = np.random.default_rng(self.options.random_seed)
        self._coor_set = [self.conformer.coor]
        self._occupancies = [1.0]
        self._bs = [self.conformer.b]
        self._smax = None
        self._simple = True
        self._rmask = 1.5

        reso = None
        if self.xmap.resolution.high is not None:
            reso = self.xmap.resolution.high
        elif options.resolution is not None:
            reso = options.resolution

        if reso is not None:
            self._smax = 1 / (2 * reso)
            self._simple = False
            self._rmask = 0.5 + reso / 3.0

        self._smin = None
        if self.xmap.resolution.low is not None:
            self._smin = 1 / (2 * self.xmap.resolution.low)
        elif self.options.resolution_min is not None:
            self._smin = 1 / (2 * options.resolution_min)

        self._xmap_model = xmap.zeros_like(self.xmap)
        self._xmap_model2 = xmap.zeros_like(self.xmap)

        # To speed up the density creation steps, reduce symmetry to P1
        self._xmap_model.set_space_group("P1")
        self._xmap_model2.set_space_group("P1")
        self._voxel_volume = self.xmap.unit_cell.calc_volume()
        self._voxel_volume /= self.xmap.array.size
    
    @property
    def directory_name(self):
        dname = self.options.directory
        return dname

    def get_conformers(self):
        conformers = []
        for q, coor, b in zip(self._occupancies, self._coor_set, self._bs):
            conformer = self.conformer.copy()
            conformer = conformer.extract(f"resi {self.conformer.resi[0]} and "
                                          f"chain {self.conformer.chain[0]}")
            conformer.q = q
            conformer.coor = coor
            conformer.b = b
            conformers.append(conformer)
        return conformers

    def _update_transformer(self, structure):
        self.conformer = structure
        self._transformer = Transformer(
            structure, self._xmap_model,
            smax=self._smax, smin=self._smin,
            simple=self._simple,
            scattering=self.options.scattering,
        )
        logger.debug("[_BaseQFit._update_transformer]: Initializing radial density lookup table.")
        self._transformer.initialize()

    def _subtract_transformer(self, residue, structure):
        # Select the atoms whose density we are going to subtract:
        subtract_structure = structure.extract_neighbors(residue, self.options.padding)
        if not self.options.waters_clash:
            subtract_structure = subtract_structure.extract("resn", "HOH", "!=")

        # Calculate the density that we are going to subtract:
        self._subtransformer = Transformer(
            subtract_structure, self._xmap_model2,
            smax=self._smax, smin=self._smin,
            simple=self._simple,
            scattering=self.options.scattering,
        )
        self._subtransformer.initialize()
        self._subtransformer.reset(full=True)
        self._subtransformer.density()

        # Set the lowest values in the map to the bulk solvent level:
        np.maximum(self._subtransformer.xmap.array,
                   self.options.bulk_solvent_level,
                   out=self._subtransformer.xmap.array)

        # Subtract the density:
        self.xmap.array -= self._subtransformer.xmap.array

    def _convert(self):
        """Convert structures to densities and extract relevant values for (MI)QP."""
        logger.info("Converting conformers to density")
        logger.debug("Masking")
        self._transformer.reset(full=True)
        for n, coor in enumerate(self._coor_set):
            self.conformer.coor = coor
            self._transformer.mask(self._rmask)
        mask = (self._transformer.xmap.array > 0)
        self._transformer.reset(full=True)

        nvalues = mask.sum()
        self._target = self.xmap.array[mask]
        logger.debug("Density")
        nmodels = len(self._coor_set)
        self._models = np.zeros((nmodels, nvalues), float)
        # target_sum = self._target.sum()
        # Create an initial density to calculate the total integral density of
        # the model. This way we can derive an approximation of the solvent
        # level.
        # self.conformer.coor = self._coor_set[0]
        # self._transformer.density()
        # model_sum = self._transformer.xmap.array.sum()
        # residual_sum = target_sum - model_sum
        # solvent_level = residual_sum / nvalues
        # scaling_factor = model_sum / target_sum
        # self._target *= scaling_factor
        # logger.debug("Solvent level advice:", solvent_level)
        # logger.debug("Scaling factor:", scaling_factor)
        # logger.debug("Target sum:", target_sum)
        # logger.debug("Model sum:", model_sum)
        # self._transformer.reset(full=True)
        for n, coor in enumerate(self._coor_set):
            self.conformer.coor = coor
            self.conformer.b = self._bs[n]
            if self.options.randomize_b:
                self._update_transformer(self.conformer)
            self._transformer.density()
            model = self._models[n]
            model[:] = self._transformer.xmap.array[mask]
            np.maximum(model, self.options.bulk_solvent_level, out=model)
            self._transformer.reset(full=True)

    def _randomize_bs(self, bs, atoms):
        bs_copy = copy.deepcopy(bs)
        if self.options.randomize_b:
            mask = np.in1d(self.conformer.name, atoms)
            add = 0.2 * self.prng.random(bs_copy[mask].shape[0]) - 0.1
            bs_copy[mask] += np.multiply(bs[mask], add)
        return bs_copy

    def _solve(self, cardinality=None, threshold=None,
               loop_range=[0.5, 0.4, 0.33, 0.3, 0.25, 0.2]):
        # Create and run QP or MIQP solver
        do_qp = cardinality is threshold is None
        if do_qp:
            logger.info("Solving QP")
            solver = QPSolver(self._target, self._models, use_cplex=self.options.cplex)
            solver()
        else:
            logger.info("Solving MIQP")
            solver = MIQPSolver(self._target, self._models, use_cplex=self.options.cplex)

            # Threshold selection by BIC:
            if self.options.bic_threshold:
                self.BIC = np.inf
                for threshold in loop_range:
                    solver(cardinality=None, threshold=threshold)
                    if solver.error == True: #catch if MIQP failed
                       logger.debug('CPLEX ERROR: Removing conformer')
                       self.CPLEX_error = True
                       return
                    rss = solver.obj_value * self._voxel_volume
                    confs = np.sum(solver.weights >= 0.002)
                    n = len(self._target)
                    try:
                        natoms = len(self.residue._rotamers['atoms'])
                        k = 4 * confs * natoms
                    except AttributeError:
                        k = 4 * confs
                    except:
                        natoms = np.sum(self.ligand.active)
                        k = 4 * confs * natoms
                    BIC = n * np.log(rss / n) + k * np.log(n)
                    if BIC < self.BIC:
                        self.BIC = BIC
                    # else:
                    #     break
            else:
                solver(cardinality=cardinality, threshold=threshold)
       if solver.error == True:
           logger.debug('CPLEX ERROR: Removing conformer')
           self.CPLEX_error = True
           return
        else:
            # Update occupancies from solver weights
            self._occupancies = solver.weights
            return solver.obj_value

    def _update_conformers(self, cutoff=0.002):
        """Removes conformers with occupancy lower than cutoff.

        Args:
            cutoff (float, optional): Lowest acceptable occupancy for a conformer.
                Cutoff should be in range (0 < cutoff < 1).
        """
        logger.debug("Updating conformers based on occupancy")

        # Check that all arrays match dimensions.
        assert len(self._occupancies) == len(self._coor_set) == len(self._bs)

        # Filter all arrays & lists based on self._occupancies
        # NB: _coor_set and _bs are lists (not arrays). We must compress, not slice.
        filterarray = (self._occupancies >= cutoff)
        self._occupancies = self._occupancies[filterarray]
        self._coor_set = list(itertools.compress(self._coor_set, filterarray))
        self._bs = list(itertools.compress(self._bs, filterarray))

        logger.debug(f"Remaining valid conformations: {len(self._coor_set)}")

    def _write_intermediate_conformers(self, prefix="_conformer"):
        for n, coor in enumerate(self._coor_set):
            self.conformer.coor = coor
            fname = os.path.join(self.directory_name, f"{prefix}_{n}.pdb")

            data = {}
            for attr in self.conformer.data:
                array1 = getattr(self.conformer, attr)
                data[attr] = array1[self.conformer.active]
            Structure(data).tofile(fname)

    def write_maps(self):
        """Write out model and difference map."""
        if np.allclose(self.xmap.origin, 0):
            ext = 'ccp4'
        else:
            ext = 'mrc'

        # Create maps
        # for q, coor in zip(self._occupancies, self._coor_set):
        #    self.conformer.q = q
        #    self.conformer.coor = coor
        #    self._transformer.mask(self._rmask)
        # fname = os.path.join(self.directory_name, f'mask.{ext}')
        # self._transformer.xmap.tofile(fname)
        # mask = self._transformer.xmap.array > 0
        # self._transformer.reset(full=True)

        for q, coor, b in zip(self._occupancies, self._coor_set, self._bs):
            self.conformer.q = q
            self.conformer.coor = coor
            self.conformer.b = b
            self._transformer.density()
        fname = os.path.join(self.directory_name, f'model.{ext}')
        self._transformer.xmap.tofile(fname)
        self._transformer.xmap.array -= self.xmap.array
        fname = os.path.join(self.directory_name, f'diff.{ext}')
        self._transformer.xmap.tofile(fname)
        self._transformer.reset(full=True)
        # self._transformer.xmap.array *= -1
        # fname = os.path.join(self.directory_name, f'diff_negative.{ext}')
        # self._transformer.xmap.tofile(fname)

        # self._transformer.reset(full=True)
        # self._transformer.xmap.array[mask] = values
        # fname = os.path.join(self.directory_name, f'model_masked.{ext}')
        # self._transformer.xmap.tofile(fname)
        # values = self.xmap.array[mask]
        # self._transformer.xmap.array[mask] -= values
        # fname = os.path.join(self.directory_name, f'diff_masked.{ext}')
        # self._transformer.xmap.tofile(fname)


class QFitRotamericResidue(_BaseQFit):
    def __init__(self, residue, structure, xmap, options):
        super().__init__(residue, structure, xmap, options)
        self.residue = residue
        self.chain = residue.chain[0]
        self.resn = residue.resn[0]
        self.resi, self.icode = residue.id
        self.identifier = f"{self.chain}/{self.resn}{''.join(map(str, residue.id))}"

        if options.phenix_aniso:
            self.prv_resi = structure.resi[(residue._selection[0] - 1)]
            # Identify which atoms to refine anisotropically:
            if xmap.resolution.high < 1.45:
                adp = "not (water or element H)"
            else:
                adp = f"chain {self.chain} and resid {self.resi}"

            # Generate the parameter file for phenix refinement:
            labels = options.label.split(",")
            with open(f"chain_{self.chain}_res_{self.resi}_adp.params", "w") as params:
                params.write("refinement {\n"
                             "  electron_density_maps {\n"
                             "    map_coefficients {\n"
                            f"      mtz_label_amplitudes = {labels[0]}\n"
                            f"      mtz_label_phases = {labels[1]}\n"
                             "      map_type = 2mFo-DFc\n"
                             "    }\n"
                             "  }\n"
                             "  refine {\n"
                             "    strategy = *individual_sites *individual_adp\n"
                             "    adp {\n"
                             "      individual {\n"
                            f"        anisotropic = {adp}\n"
                             "      }\n"
                             "    }\n"
                             "  }\n"
                             "}\n")

            # Set the occupancy of the side chain to zero for omit map calculation
            out_root = f'out_{self.chain}_{self.resi}'
            structure.tofile(f'{out_root}.pdb')
            subprocess.run(["phenix.pdbtools",
                            "modify.selection="
                                f"\"chain {self.chain} and "
                                f"( resseq {self.resi} and not "
                                f"( name n or name ca or name c or name o or name cb ) or "
                                f"( resseq {self.prv_resi} and name n ) )\"",
                            "modify.occupancies.set=0",
                            "stop_for_unknowns=False",
                           f"{out_root}.pdb",
                           f"output.file_name={out_root}_modified.pdb"])

            # Add hydrogens to the structure:
            with open(f"{out_root}_modified_H.pdb", "w") as out_mod_H:
                subprocess.run(["phenix.reduce", f"{out_root}_modified.pdb"],
                               stdout=out_mod_H)

            # Generate CIF file of unknown ligands for refinement:
            subprocess.run(["phenix.elbow", "--do_all",
                            f"{out_root}_modified_H.pdb"])

            # Run the refinement protocol:
            if os.path.isfile(f'elbow.{out_root}_modified_H_pdb.all.001.cif'):
                elbow = f'elbow.{out_root}_modified_H_pdb.all.001.cif'
                subprocess.run(["phenix.refine",
                                f'{options.map}',
                                f'{out_root}_modified_H.pdb',
                                "--overwrite",
                                f'chain_{self.chain}_res_{self.resi}_adp.params',
                                f'refinement.input.xray_data.labels=F-obs',
                                f'{elbow}'])
            else:
                # Run the refinement protocol:
                subprocess.run(["phenix.refine",
                                f'{options.map}',
                                f'{out_root}_modified_H.pdb',
                                "--overwrite",
                                f'chain_{self.chain}_res_{self.resi}_adp.params',
                                f'refinement.input.xray_data.labels=F-obs'])

            # Reload structure and xmap as omit map:
            structure = Structure.fromfile(f'{out_root}_modified_H_refine_001.pdb').reorder()
            if not options.hydro:
                structure = structure.extract('e', 'H', '!=')
            structure_resi = structure.extract(f'resi {self.resi} and chain {self.chain}')
            if residue.icode[0]:
                structure_resi = structure_resi.extract('icode', residue.icode[0])
            chain = structure_resi[self.chain]
            conformer = chain.conformers[0]
            if residue.icode[0]:
                residue_id = (int(self.resi), residue.icode[0])
                residue = conformer[residue_id]
            else:
                residue = conformer[int(self.resi)]

            xmap = XMap.fromfile(f'{out_root}_modified_H_refine_001.mtz',
                                 resolution=None, label=options.label)
            xmap = xmap.canonical_unit_cell()
            if options.scale:
                # Prepare X-ray map
                scaler = MapScaler(xmap, scattering=options.scattering)
                sel_str = f"resi {self.resi} and chain {self.chain}"
                sel_str = f"not ({sel_str})"
                footprint = structure.extract(sel_str)
                footprint = footprint.extract('record', 'ATOM')
                scaler.scale(footprint, radius=1)
            xmap = xmap.extract(residue.coor, padding=options.padding)

        # Check if residue has complete heavy atoms. If not, complete it.
        expected_atoms = np.array(self.residue._rotamers['atoms'])
        missing_atoms = np.isin(expected_atoms, test_elements=self.residue.name, invert=True)
        if np.any(missing_atoms):
            logger.info(f"[{self.identifier}] {', '.join(expected_atoms[missing_atoms])} "
                        f"are not in structure. Rebuilding residue.")
            try:
                self.residue.complete_residue()
            except RuntimeError as e:
                raise RuntimeError(f"[{self.identifier}] Unable to rebuild residue.") from e
            else:
                logger.debug(f"[{self.identifier}] Rebuilt. Now has {', '.join(self.residue.name)} atoms.\n"
                             f"{self.residue.coor}")

            # Rebuild to include the new residue atoms
            index = len(self.structure.record)
            mask = getattr(self.residue, 'atomid') >= index
            data = {}
            for attr in self.structure.data:
                data[attr] = np.concatenate((getattr(structure, attr),
                                             getattr(residue, attr)[mask]))

            # Create a new Structure, and re-extract the current residue from it.
            #     This ensures the object-tree (i.e. residue.parent, etc.) is correct.
            # Then reinitialise _BaseQFit with these.
            #     This ensures _BaseQFit class attributes (self.residue, but also
            #     self._b, self._coor_set, etc.) come from the rebuilt data-structure.
            #     It is essential to have uniform dimensions on all data before
            #     we begin sampling.
            structure = Structure(data)
            residue = structure[self.chain].conformers[0][residue.id]
            super().__init__(residue, structure, xmap, options)
            self.residue = residue
            if self.options.debug:
                # This should be output with debugging, and shouldn't
                #   require the write_intermediate_conformers option.
                fname = os.path.join(self.directory_name, "rebuilt_residue.pdb")
                self.residue.tofile(fname)

        # If including hydrogens, report if any H are missing
        if options.hydro:
            expected_h_atoms = np.array(self.residue._rotamers['hydrogens'])
            missing_h_atoms = np.isin(expected_h_atoms, test_elements=self.residue.name, invert=True)
            if np.any(missing_h_atoms):
                logger.warning(f"[{self.identifier}] Missing hydrogens "
                               f"{', '.join(expected_atoms[missing_h_atoms])}.")

        # Ensure clash detection matrix is filled.
        self.residue._init_clash_detection(self.options.clash_scaling_factor)

        # Get the segment that the residue belongs to
        self.segment = None
        for segment in self.structure.segments:
            if segment.chain[0] == self.chain and self.residue in segment:
                index = segment.find(self.residue.id)
                if (len(segment[index].name) == len(self.residue.name)) and \
                        (segment[index].altloc[-1] == self.residue.altloc[-1]):
                    self.segment = segment
                    logger.info(f"[{self.identifier}] index {index} in {segment}")
                    break
        if self.segment is None:
            rtype = residue_type(self.residue)
            if rtype == "rotamer-residue":
                self.segment = _Segment(self.structure.data, selection=self.residue._selection,
                                        parent=self.structure, residues=[self.residue])
                logger.warning(f"[{self.identifier}] Could not determine protein segment. "
                               f"Using independent protein segment.")

        # Set up the clash detector, exclude the bonded interaction of the N and
        # C atom of the residue
        self._setup_clash_detector()
        if options.subtract:
            self._subtract_transformer(self.residue, self.structure)
        self._update_transformer(self.residue)

    @property
    def directory_name(self):
        # This is a QFitRotamericResidue, so we're working on a residue.
        # Which residue are we working on?
        resi_identifier = f"{self.chain}_{self.resi}"
        if self.icode:
            resi_identifier += f"_{self.icode}"

        dname = os.path.join(super().directory_name, resi_identifier)
        return dname

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

        # Obtain atoms with which the residue can clash
        resi, icode = residue.id
        chainid = self.segment.chain[0]
        if icode:
            selection_str = f"not (resi {resi} and icode {icode} and chain {chainid})"
            receptor = self.structure.extract(selection_str)
        else:
            sel_str = f"not (resi {resi} and chain {chainid})"
            receptor = self.structure.extract(sel_str).copy()

        # Find symmetry mates of the receptor
        starting_coor = self.structure.coor.copy()
        iterator = self.xmap.unit_cell.iter_struct_orth_symops
        for symop in iterator(self.structure, target=self.residue, cushion=5):
            if symop.is_identity():
                continue
            logger.debug(f"[{self.identifier}] Building symmetry partner for clash_detector: [R|t]\n"
                         f"{symop}")
            self.structure.rotate(symop.R)
            self.structure.translate(symop.t)
            receptor = receptor.combine(self.structure)
            self.structure.coor = starting_coor

        self._cd = ClashDetector(residue, receptor,
                                 exclude=exclude,
                                 scaling_factor=self.options.clash_scaling_factor)
        # receptor.tofile('clash_receptor.pdb')

    def run(self):
        if self.options.sample_backbone:
            self._sample_backbone()

        if self.options.sample_angle:
            self._sample_angle()

        if self.residue.nchi >= 1 and self.options.sample_rotamers:
            self._sample_sidechain()
        else:
            # Perform a final QP / MIQP step
            self.residue.active = True
            self.residue.update_clash_mask()
            new_coor_set = []
            new_bs = []
            for coor, b in zip(self._coor_set, self._bs):
                self.residue.coor = coor
                self.residue.b = b
                if self.options.external_clash:
                    if not self._cd() and self.residue.clashes() == 0:
                        new_coor_set.append(coor)
                        new_bs.append(b)
                elif self.residue.clashes() == 0:
                    new_coor_set.append(coor)
                    new_bs.append(b)
            self._coor_set = new_coor_set
            self._bs = new_bs

            # QP score conformer occupancy
            self._convert()
            self._solve()
            self._update_conformers()
            if self.options.write_intermediate_conformers:
                self._write_intermediate_conformers(prefix="qp_solution")

            # MIQP score conformer occupancy
            self._convert()
            self._solve(threshold=self.options.threshold,
                        cardinality=self.options.cardinality)
            self._update_conformers()
            if self.options.write_intermediate_conformers:
                self._write_intermediate_conformers(prefix="miqp_solution")

        # Now that the conformers have been generated, the resulting
        # conformations should be examined via GoodnessOfFit:
        validator = Validator(self.xmap, self.xmap.resolution,
                              self.options.directory)

        if self.xmap.resolution.high < 3.0:
            cutoff = 0.7 + (self.xmap.resolution.high - 0.6) / 3.0
        else:
            cutoff = 0.5 * self.xmap.resolution.high

        self.validation_metrics = validator.GoodnessOfFit(self.conformer,
                                                          self._coor_set,
                                                          self._occupancies,
                                                          cutoff)

    def _sample_backbone(self):
        # Check if residue has enough neighboring residues
        index = self.segment.find(self.residue.id)
        # active = self.residue.active
        nn = self.options.neighbor_residues_required
        if index < nn or index + nn > len(self.segment):
            logger.info(f"[_sample_backbone] Not enough (<{nn}) neighbor residues: "
                        f"lower {index < nn}, upper {index + nn > len(self.segment)}")
            return
        segment = self.segment[(index - nn):(index + nn + 1)]

        # We will work on CB for all residues, but O for GLY.
        atom_name = "CB"
        if self.residue.resn[0] == "GLY":
            atom_name = "O"

        # Determine directions for backbone sampling
        atom = self.residue.extract('name', atom_name)
        try:
            u_matrix = [[atom.u00[0], atom.u01[0], atom.u02[0]],
                        [atom.u01[0], atom.u11[0], atom.u12[0]],
                        [atom.u02[0], atom.u12[0], atom.u22[0]]]
            directions = adp_ellipsoid_axes(u_matrix)
            logger.debug(f"[_sample_backbone] u_matrix = {u_matrix}")
            logger.debug(f"[_sample_backbone] directions = {directions}")
        except AttributeError:
            logger.info(f"[{self.identifier}] Got AttributeError for directions at Cβ. Treating as isotropic B, using x,y,z vectors.")
            # TODO: Probably choose to put one of these as Cβ-Cα, C-N, and then (Cβ-Cα × C-N)
            directions = np.identity(3)

        # If we are missing a backbone atom in our segment,
        #     use current coords for this residue, and abort.
        for n, residue in enumerate(self.segment.residues[::-1]):
            for backbone_atom in ['N', 'CA', 'C', 'O']:
                if backbone_atom not in residue.name:
                    relative_to_residue = n - index
                    logger.warning(f"[{self.identifier}] Missing backbone atom in segment residue {relative_to_residue:+d}.")
                    logger.warning(f"[{self.identifier}] Skipping backbone sampling.")
                    self._coor_set.append(self.segment[index].coor)
                    self._bs.append(self.conformer.b)
                    return

        # Retrieve the amplitudes and stepsizes from options.
        sigma = self.options.sample_backbone_sigma
        bba, bbs = self.options.sample_backbone_amplitude, self.options.sample_backbone_step
        assert bba >= bbs > 0

        # Create an array of amplitudes to scan:
        #   We start from stepsize, making sure to stop only after bba.
        #   Also include negative amplitudes.
        eps = ((bba / bbs) / 2) * np.finfo(float).epsneg  # ε to avoid FP errors in arange
        amplitudes = np.arange(start=bbs, stop=bba + bbs - eps, step=bbs)
        amplitudes = np.concatenate([-amplitudes[::-1], amplitudes])

        # Optimize in torsion space to achieve the target atom position
        optimizer = NullSpaceOptimizer(segment)
        start_coor = atom.coor[0]  # We are working on a single atom.
        torsion_solutions = []
        for amplitude, direction in itertools.product(amplitudes, directions):
            delta = self.prng.uniform(-sigma, sigma)
            endpoint = start_coor + (amplitude + delta) * direction
            optimize_result = optimizer.optimize(atom_name, endpoint)
            torsion_solutions.append(optimize_result['x'])

        # Capture starting coordinates for the segment, so that we can restart after every rotator
        starting_coor = segment.coor
        for solution in torsion_solutions:
            optimizer.rotator(solution)
            self._coor_set.append(self.segment[index].coor)
            self._bs.append(self.conformer.b)
            segment.coor = starting_coor

        logger.debug(f"[_sample_backbone] Backbone sampling generated {len(self._coor_set)} conformers.")
        if self.options.write_intermediate_conformers:
            self._write_intermediate_conformers(prefix=f"_sample_backbone_segment{index:03d}")

    def _sample_angle(self):
        """Sample residue conformations by flexing α-β-γ angle.

        Only operates on residues with large aromatic sidechains
            (Trp, Tyr, Phe, His) where CG is a member of the aromatic ring.
        Here, slight deflections of the ring are likely to lead to better-
            scoring conformers when we scan χ(Cα-Cβ) and χ(Cβ-Cγ) later.

        This angle does not exist in {Gly, Ala}, and it does not make sense to
            sample this angle in Pro.

        We choose not to do this sampling on smaller amino acids for reason of
            a trade-off in complexity. Sampling the α-β-γ angle increases the
            amount & density of sampled conformations, but is unlikely to yield
            better quality sampling (conformations with good matches to
            electron density). In the case of Arg, the planar aromatic region
            does not start at CG. Later χ angles are more effective at moving
            the guanidinium group.
        """
        # Only operate on aromatics!
        if self.resn not in ("TRP", "TYR", "PHE", "HIS"):
            logger.debug(f"[{self.identifier}] Not F/H/W/Y. Cα-Cβ-Cγ angle sampling skipped.")
            return

        # Limit active atoms
        active_names = ('N', 'CA', 'C', 'O', 'CB', 'H', 'HA', 'CG', 'HB2', 'HB3')
        selection = self.residue.select('name', active_names)
        self.residue.active = False
        self.residue._active[selection] = True
        self.residue.update_clash_mask()
        active_mask = self.residue.active

        # Define sampling range
        angles = np.arange(-self.options.sample_angle_range,
                           self.options.sample_angle_range + self.options.sample_angle_step,
                           self.options.sample_angle_step)

        # Commence sampling, building on each existing conformer in self._coor_set
        new_coor_set = []
        new_bs = []
        for coor in self._coor_set:
            self.residue.coor = coor
            rotator = CBAngleRotator(self.residue)
            for angle in angles:
                rotator(angle)
                coor = self.residue.coor

                # Move on if these coordinates are unsupported by density
                if self.options.remove_conformers_below_cutoff:
                    values = self.xmap.interpolate(coor[active_mask])
                    mask = (self.residue.e[active_mask] != "H")
                    if np.min(values[mask]) < self.options.density_cutoff:
                        continue

                # Move on if these coordinates cause a clash
                if self.options.external_clash:
                    if self._cd() and self.residue.clashes():
                        continue
                elif self.residue.clashes():
                    continue

                # Valid, non-clashing conformer found!
                new_coor_set.append(self.residue.coor)
                new_bs.append(self.conformer.b)

        # Update sampled coords
        self._coor_set = new_coor_set
        self._bs = new_bs
        logger.debug(f"Bond angle sampling generated {len(self._coor_set)} conformers.")
        if self.options.write_intermediate_conformers:
            self._write_intermediate_conformers(prefix=f"_sample_angle")

    def _sample_sidechain(self):
        opt = self.options
        start_chi_index = 1
        if self.residue.resn[0] != 'PRO':
            sampling_window = np.arange(
                -opt.rotamer_neighborhood,
                opt.rotamer_neighborhood + opt.dihedral_stepsize,
                opt.dihedral_stepsize,
            )
        else:
            sampling_window = [0]

        rotamers = self.residue.rotamers
        rotamers.append([self.residue.get_chi(i) for i in range(1, self.residue.nchi + 1)])
        iteration = 0
        new_bs = []
        for b in self._bs:
            new_bs.append(self._randomize_bs(b, ['N', 'CA', 'C', 'O', 'CB', 'H', 'HA']))
        self._bs = new_bs
        while True:
            chis_to_sample = opt.dofs_per_iteration
            if iteration == 0 and (opt.sample_backbone or opt.sample_angle):
                chis_to_sample = max(1, opt.dofs_per_iteration - 1)
            end_chi_index = min(start_chi_index + chis_to_sample,
                                self.residue.nchi + 1)
            iter_coor_set = []
            for chi_index in range(start_chi_index, end_chi_index):
                # Set active and passive atoms, since we are iteratively
                # building up the sidechain. This updates the internal
                # clash mask.
                self.residue.active = True
                if chi_index < self.residue.nchi:
                    current = self.residue._rotamers['chi-rotate'][chi_index]
                    deactivate = self.residue._rotamers['chi-rotate'][chi_index + 1]
                    selection = self.residue.select('name', deactivate)
                    self.residue._active[selection] = False
                    bs_atoms = list(set(current) - set(deactivate))
                else:
                    bs_atoms = self.residue._rotamers['chi-rotate'][chi_index]

                self.residue.update_clash_mask()
                active = self.residue.active

                logger.info(f"Sampling chi: {chi_index} ({self.residue.nchi})")
                new_coor_set = []
                new_bs = []
                n = 0
                ex = 0
                # For each backbone conformation so far:
                for coor, b in zip(self._coor_set, self._bs):
                    self.residue.coor = coor
                    self.residue.b = b
                    chis = [self.residue.get_chi(i) for i in range(1, chi_index)]
                    # Try each rotamer in the library for this backbone conformation:
                    for rotamer in rotamers:
                        # Check if the current sidechain configuration for this residue 
                        # closely matches the rotamer being considered from the library
                        is_this_same_rotamer = True
                        for curr_chi, rotamer_chi in zip(chis, rotamer):
                            diff_chi = abs(curr_chi - rotamer_chi)
                            if 360 - opt.rotamer_neighborhood > diff_chi > opt.rotamer_neighborhood:
                                is_this_same_rotamer = False
                                break
                        if not is_this_same_rotamer:
                            continue
                        # Set the chi angle to the standard rotamer value.
                        self.residue.set_chi(chi_index, rotamer[chi_index - 1])

                        # Sample around the neighborhood of the rotamer
                        chi_rotator = ChiRotator(self.residue, chi_index)

                        for angle in sampling_window:
                            # Rotate around the chi angle, hitting each of the angle values
                            # in our predetermined, generic chi-angle sampling window
                            n += 1
                            chi_rotator(angle)
                            coor = self.residue.coor
                            
                            # See if this (partial) conformer clashes, 
                            # based on a density mask
                            if opt.remove_conformers_below_cutoff:
                                values = self.xmap.interpolate(coor[active])
                                mask = (self.residue.e[active] != "H")
                                if np.min(values[mask]) < self.options.density_cutoff:
                                    ex += 1
                                    continue
                            
                            # See if this (partial) conformer clashes (so far), 
                            # based on all-atom sterics (if the user wanted that)
                            keep_coor_set = False
                            if self.options.external_clash:
                                if not self._cd() and self.residue.clashes() == 0:
                                    keep_coor_set = True
                            elif self.residue.clashes() == 0:
                                keep_coor_set = True

                            # Based on that, decide whether to keep or reject this (partial) conformer
                            if keep_coor_set:
                                if new_coor_set:
                                    delta = np.array(new_coor_set) - np.array(self.residue.coor)
                                    if np.sqrt(min(np.square((delta)).sum(axis=2).sum(axis=1))) >= 0.01:
                                        new_coor_set.append(self.residue.coor)
                                        new_bs.append(self._randomize_bs(b, bs_atoms))
                                    else:
                                        ex += 1
                                else:
                                    new_coor_set.append(self.residue.coor)
                                    new_bs.append(self._randomize_bs(b, bs_atoms))
                            else:
                                ex += 1

                iter_coor_set.append(new_coor_set)
                self._coor_set = new_coor_set
                self._bs = new_bs

            if len(self._coor_set) > 15000:
                logger.warning(f"[{self.identifier}] Too many conformers generated ({len(self._coor_set)}). "
                               f"Reverting to a previous iteration of degrees of freedom: item 0. "
                               f"n_coords: {[len(cs) for (cs) in iter_coor_set]}")
                self._coor_set = iter_coor_set[0]

            if not self._coor_set:
                msg = ("No conformers could be generated. Check for initial "
                       "clashes and density support.")
                raise RuntimeError(msg)

            logger.debug(f"Side chain sampling generated {len(self._coor_set)} conformers")
            if self.options.write_intermediate_conformers:
                self._write_intermediate_conformers(prefix=f"_sample_sidechain_iter{iteration}")

            # QP score conformer occupancy
            self._convert()
            self._solve()
            self._update_conformers()
            if self.options.write_intermediate_conformers:
                self._write_intermediate_conformers(prefix=f"_sample_sidechain_iter{iteration}_qp")

            # MIQP score conformer occupancy
            self._convert()
            self._solve(threshold=self.options.threshold,
                        cardinality=self.options.cardinality)
            self._update_conformers()
            if self.options.write_intermediate_conformers:
                self._write_intermediate_conformers(prefix=f"_sample_sidechain_iter{iteration}_miqp")

            # Check if we are done
            if chi_index == self.residue.nchi:
                break

            # Use the next chi angle as starting point, except when we are in
            # the first iteration and have selected backbone sampling and we
            # are sampling more than 1 dof per iteration
            increase_chi = not ((opt.sample_backbone or opt.sample_angle)
                                and iteration == 0
                                and opt.dofs_per_iteration > 1)
            if increase_chi:
                start_chi_index += 1
            iteration += 1

    def tofile(self):
        # Save the individual conformers
        conformers = self.get_conformers()
        for n, conformer in enumerate(conformers, start=1):
            fname = os.path.join(self.directory_name, f'conformer_{n}.pdb')
            conformer.tofile(fname)

        # Make a multiconformer residue
        nconformers = len(conformers)
        if nconformers < 1:
            msg = ("No conformers could be generated. "
                   "Check for initial clashes.")
            raise RuntimeError(msg)
        mc_residue = Structure.fromstructurelike(conformers[0])
        if nconformers == 1:
            mc_residue.altloc = ''
        else:
            mc_residue.altloc = 'A'
            for altloc, conformer in zip(ascii_uppercase[1:], conformers[1:]):
                conformer.altloc = altloc
                mc_residue = mc_residue.combine(conformer)
        mc_residue = mc_residue.reorder()

        # Save the multiconformer residue
        logger.info(f"[{self.identifier}] Saving multiconformer_residue.pdb")
        fname = os.path.join(self.directory_name, f"multiconformer_residue.pdb")
        mc_residue.tofile(fname)


class QFitSegment(_BaseQFit):
    """Determines consistent protein segments based on occupancy and
       density fit"""
    def __init__(self, structure, xmap, options):
        self.segment = structure
        self.conformer = structure
        # self.conformer.q = 1
        self.xmap = xmap
        self.options = options
        self.options.bic_threshold = self.options.seg_bic_threshold
        self.fragment_length = options.fragment_length
        self.BIC = np.inf
        self._coor_set = [self.conformer.coor]
        self._occupancies = [self.conformer.q]
        self._bs = [self.conformer.b]
        self.orderings = []
        self.charseq = []
        self._smax = None
        self._simple = True
        self._rmask = 1.5
        reso = None
        if self.xmap.resolution.high is not None:
            reso = self.xmap.resolution.high
        elif options.resolution is not None:
            reso = options.resolution
        if reso is not None:
            self._smax = 1 / (2 * reso)
            self._simple = False
            self._rmask = 0.5 + reso / 3.0

        self._smin = None
        if self.xmap.resolution.low is not None:
            self._smin = 1 / (2 * self.xmap.resolution.low)
        elif self.options.resolution_min is not None:
            self._smin = 1 / (2 * options.resolution_min)

        self._xmap_model = xmap.zeros_like(self.xmap)
        # To speed up the density creation steps, reduce space group symmetry
        # to P1
        self._xmap_model.set_space_group("P1")
        self._voxel_volume = self.xmap.unit_cell.calc_volume()
        self._voxel_volume /= self.xmap.array.size

    def __call__(self):
        logger.info(f"Average number of conformers before qfit_segment run: "
                    f"{self.segment.average_conformers():.2f}")
        # Extract hetatms
        hetatms = self.segment.extract('record', "HETATM")
        # Create an empty structure:
        multiconformers = Structure.fromstructurelike(self.segment.extract('altloc', "Z"))
        segment = []

        # Construct progress iterator
        residue_groups = self.segment.extract("record", "ATOM").residue_groups
        residue_groups_pbar = tqdm.tqdm(residue_groups,
                                        total=self.segment.n_residues,
                                        desc="Building segments",
                                        unit="res",
                                        leave=True)

        # Iterate over all residue groups
        for rg in residue_groups_pbar:
            if rg.resn[0] not in ROTAMERS:
                multiconformers = multiconformers.combine(rg)
                continue
            altlocs = np.unique(rg.altloc)
            naltlocs = len(altlocs)
            multiconformer = []
            CA_single = True
            O_single = True
            CA_pos = None
            O_pos = None
            for altloc in altlocs:
                if altloc == '' and naltlocs > 1:
                    continue
                conformer = Structure.fromstructurelike(rg.extract('altloc',
                                                                   (altloc, '')
                                                                   ))
                # Reproducing the code in idmulti.cpp:
                if (CA_single and O_single):
                    mask = np.isin(conformer.name, ['CA', 'O'])
                    if np.sum(mask) > 2:
                        logger.warning(f"Conformer {altloc} of residue "
                                       f"{rg.resi[0]} has more than one coordinate "
                                       f"for CA/O atoms.")
                        mask = mask[:2]
                    try:
                        CA_single = np.linalg.norm(CA_pos - conformer.coor[mask][0])
                        CA_single = CA_single <= 0.05
                        O_single = np.linalg.norm(O_pos - conformer.coor[mask][1])
                        O_single = O_single <= 0.05
                    except TypeError:
                        CA_pos, O_pos = [coor for coor in conformer.coor[mask]]
                multiconformer.append(conformer)

            # Check to see if the residue has a single conformer:
            if naltlocs == 1:
                # Process the existing segment
                if len(segment):
                    for path in self.find_paths(segment):
                        multiconformers = multiconformers.combine(path)
                segment = []
                # Set the occupancy of all atoms of the residue to 1
                rg.q = np.ones_like(rg.q)
                # Add the single conformer residue to the
                # existing multiconformer:
                multiconformers = multiconformers.combine(rg)

            # Check if we need to collapse the backbone
            elif CA_single and O_single:
                # Process the existing segment
                if len(segment):
                    for path in self.find_paths(segment):
                        multiconformers = multiconformers.combine(path)
                segment = []
                collapsed = multiconformer[:]
                for multi in collapsed:
                    multiconformers = multiconformers.combine(multi.collapse_backbone(multi.resi[0],
                                                                                      multi.chain[0]))

            else:
                segment.append(multiconformer)

        # Teardown progress bar
        residue_groups_pbar.close()

        if len(segment):
            logger.debug(f"Running find_paths for segment of length {len(segment)}")
            for path in self.find_paths(segment):
                multiconformers = multiconformers.combine(path)

        logger.info(f"Average number of conformers after qfit_segment run: "
                    f"{multiconformers.average_conformers():.2f}")
        multiconformers = multiconformers.reorder()
        multiconformers = multiconformers.remove_identical_conformers(self.options.rmsd_cutoff)
        logger.info(f"Average number of conformers after removal of identical conformers: "
                    f"{multiconformers.average_conformers():.2f}")

        # Build an instance of Relabeller
        relab_options = RelabellerOptions()
        relab_options.apply_command_args(self.options)  # Update RelabellerOptions with QFitSegmentOptions
        relabeller = Relabeller(multiconformers, relab_options)
        multiconformers = relabeller.run()
        multiconformers = multiconformers.combine(hetatms)
        multiconformers = multiconformers.reorder()
        return multiconformers

    def find_paths(self, segment_original):
        segment = segment_original[:]
        fl = self.fragment_length
        possible_conformers = list(map(chr, range(65, 90)))
        possible_conformers = possible_conformers[0:int(round(1. / self.options.threshold))]

        while len(segment) > 1:
            n = len(segment)

            fragment_multiconformers = [segment[i: i + fl] for i in range(0, n, fl)]
            segment = []
            for fragment_multiconformer in fragment_multiconformers:
                fragments = []
                # Create all combinations of alternate residue conformers
                for fragment_conformer in itertools.product(*fragment_multiconformer):
                    fragment = fragment_conformer[0].set_backbone_occ()
                    for element in fragment_conformer[1:]:
                        fragment = fragment.combine(element.set_backbone_occ())
                    combine = True
                    for fragment2 in fragments:
                        diff = (fragment.coor - fragment2.coor).ravel()
                        if np.sqrt(3 * np.inner(diff, diff) / diff.size) < np.min([0.005 * diff.size, 0.3]):
                            combine = False
                            break
                    if combine:
                        fragments.append(fragment)

                # We have the fragments, select consistent optimal set
                self._update_transformer(fragments[0])
                self._coor_set = [fragment.coor for fragment in fragments]
                self._bs = [fragment.b for fragment in fragments]

                # QP score segment occupancy
                self._convert()
                self._solve()

                # Update conformers
                fragments = np.array(fragments)
                mask = self._occupancies >= 0.002
                fragments = fragments[mask]
                self._occupancies = self._occupancies[mask]
                self._coor_set = [fragment.coor for fragment in fragments]
                self._bs = [fragment.b for fragment in fragments]

                # MIQP score segment occupancy
                self._convert()
                self._solve(threshold=self.options.threshold,
                            cardinality=self.options.cardinality,
                            loop_range=[0.34, 0.25, 0.2, 0.16, 0.14])

                # Update conformers
                mask = self._occupancies >= 0.002
                for fragment, occ in zip(fragments[mask],
                                         self._occupancies[mask]):
                    fragment.q = occ
                segment.append(fragments[mask])

        for path, altloc in zip(segment[0], possible_conformers):
            path.altloc = altloc
        return segment[0]

    def print_paths(self, segment):
        # Calculate the path matrix:
        for k, fragment in enumerate(segment):
            path = []
            for i, residue_altloc in enumerate(fragment.altloc):
                if fragment.name[i] == "CA":
                    path.append(residue_altloc)
                    # coor.append(fragment.coor[i])
            logger.info(f"Path {k+1}:\t{path}\t{fragment.q[-1]}")


class QFitLigand(_BaseQFit):
    def __init__(self, ligand, receptor, xmap, options):
        # Initialize using the base qfit class
        super().__init__(ligand, receptor, xmap, options)

        # These lists will be used to combine coor_sets output for
        # each of the clusters that we sample:
        self._all_coor_set = []
        self._all_bs = []

        # Populate useful attributes:
        self.ligand = ligand
        self.receptor = receptor
        self.xmap = xmap
        self.options = options
        csf = self.options.clash_scaling_factor
        self._trans_box = [(-0.2, 0.21, 0.1)] * 3
        self._bs = [self.ligand.b]

        # External clash detection:
        self._cd = ClashDetector(ligand, receptor, scaling_factor=self.options.clash_scaling_factor)

        # Determine which roots to start building from
        self._rigid_clusters = ligand.rigid_clusters()
        self.roots = None
        if self.roots is None:
            self._clusters_to_sample = []
            for cluster in self._rigid_clusters:
                nhydrogen = (self.ligand.e[cluster] == 'H').sum()
                if len(cluster) - nhydrogen > 1:
                    self._clusters_to_sample.append(cluster)
        logger.debug(f"Number of clusters to sample: {len(self._clusters_to_sample)}")

        # Initialize the transformer
        if options.subtract:
            self._subtract_transformer(self.ligand, self.receptor)
        self._update_transformer(self.ligand)
        self._starting_coor_set = [ligand.coor.copy()]
        self._starting_bs = [ligand.b.copy()]

    def run(self):
        for self._cluster_index, self._cluster in enumerate(self._clusters_to_sample):
            self._coor_set = list(self._starting_coor_set)
            if self.options.local_search:
                logger.info("Starting local search")
                self._local_search()
            self._coor_set.append(self._starting_coor_set)
            self.ligand._active[self.ligand._selection] = True
            logger.info("Starting sample internal dofs")
            self._sample_internal_dofs()
            self._all_coor_set += self._coor_set
            self._all_bs += self._bs
            prefix_tmp = 'run_' + str(self._cluster)
            self._write_intermediate_conformers(prefix=prefix_tmp)
            logger.info(f"Number of conformers: {len(self._coor_set)}")
            logger.info(f"Number of final conformers: {len(self._all_coor_set)}")

        # Find consensus across roots:
        self.conformer = self.ligand
        self.ligand._q[self.ligand._selection] = 1.0
        self.ligand._active[self.ligand._selection] = True
        self._coor_set = self._all_coor_set
        self._bs = self._all_bs
        if len(self._coor_set) < 1:
            logger.error("qFit-ligand failed to produce a valid conformer.")
            return

        # QP score conformer occupancy
        logger.debug("Converting densities within run.")
        self._convert()
        logger.info("Solving QP within run.")
        self._solve()
        logger.debug("Updating conformers within run.")
        self._update_conformers()
        if len(self._coor_set) < 1:
            print(f"{self.ligand.resn[0]}: QP {self._cluster_index}: {len(self._coor_set)} conformers")
            return

        # MIQP score conformer occupancy
        logger.info("Solving MIQP within run.")
        self._convert()
        self._solve(threshold=self.options.threshold,
                    cardinality=self.options.cardinality)
        self._update_conformers()
        if self.options.write_intermediate_conformers:
            self._write_intermediate_conformers(prefix="miqp_solution")

    def _local_search(self):
        """Perform a local rigid body search on the cluster."""

        # Set occupancies of rigid cluster and its direct neighboring atoms to
        # 1 for clash detection and MIQP
        selection = self.ligand._selection
        self.ligand._active[selection] = True
        center = self.ligand.coor[self._cluster].mean(axis=0)
        new_coor_set = []
        new_bs = []
        for coor, b in zip(self._coor_set, self._bs):
            self.ligand._coor[selection] = coor
            self.ligand._b[selection] = b
            rotator = GlobalRotator(self.ligand, center=center)
            for rotmat in RotationSets.get_local_set():
                rotator(rotmat)
                translator = Translator(self.ligand)
                iterator = itertools.product(*[
                    np.arange(*trans) for trans in self._trans_box])
                for translation in iterator:
                    translator(translation)
                    new_coor = self.ligand.coor
                    if self.options.remove_conformers_below_cutoff:
                        values = self.xmap.interpolate(new_coor)
                        mask = (self.ligand.e != "H")
                        if np.min(values[mask]) < self.options.density_cutoff:
                            continue
                    if self.options.external_clash:
                        if not self._cd() and not self.ligand.clashes():
                            if new_coor_set:
                                delta = np.array(new_coor_set) - np.array(new_coor)
                                if np.sqrt(min(np.square((delta)).sum(axis=2).sum(axis=1))) >= self.options.rmsd_cutoff:
                                    new_coor_set.append(new_coor)
                                    new_bs.append(b)
                            else:
                                new_coor_set.append(new_coor)
                                new_bs.append(b)
                    elif not self.ligand.clashes():
                        if new_coor_set:
                            delta = np.array(new_coor_set) - np.array(new_coor)
                            if np.sqrt(min(np.square((delta)).sum(axis=2).sum(axis=1))) >= self.options.rmsd_cutoff:
                                new_coor_set.append(new_coor)
                                new_bs.append(b)
                        else:
                            new_coor_set.append(new_coor)
                            new_bs.append(b)
        self.ligand._active[self.ligand._selection] = False
        selection = self.ligand._selection[self._cluster]
        self.ligand._active[selection] = True
        for atom in self._cluster:
            atom_sel = self.ligand._selection[self.ligand.connectivity[atom]]
            self.ligand._active[atom_sel] = True
        self.conformer = self.ligand
        self._coor_set = new_coor_set
        self._bs = new_bs
        if len(self._coor_set) < 1:
            logger.warning(f"{self.ligand.resn[0]}: "
                           f"Local search {self._cluster_index}: {len(self._coor_set)} conformers")
            return

        # QP score conformer occupancy
        logger.debug("Converting densities.")
        self._convert()
        self._solve()
        logger.debug("Updating conformers")
        self._update_conformers()
        if self.options.write_intermediate_conformers:
            self._write_intermediate_conformers(prefix="_localsearch_ligand_qp")
        if len(self._coor_set) < 1:
            logger.warning(f"{self.ligand.resn[0]}: "
                           f"Local search QP {self._cluster_index}: {len(self._coor_set)} conformers")
            return

        # MIQP score conformer occupancy
        self._convert()
        self._solve(threshold=self.options.threshold,
                    cardinality=self.options.cardinality)
        self._update_conformers()
        if self.options.write_intermediate_conformers:
            self._write_intermediate_conformers(prefix="_localsearch_ligand_miqp")

    def _sample_internal_dofs(self):
        opt = self.options
        sampling_range = np.deg2rad(np.arange(0, 360, self.options.sample_ligand_stepsize))

        # bond_order = self.ligand.rotation_order(self._cluster[0])
        # bond_list = self.ligand.convert_rotation_tree_to_list(bond_order)
        # nbonds = len(bond_list)
        # if nbonds == 0:
        #     return
        bond_order = BondOrder(self.ligand, self._cluster[0])
        bonds = bond_order.order
        depths = bond_order.depth
        nbonds = len(bonds)

        starting_bond_index = 0

        sel_str = f"chain {self.ligand.chain[0]} and resi {self.ligand.resi[0]}"
        if self.ligand.icode[0]:
            sel_str = f"{sel_str} and icode {self.ligand.icode[0]}"
        selection = self.ligand._selection
        iteration = 1
        while True:
            if iteration == 1 and self.options.local_search and self.options.dofs_per_iteration > 1:
                end_bond_index = starting_bond_index + self.options.dofs_per_iteration - 1
            else:
                end_bond_index = min(starting_bond_index + self.options.dofs_per_iteration, nbonds)
            self.ligand._active[selection] = True
            for bond_index in range(starting_bond_index, end_bond_index):
                nbonds_sampled = bond_index + 1

                bond = bonds[bond_index]
                atoms = [self.ligand.name[bond[0]], self.ligand.name[bond[1]]]
                new_coor_set = []
                new_bs = []
                for coor, b in zip(self._coor_set, self._bs):
                    self.ligand._coor[selection] = coor
                    self.ligand._b[selection] = b
                    rotator = BondRotator(self.ligand, *atoms)
                    for angle in sampling_range:
                        new_coor = rotator(angle)
                        if opt.remove_conformers_below_cutoff:
                            values = self.xmap.interpolate(new_coor[self.ligand._active[selection]])
                            mask = (self.ligand.e[self.ligand._active[selection]] != "H")
                            if np.min(values[mask]) < self.options.density_cutoff:
                                continue
                        if self.options.external_clash:
                            if not self._cd() and not self.ligand.clashes():
                                if new_coor_set:
                                    delta = np.array(new_coor_set) - np.array(new_coor)
                                    if np.sqrt(min(np.square((delta)).sum(axis=2).sum(axis=1))) >= self.options.rmsd_cutoff:
                                        new_coor_set.append(new_coor)
                                        new_bs.append(b)
                                else:
                                    new_coor_set.append(new_coor)
                                    new_bs.append(b)
                        elif not self.ligand.clashes():
                            if new_coor_set:
                                delta = np.array(new_coor_set) - np.array(new_coor)
                                if np.sqrt(min(np.square((delta)).sum(axis=2).sum(axis=1))) >= self.options.rmsd_cutoff:
                                    new_coor_set.append(new_coor)
                                    new_bs.append(b)
                            else:
                                new_coor_set.append(new_coor)
                                new_bs.append(b)
                self._coor_set = new_coor_set
                self._bs = new_bs

            self.ligand._active[selection] = False
            active = np.zeros_like(self.ligand._active[selection], dtype=bool)
            # Activate all the atoms of the ligand that have been sampled
            # up until the bond we are currently sampling:
            for cluster in self._rigid_clusters:
                for sampled_bond in bonds[:nbonds_sampled]:
                    if sampled_bond[0] in cluster or sampled_bond[1] in cluster:
                        active[cluster] = True
                        for atom in cluster:
                            active[self.ligand.connectivity[atom]] = True
            self.ligand._active[selection] = active
            self.conformer = self.ligand

            logger.info(f"Nconf: {len(self._coor_set)}")

            if len(self._coor_set) < 1:
                logger.warning(f"{self.ligand.resn[0]}: "
                               f"DOF search cluster {self._cluster_index} iteration {iteration}: "
                               f"{len(self._coor_set)} conformers.")
                return

            # QP score conformer occupancy
            self._convert()
            self._solve()
            self._update_conformers()
            if self.options.write_intermediate_conformers:
                self._write_intermediate_conformers(prefix=f"_sample_ligand_iter{iteration}_qp")
            if len(self._coor_set) < 1:
                logger.warning(f"{self.ligand.resn[0]}: "
                               f"QP search cluster {self._cluster_index} iteration {iteration}: "
                               f"{len(self._coor_set)} conformers")
                return

            # MIQP score conformer occupancy
            self._convert()
            self._solve(threshold=self.options.threshold,
                        cardinality=self.options.cardinality)
            self._update_conformers()
            if self.options.write_intermediate_conformers:
                self._write_intermediate_conformers(prefix=f"_sample_ligand_iter{iteration}_miqp")

            # Check if we are done
            if end_bond_index == nbonds:
                break

            # Go to the next bonds to be sampled
            if iteration == 1 and self.options.local_search and self.options.dofs_per_iteration > 1:
                starting_bond_index += self.options.dofs_per_iteration - 1
            else:
                starting_bond_index += self.options.dofs_per_iteration
            iteration += 1


class QFitCovalentLigand(_BaseQFit):
    def __init__(self, covalent_ligand, receptor, xmap, options):
        self.chain = covalent_ligand.chain[0]
        self.resi = covalent_ligand.resi[0]
        self.covalent_ligand = covalent_ligand
        self._rigid_clusters = covalent_ligand.rigid_clusters()
        if covalent_ligand.covalent_bonds == 0:
            pass
        elif covalent_ligand.covalent_bonds == 1:
            # Extract the information about the residue that the
            # ligand is covalently bonded to:
            partner_chain, partner_resi, \
                partner_icode, self.partner_atom = (covalent_ligand.covalent_partners[0])
            if partner_icode:
                self.partner_id = (int(partner_resi), partner_icode)
            else:
                self.partner_id = int(partner_resi)

            # Extract the information about the covalent bond itself:
            self.covalent_atom = covalent_ligand.covalent_atoms[0][3]
            self.covalent_bond = [self.partner_atom, self.covalent_atom]

            # Select the partner residue from the receptor:
            sel_str = f"chain {partner_chain} and resi {partner_resi}"
            if partner_icode:
                sel_str = f"{sel_str} and icode {partner_icode}"
            self.covalent_partner = receptor.extract(sel_str)

            # Alter the structure so that covalent ligand and residue are
            # treated as a single residue:
            data = {}
            for attr in receptor.data:
                data[attr] = np.concatenate((
                    getattr(receptor, attr),
                    getattr(covalent_ligand, attr)))
            mask = (data['resi'] == covalent_ligand.resi[0]) & (
                data['chain'] == covalent_ligand.chain[0])
            data['resi'][mask] = self.covalent_partner.resi[0]
            data['chain'][mask] = self.covalent_partner.chain[0]
            data['resn'][mask] = self.covalent_partner.resn[0]
            self.structure = Structure(data)

            # Extract the covalent residue as a Residue object:
            combined_residue = self.structure.extract(sel_str)
            chain = combined_residue[partner_chain]
            conformer = chain.conformers[0]
            self.covalent_residue = conformer[self.partner_id]

            # Initialize using the base qFit class:
            super().__init__(self.covalent_residue, self.structure, xmap, options)

            # Update the transformer:
            self._update_transformer(self.covalent_residue)

            # Identify the bonds in the residue for internal clash detection:
            bonds = covalent_ligand.get_bonds()
            # Add the covalent bond itself to the list:
            bonds.append(self.covalent_bond)
            # Initialize internal clash detection
            self.covalent_residue._init_clash_detection(self.options.clash_scaling_factor, bonds)

            # Identify the segment to which the covalent residue belongs to:
            self.segment = None
            for segment in self.structure.segments:
                if segment.chain[0] == partner_chain and self.covalent_residue in segment:
                    self.segment = segment
                    break
            if self.segment is None:
                raise RuntimeError("Could not determine segment.")

            # Set up external clash detection:
            self._setup_clash_detector()
            if options.subtract:
                self._subtract_transformer(self.covalent_residue, self.structure)
            self._update_transformer(self.covalent_residue)

        # More than one covalent bond:
        else:
            pass

    def _setup_clash_detector(self):
        residue = self.covalent_residue
        segment = self.segment
        index = segment.find(self.partner_id)
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
        # Obtain atoms with which the residue can clash
        resi = self.partner_id
        chainid = self.segment.chain[0]
        sel_str = f'not (resi {resi} and chain {chainid})'
        receptor = self.structure.extract(sel_str).copy()
        # Find symmetry mates of the receptor
        starting_coor = self.structure.coor.copy()
        iterator = self.xmap.unit_cell.iter_struct_orth_symops
        for symop in iterator(self.structure,
                              target=self.covalent_residue, cushion=5):
            if symop.is_identity():
                continue
            self.structure.rotate(symop.R)
            self.structure.translate(symop.t)
            receptor = receptor.combine(self.structure)
            self.structure.coor = starting_coor

        self._cd = ClashDetector(residue, receptor, exclude=exclude,
                                 scaling_factor=self.options.clash_scaling_factor)

    def run(self):
        if self.options.sample_backbone:
            self._sample_backbone()
        if self.options.sample_angle:
            # Is the ligand bound to the backbone or the side chain?
            if self.partner_atom not in ['N', 'C', 'CA', 'O']:
                self._sample_angle()
        if self.covalent_residue.nchi >= 1 and self.options.sample_rotamers:
            # Is the ligand bound to the backbone or the side chain?
            if self.partner_atom not in ['N', 'C', 'CA', 'O']:
                self._sample_sidechain()
        if self.options.sample_ligand:
            # Sample around the covalent bond, if rotatable:
            self._sample_covalent_bond()
            # Sample the remaining rotatable bonds of the ligand:
            self._sample_ligand()
        return

    def _sample_backbone(self):
        # Check if residue has enough neighboring residues
        index = self.segment.find(self.partner_id)
        # self.covalent_residue.update_clash_mask()
        nn = self.options.neighbor_residues_required
        if index < nn or index + nn > len(self.segment):
            return

        segment = self.segment[index - nn: index + nn + 1]
        atom_name = "CB"
        if self.covalent_residue.resn[0] == "GLY":
            atom_name = "O"
        atom = self.covalent_residue.extract('name', atom_name)
        try:
            u_matrix = [[atom.u00[0], atom.u01[0], atom.u02[0]],
                        [atom.u01[0], atom.u11[0], atom.u12[0]],
                        [atom.u02[0], atom.u12[0], atom.u22[0]]]
            directions = adp_ellipsoid_axes(u_matrix)
        except AttributeError:
            directions = np.identity(3)

        optimizer = NullSpaceOptimizer(segment)
        start_coor = atom.coor[0]
        torsion_solutions = []
        amplitudes = np.arange(0.1, self.options.sample_backbone_amplitude + 0.01,
                               self.options.sample_backbone_step)
        sigma = self.options.sample_backbone_sigma

        for amplitude, direction in itertools.product(amplitudes, directions):
            endpoint = start_coor + (amplitude + sigma * self.prng.random()) * direction
            optimize_result = optimizer.optimize(atom_name, endpoint)
            torsion_solutions.append(optimize_result['x'])

            endpoint = start_coor - (amplitude + sigma * self.prng.random()) * direction
            optimize_result = optimizer.optimize(atom_name, endpoint)
            torsion_solutions.append(optimize_result['x'])

        starting_coor = segment.coor

        for solution in torsion_solutions:
            optimizer.rotator(solution)
            self._coor_set.append(self.segment[index].coor)
            segment.coor = starting_coor
        # logger.debug(f"Backbone sampling generated {len(self._coor_set)} conformers")

    def _sample_angle(self):
        """Sample residue along the N-CA-CB angle."""
        active_names = ('N', 'CA', 'C', 'O', 'CB', 'H', 'HA')
        selection = self.covalent_residue.select('name', active_names)
        self.covalent_residue._active = False
        self.covalent_residue._active[selection] = True
        self.covalent_residue.update_clash_mask()
        active = self.covalent_residue.active
        angles = np.arange(-self.options.sample_angle_range,
                            self.options.sample_angle_range + 0.001,
                            self.options.sample_angle_step)
        new_coor_set = []
        new_bs = []
        for coor in self._coor_set:
            self.covalent_residue.coor = coor
            rotator = CBAngleRotator(self.covalent_residue)
            for angle in angles:
                rotator(angle)
                coor = self.covalent_residue.coor
                if self.options.remove_conformers_below_cutoff:
                    values = self.xmap.interpolate(coor[active])
                    mask = (self.covalent_residue.e[active] != "H")
                    if np.min(values[mask]) < self.options.density_cutoff:
                        continue
                if self.options.external_clash:
                    if self._cd() or self.covalent_residue.clashes():
                        continue
                elif self.covalent_residue.clashes():
                    continue
                new_coor_set.append(self.covalent_residue.coor)
                new_bs.append(self.conformer.b)

        self._coor_set = new_coor_set
        self._bs = new_bs
        # logger.debug(f"Bond angle sampling generated {len(self._coor_set)} conformers.")

    def _sample_sidechain(self):
        opt = self.options
        start_chi_index = 1
        partner_length = self.covalent_partner.name.shape[0]
        if self.covalent_residue.resn[0] != 'PRO':
            sampling_window = np.arange(
                -opt.rotamer_neighborhood,
                opt.rotamer_neighborhood + opt.dihedral_stepsize,
                opt.dihedral_stepsize)
        else:
            sampling_window = [0]

        rotamers = self.covalent_residue.rotamers
        rotamers.append([self.covalent_residue.get_chi(i) for i in range(1,
                        self.covalent_residue.nchi + 1)])
        iteration = 0
        new_bs = []
        for b in self._bs:
            new_bs.append(self._randomize_bs(b, ['N', 'CA', 'C', 'O', 'CB', 'H', 'HA']))
        self._bs = new_bs

        while True:
            chis_to_sample = opt.dofs_per_iteration
            if iteration == 0 and (opt.sample_backbone or opt.sample_angle):
                chis_to_sample = max(1, opt.dofs_per_iteration - 1)
            end_chi_index = min(start_chi_index + chis_to_sample,
                                self.covalent_residue.nchi + 1)
            iter_coor_set = []
            for chi_index in range(start_chi_index, end_chi_index):
                # Set active and passive atoms, since we are iteratively
                # building up the sidechain. This updates the internal
                # clash mask.
                self.covalent_residue.active = True
                if chi_index < self.covalent_residue.nchi:
                    deactivate = self.covalent_residue._rotamers['chi-rotate'][chi_index + 1]
                    selection = self.covalent_residue.select('name', deactivate)
                    self.covalent_residue._active[selection] = False
                    bs_atoms = list(set(current) - set(deactivate))
                else:
                    bs_atoms = self.covalent_residue._rotamers['chi-rotate'][chi_index]
                if self.options.sample_ligand:
                    sel_str = f"chain {self.covalent_residue.chain[0]} " \
                              f"and resi {self.covalent_residue.resi[0]}"
                    if self.covalent_residue.icode[0]:
                        sel_str = f"{sel_str} and icode {self.covalent_residue.icode[0]}"
                    selection = self.covalent_residue.select(sel_str)
                    tmp = copy.deepcopy(self.covalent_residue.active)
                    tmp[partner_length:] = False
                    self.covalent_residue._active[selection] = tmp

                active = self.covalent_residue.active
                self.covalent_residue.update_clash_mask()

                new_coor_set = []
                new_bs = []
                n = 0
                for coor, b in zip(self._coor_set, self._bs):
                    n += 1
                    self.covalent_residue.coor = coor
                    self.covalent_residue.b = b
                    chis = [self.covalent_residue.get_chi(i) for i in range(
                            1, chi_index)]
                    for rotamer in rotamers:
                        # Check if the residue configuration corresponds to the
                        # current rotamer
                        is_this_rotamer = True
                        for curr_chi, rotamer_chi in zip(chis, rotamer):
                            diff_chi = abs(curr_chi - rotamer_chi)
                            if 360 - opt.rotamer_neighborhood > diff_chi > opt.rotamer_neighborhood:
                                is_this_rotamer = False
                                break
                        if not is_this_rotamer:
                            continue
                        # Set the chi angle to the standard rotamer value.
                        self.covalent_residue.set_chi(chi_index,
                                                      rotamer[chi_index - 1],
                                                      covalent=self.partner_atom,
                                                      length=partner_length)

                        # Sample around the neighborhood of the rotamer
                        chi_rotator = ChiRotator(self.covalent_residue,
                                                 chi_index,
                                                 covalent=self.partner_atom,
                                                 length=partner_length)

                        for angle in sampling_window:
                            n += 1
                            chi_rotator(angle)
                            atoms = self.covalent_residue.name
                            atom_selection = self.covalent_residue.select('name', atoms)
                            coor = self.covalent_residue._coor[atom_selection]
                            if opt.remove_conformers_below_cutoff:
                                values = self.xmap.interpolate(coor[active])
                                mask = (self.covalent_residue.e[active] != "H")
                                if np.min(values[mask]) < self.options.density_cutoff:
                                    continue
                            if self.options.external_clash:
                                if not self._cd() and not self.covalent_residue.clashes():
                                    if new_coor_set:
                                        delta = np.array(new_coor_set) - np.array(coor)
                                        if np.sqrt(min(np.square((delta)).sum(axis=2).sum(axis=1))) >= 0.01:
                                            new_coor_set.append(coor)
                                            new_bs.append(self._randomize_bs(b, bs_atoms))
                                    else:
                                        new_coor_set.append(coor)
                                        new_bs.append(self._randomize_bs(b, bs_atoms))
                            elif self.covalent_residue.clashes() == 0:
                                if new_coor_set:
                                    delta = np.array(new_coor_set) - np.array(coor)
                                    if np.sqrt(min(np.square((delta)).sum(axis=2).sum(axis=1))) >= 0.01:
                                        new_coor_set.append(coor)
                                        new_bs.append(self._randomize_bs(b, bs_atoms))
                                else:
                                    new_coor_set.append(coor)
                                    new_bs.append(self._randomize_bs(b, bs_atoms))

                self._coor_set = new_coor_set
                self._bs = new_bs

            if not self._coor_set:
                msg = ("No conformers could be generated. Check for initial "
                       "clashes and density support.")
                raise RuntimeError(msg)
            else:
                logger.info(f"Side chain sampling produced {len(self._coor_set)} conformers")

            if self.options.write_intermediate_conformers:
                self._write_intermediate_conformers(prefix=f"_sample_sidechain_iter{iteration}")

            # QP score conformer occupancy
            self._convert()
            self._solve()
            self._update_conformers()
            if self.options.write_intermediate_conformers:
                self._write_intermediate_conformers(prefix=f"_sample_sidechain_iter{iteration}_qp")

            # MIQP score conformer occupancy
            self._convert()
            self._solve(threshold=self.options.threshold,
                        cardinality=self.options.cardinality)
            self._update_conformers()
            if self.options.write_intermediate_conformers:
                self._write_intermediate_conformers(prefix=f"_sample_sidechain_iter{iteration}_miqp")

            # Check if we are done
            if chi_index == self.covalent_residue.nchi:
                break
            # Use the next chi angle as starting point, except when we are in
            # the first iteration and have selected backbone sampling and we
            # are sampling more than 1 dof per iteration
            increase_chi = not ((opt.sample_backbone or opt.sample_angle)
                                and iteration == 0
                                and opt.dofs_per_iteration > 1)
            if increase_chi:
                start_chi_index += 1
            iteration += 1

    def _sample_covalent_bond(self):
        opt = self.options
        atoms = self.covalent_bond
        partner_length = self.covalent_partner.name.shape[0]
        self._sampling_range = np.deg2rad(np.arange(0, 360, self.options.sample_ligand_stepsize))
        sel_str = f"chain {self.covalent_residue.chain[0]} and resi {self.covalent_residue.resi[0]}"
        if self.covalent_residue.icode[0]:
            sel_str = f"{sel_str} and icode {self.covalent_residue.icode[0]}"
        selection = self.covalent_residue.select(sel_str)
        self.covalent_residue._active[selection] = True
        self.covalent_ligand._active[selection] = True
        bs_atoms = list(set(current))
        new_coor_set = []
        new_bs = []
        n = 0
        for coor, b in zip(self._coor_set, self._bs):
            n += 1
            self.covalent_residue.coor = coor
            self.covalent_residue.b = b
            self.covalent_ligand.coor = coor[partner_length:]
            rotator = CovalentBondRotator(self.covalent_residue,
                                          self.covalent_ligand, *atoms)
            for angle in self._sampling_range:
                n += 1
                new_coor = np.concatenate((coor[:partner_length], rotator(angle)), axis=0)
                if opt.remove_conformers_below_cutoff:
                    values = self.xmap.interpolate(new_coor[active])
                    mask = (self.covalent_residue.e[active] != "H")
                    if np.min(values[mask]) < self.options.density_cutoff:
                        continue
                if self.options.external_clash:
                    if not self._cd() and not self.covalent_residue.clashes():
                        if new_coor_set:
                            delta = np.array(new_coor_set) - np.array(new_coor)
                            if np.sqrt(min(np.square((delta)).sum(axis=2).sum(axis=1))) >= 0.01:
                                new_coor_set.append(new_coor)
                                new_bs.append(self._randomize_bs(b, bs_atoms))
                        else:
                            new_coor_set.append(new_coor)
                            new_bs.append(self._randomize_bs(b, bs_atoms))
                elif self.covalent_residue.clashes() == 0:
                    if new_coor_set:
                        delta = np.array(new_coor_set) - np.array(new_coor)
                        if np.sqrt(min(np.square((delta)).sum(axis=2).sum(axis=1))) >= 0.01:
                            new_coor_set.append(new_coor)
                            new_bs.append(self._randomize_bs(b, bs_atoms))
                    else:
                        new_coor_set.append(new_coor)
                        new_bs.append(self._randomize_bs(b, bs_atoms))
        self._coor_set = new_coor_set
        self._bs = new_bs
        self.conformer = self.covalent_residue

        if not self._coor_set:
            msg = ("No conformers could be generated. Check for initial "
                   "clashes and density support.")
            raise RuntimeError(msg)
        else:
            logger.info(f"Covalent bond angle sampling generated {len(self._coor_set)} conformers.")

        # QP score conformer occupancy
        self._convert()
        self._solve()
        self._update_conformers()
        if self.options.write_intermediate_conformers:
            self._write_intermediate_conformers(prefix="_sample_covalent_bond_qp")

        # MIQP score conformer occupancy
        self._convert()
        self._solve(threshold=self.options.threshold,
                    cardinality=self.options.cardinality)
        self._update_conformers()
        if self.options.write_intermediate_conformers:
            self._write_intermediate_conformers(prefix="_sample_covalent_bond_miqp")

    def _sample_ligand(self):
        opt = self.options

        self._sampling_range = np.deg2rad(np.arange(0, 360, self.options.sample_ligand_stepsize))
        nbonds = len(self.covalent_ligand.bond_list)
        if nbonds == 0:
            return

        # This contains an ordered list of bonds, in which the list order
        # was determined by the tree hierarchy starting at the root bond.
        bonds = self.covalent_ligand.bond_list
        partner_length = self.covalent_partner.name.shape[0]
        root = self.covalent_ligand.root

        sel_str = f"chain {self.covalent_residue.chain[0]} and resi {self.covalent_residue.resi[0]}"
        if self.covalent_residue.icode[0]:
            sel_str = f"{sel_str} and icode {self.covalent_residue.icode[0]}"
        selection = self.covalent_residue.select(sel_str)
        iteration = 1
        starting_bond_index = 0
        while True:
            # Identify the bonds that we are going to sample
            end_bond_index = min(starting_bond_index + self.options.dofs_per_iteration, nbonds)

            # Identify the atoms that are active:
            active = np.ones_like(self.covalent_residue.active, dtype=bool)
            # for bond in bonds[:end_bond_index]:
            #     for cluster in self._rigid_clusters:
            #         if bond[0] in cluster or bond[1] in cluster:
            #             active[partner_length:][cluster] = True
            #             for atom in cluster:
            #                 active[partner_length:][self.covalent_ligand.connectivity[atom]] = True
            self.covalent_residue._active[selection] = active
            self.covalent_ligand._active[selection] = active
            self.covalent_residue.update_clash_mask()

            # Sample each of the bonds in the bond range
            n = 0
            for bond_index in range(starting_bond_index, end_bond_index):
                sampled_bond = bonds[bond_index]
                atoms = [self.covalent_ligand.name[sampled_bond[0]],
                         self.covalent_ligand.name[sampled_bond[1]]]
                new_coor_set = []
                for coor in self._coor_set:
                    self.covalent_residue.coor = coor
                    self.covalent_ligand.coor = coor[partner_length:]
                    rotator = BondRotator(self.covalent_ligand, *atoms)
                    for angle in self._sampling_range:
                        new_coor = np.concatenate((coor[:partner_length], rotator(angle)), axis=0)
                        if opt.remove_conformers_below_cutoff:
                            values = self.xmap.interpolate(new_coor[active])
                            mask = (self.covalent_residue.e[active] != "H")
                            if np.min(values[mask]) < self.options.density_cutoff:
                                continue
                        if self.options.external_clash:
                            if not self._cd() and not self.covalent_residue.clashes():
                                if new_coor_set:
                                    delta = np.array(new_coor_set) - np.array(new_coor)
                                    if np.sqrt(min(np.square((delta)).sum(axis=2).sum(axis=1))) >= 0.01:
                                        new_coor_set.append(new_coor)
                                else:
                                    new_coor_set.append(new_coor)
                        elif self.covalent_residue.clashes() == 0:
                            if new_coor_set:
                                delta = np.array(new_coor_set) - np.array(new_coor)
                                if np.sqrt(min(np.square((delta)).sum(axis=2).sum(axis=1))) >= 0.01:
                                    new_coor_set.append(new_coor)
                            else:
                                new_coor_set.append(new_coor)
                self._coor_set = new_coor_set

            if not self._coor_set:
                msg = ("No conformers could be generated. Check for initial "
                       "clashes and density support.")
                raise RuntimeError(msg)
            else:
                logger.info(f"Ligand sampling {iteration} generated {len(self._coor_set)} conformers")

            # QP score conformer occupancy
            self._convert()
            self._solve()
            self._update_conformers()
            if self.options.write_intermediate_conformers:
                self._write_intermediate_conformers(prefix=f"_sample_ligand_iter{iteration}_qp")

            # MIQP score conformer occupancy
            self._convert()
            self._solve(threshold=self.options.threshold,
                        cardinality=self.options.cardinality)
            self._update_conformers()
            if self.options.write_intermediate_conformers:
                self._write_intermediate_conformers(prefix=f"_sample_ligand_iter{iteration}_miqp")

            # Check if we are done
            if end_bond_index == nbonds:
                break

            # Go to the next bonds to be sampled
            starting_bond_index += self.options.dofs_per_iteration
            iteration += 1

    def get_conformers_covalent(self):
        conformers = []
        for q, coor, b in zip(self._occupancies, self._coor_set, self._bs):
            conformer = self.covalent_residue.copy()
            conformer.coor = coor
            conformer.q = q
            conformer.b = b
            conformers.append(conformer)
        return conformers
