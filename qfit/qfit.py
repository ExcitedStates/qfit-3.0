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

import itertools
import logging
import os
import copy
from string import ascii_uppercase
import subprocess
import numpy as np

from .backbone import NullSpaceOptimizer, move_direction_adp
from .clash import ClashDetector
from .samplers import ChiRotator, CBAngleRotator
from .solvers import QPSolver, MIQPSolver, QPSolver2, MIQPSolver2
from .structure import Structure
from .transformer import Transformer
from .validator import Validator
from .volume import XMap
from .scaler import MapScaler
from .relabel import RelabellerOptions, Relabeller
from .structure.rotamers import ROTAMERS


logger = logging.getLogger(__name__)


class _BaseQFitOptions:

    def __init__(self):

        # General options
        self.directory = '.'
        self.debug = False
        self.label = None
        self.map = None


        # Density preparation options
        self.density_cutoff = 0.3
        self.density_cutoff_value = -1

        # Density creation options
        self.map_type = None
        self.resolution = None
        self.resolution_min = None
        self.scattering = 'xray'
        self.omit = False
        self.scale = True
        self.randomize_b = False

        # Sampling options
        self.clash_scaling_factor = 0.75
        self.external_clash = False
        self.dofs_per_iteration = 2
        self.dofs_stepsize = 8
        self.hydro = False

        # MIQP options
        self.cplex = True
        self.cardinality = None
        self.threshold = 0.20
        self.bic_threshold = False
        self.seg_bic_threshold = True

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
        self.sample_backbone_amplitude = 0.30
        self.sample_backbone_step = 0.1
        self.sample_backbone_sigma = 0.125

        # N-CA-CB angle sampling
        self.sample_angle = False
        self.sample_angle_range = 7.5
        self.sample_angle_step = 3.75

        # Rotamer sampling
        self.sample_rotamers = True
        self.rotamer_neighborhood = 40
        self.remove_conformers_below_cutoff = True

        self.bulk_solvent_level = 0.3

        # Anisotropic refinement using phenix
        self.phenix_aniso = False

        # General settings
        # Exclude certain atoms always during density and mask creation to
        # influence QP / MIQP. Provide a list of atom names, e.g. ['N', 'CA']
        # TODO not implemented
        self.exclude_atoms = None


class _BaseQFit:

    def __init__(self, conformer, structure, xmap, options):
        self.structure = structure
        self.conformer = conformer
        self.conformer.q = 1
        self.xmap = xmap
        self.options = options
        self.BIC = np.inf
        self._coor_set = [self.conformer.coor]
        self._occupancies = [1.0]

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
            simple=self._simple, scattering=self.options.scattering,
            randomize_b=self.options.randomize_b)
        logger.debug("Initializing radial density lookup table.")
        self._transformer.initialize()

    def _convert(self):
        """Convert structures to densities and extract relevant values for
           (MI)QP."""

        logger.info("Converting")
        logger.debug("Masking")
        self._transformer.reset(full=True)
        for n, coor in enumerate(self._coor_set):
            self.conformer.coor = coor
            self._transformer.mask(self._rmask)
        # self._transformer.xmap.tofile(f'mask_{self._n}.ccp4')
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
        # print("Solvent level advice:", solvent_level)
        # print("Scaling factor:", scaling_factor)
        # print("Target sum:", target_sum)
        # print("Model sum:", model_sum)
        # self._transformer.reset(full=True)
        for n, coor in enumerate(self._coor_set):
            self.conformer.coor = coor
            self._transformer.density()
            model = self._models[n]
            model[:] = self._transformer.xmap.array[mask]
            np.maximum(model, self.options.bulk_solvent_level, out=model)
            self._transformer.reset(full=True)

    def _solve(self, cardinality=None, threshold=None,
               loop_range=[0.5, 0.4, 0.33, 0.3, 0.25, 0.2]):
        do_qp = cardinality is threshold is None
        if do_qp:
            if self.options.cplex:
                solver = QPSolver(self._target, self._models)
            else:
                solver = QPSolver2(self._target, self._models)
            solver()
            if self.options.bic_threshold:
                self._occupancies = solver.weights
        else:
            if self.options.cplex:
                solver = MIQPSolver(self._target, self._models)
            else:
                solver = MIQPSolver2(self._target, self._models)

            # Treshold Selection by BIC:
            if self.options.bic_threshold:
                self.BIC = np.inf
                for threshold in loop_range:
                    solver(cardinality=None, threshold=threshold)
                    rss = solver.obj_value * self._voxel_volume
                    confs = np.sum(solver.weights >= 0.002)
                    n = len(self._target)
                    try:
                        natoms = len(self.residue._rotamers['atoms'])
                        k = 4 * confs * natoms
                    except AttributeError:
                        k = 4 * confs
                    BIC = n * np.log(rss / n) + k * np.log(n)
                    if BIC < self.BIC:
                        self.BIC = BIC
#                    else:
#                        break
                self._occupancies = solver.weights
            else:
                solver(cardinality=cardinality, threshold=threshold)
        if not self.options.bic_threshold:
            self._occupancies = solver.weights

        # logger.info(f"Residual under footprint: {residual:.4f}")
        # residual = 0
        return solver.obj_value

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
        # fname = os.path.join(self.options.directory, f'mask.{ext}')
        # self._transformer.xmap.tofile(fname)
        # mask = self._transformer.xmap.array > 0
        # self._transformer.reset(full=True)

        for q, coor in zip(self._occupancies, self._coor_set):
            self.conformer.q = q
            self.conformer.coor = coor
            self._transformer.density()
        fname = os.path.join(self.options.directory, f'model.{ext}')
        self._transformer.xmap.tofile(fname)
        self._transformer.xmap.array -= self.xmap.array
        fname = os.path.join(self.options.directory, f'diff.{ext}')
        self._transformer.xmap.tofile(fname)
        self._transformer.reset(full=True)
        # self._transformer.xmap.array *= -1
        # fname = os.path.join(self.options.directory, f'diff_negative.{ext}')
        # self._transformer.xmap.tofile(fname)

        # self._transformer.reset(full=True)
        # self._transformer.xmap.array[mask] = values
        # fname = os.path.join(self.options.directory, f'model_masked.{ext}')
        # self._transformer.xmap.tofile(fname)
        # values = self.xmap.array[mask]
        # self._transformer.xmap.array[mask] -= values
        # fname = os.path.join(self.options.directory, f'diff_masked.{ext}')
        # self._transformer.xmap.tofile(fname)


class QFitRotamericResidue(_BaseQFit):

    def __init__(self, residue, structure, xmap, options):
        self.chain = residue.chain[0]
        self.resi = residue.resi[0]
        if options.phenix_aniso:
            self.prv_resi = structure.resi[(residue._selection[0]-1)]
            # Identify which atoms to refine anisotropically:
            if xmap.resolution.high < 1.45:
                adp = "not (water or element H)"
            else:
                adp = f"chain {self.chain} and resid {self.resi}"

            # Generate the parameter file for phenix refinement:
            labels = options.label.split(",")
            with open(f"chain_{self.chain}_res_{self.resi}_adp.params", "w") as params:
                params.write("refinement {\n")
                params.write("  electron_density_maps {\n")
                params.write("    map_coefficients {\n")
                params.write(f"      mtz_label_amplitudes = {labels[0]}\n")
                params.write(f"      mtz_label_phases = {labels[1]}\n")
                params.write("      map_type = 2mFo-DFc\n")
                params.write("    }\n  }\n")
                params.write("  refine {\n")
                params.write("    strategy = *individual_sites *individual_adp\n")
                params.write("    adp {\n")
                params.write("      individual {\n")
                params.write(f"        anisotropic = {adp}\n")
                params.write("      }\n    }\n  }\n}\n")
            params.close()

            # Set the occupancy of structure to zero for omit map calculation
            out_root= f'out_{self.chain}_{self.resi}'
            structure.tofile(f'{out_root}.pdb')
            subprocess.run(["phenix.pdbtools",
                            f'modify.selection=\"chain {self.chain} and'
                            f'( resseq {self.resi} and not '
                            f'( name n or name ca or name c or name o or name cb )'
                            f' or ( resseq {self.prv_resi} and name n) )\"',
                            "modify.occupancies.set=0",
                            "stop_for_unknowns=False",
                            f"{out_root}.pdb",
                            f"output.file_name={out_root}_modified.pdb"])

            # Add hydrogens to the structure:
            out_mod_H = open(f"{out_root}_modified_H.pdb", "w")
            subprocess.run(["phenix.reduce", f"{out_root}_modified.pdb"],
                            stdout=out_mod_H)
            out_mod_H.close()

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
            xmap = xmap.extract(residue.coor, padding=5)

        # Check if residue is complete. If not, complete it:
        atoms = residue.name
        for atom in residue._rotamers['atoms']:
            if atom not in atoms:
                residue.complete_residue()
                break

        # If including hydrogens:
        if options.hydro:
            for atom in residue._rotamers['hydrogens']:
                if atom not in atoms:
                    print(f"[WARNING] Missing atom {atom} of residue \
                            {residue.resi[0]},{residue.resn[0]}")
                    continue


        super().__init__(residue, structure, xmap, options)
        self.residue = residue
        self.residue._init_clash_detection(self.options.clash_scaling_factor)
        # Get the segment that the residue belongs to
        chainid = self.residue.chain[0]
        for segment in self.structure.segments:
            if segment.chain[0] == chainid and self.residue in segment:
                self.segment = segment
                break
        # Override some residue specific options
        resn = self.residue.resn[0]
        if resn == "PRO":
            self.options.rotamer_neighborhood = 0
            self.options.sample_angle = False
        elif resn == 'GLY':
            self.options.sample_angle = False

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
            if np.linalg.norm(residue._coor[N_index]
                              - segment._coor[neighbor_C_index]) < 2:
                coor = N_neighbor._coor[neighbor_C_index]
                exclude.append((N_index, coor))
        if index < len(segment.residues) - 1:
            C_index = residue.select('name', 'C')[0]
            C_neighbor = segment.residues[index + 1]
            neighbor_N_index = C_neighbor.select('name', 'N')[0]
            if np.linalg.norm(residue._coor[C_index]
                              - segment._coor[neighbor_N_index]) < 2:
                coor = C_neighbor._coor[neighbor_N_index]
                exclude.append((C_index, coor))
        # Obtain atoms with which the residue can clash
        resi, icode = residue.id
        chainid = self.segment.chain[0]
        if icode:
            selection_str = f'not (resi {resi} and icode {icode} \
                              and chain {chainid})'
            receptor = self.structure.extract(selection_str)
        else:
            sel_str = f'not (resi {resi} and chain {chainid})'
            receptor = self.structure.extract(sel_str).copy()
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
            for coor in self._coor_set:
                self.residue.coor = coor
                if self.options.external_clash:
                    if not self._cd() and self.residue.clashes() == 0:
                        new_coor_set.append(coor)
                elif self.residue.clashes() == 0:
                    new_coor_set.append(coor)
            self._coor_set = new_coor_set
            self._convert()
            self._solve()
            self._update_conformers()
            self._convert()
            self._solve(threshold=self.options.threshold,
                        cardinality=self.options.cardinality)
            self._update_conformers()
        # Now that the conformers have been generated, the resulting
        # conformations should be examined via GoodnessOfFit:
        validator = Validator(self.xmap, self.xmap.resolution)
        if self.xmap.resolution.high < 3.0:
            cutoff = 0.7 + (self.xmap.resolution.high - 0.6)/3.0
        else:
            cutoff = 0.5 * self.xmap.resolution.high
        self.validation_metrics = validator.GoodnessOfFit(self.conformer,
                                                          self._coor_set,
                                                          self._occupancies,
                                                          cutoff)

    def _sample_backbone(self):
        # Check if residue has enough neighboring residues
        index = self.segment.find(self.residue.id)
        nn = self.options.neighbor_residues_required
        if index < nn or index + nn > len(self.segment):
            # self.options.sample_backbone = False
            return
        segment = self.segment[index - nn: index + nn + 1]

        atom_name = "CB"
        if self.residue.resn[0] == "GLY":
            atom_name = "O"
        atom = self.residue.extract('name', atom_name)
        try:
            unit_cell = self.xmap.unit_cell
            u_matrix = [[atom.u00[0], atom.u01[0], atom.u02[0]],
                        [atom.u01[0], atom.u11[0], atom.u12[0]],
                        [atom.u02[0], atom.u12[0], atom.u22[0]]]
            directions = move_direction_adp(u_matrix, unit_cell)
        except AttributeError:
            directions = np.identity(3)

        optimizer = NullSpaceOptimizer(segment)

        start_coor = atom.coor[0]
        torsion_solutions = []
        amplitudes = np.arange(0.1, self.options.sample_backbone_amplitude + 0.01,
                                 self.options.sample_backbone_step)
        sigma = self.options.sample_backbone_sigma
        for amplitude, direction in itertools.product(amplitudes, directions):
            endpoint = start_coor + (amplitude + sigma * np.random.random()) * direction
            optimize_result = optimizer.optimize(atom_name, endpoint)
            torsion_solutions.append(optimize_result['x'])

            endpoint = start_coor - (amplitude + sigma * np.random.random()) * direction
            optimize_result = optimizer.optimize(atom_name, endpoint)
            torsion_solutions.append(optimize_result['x'])
        starting_coor = segment.coor

        for solution in torsion_solutions:
            optimizer.rotator(solution)
            self._coor_set.append(self.segment[index].coor)
            segment.coor = starting_coor
        # print(f"Backbone sampling generated {len(self._coor_set)} conformers")

    def _sample_angle(self):
        """Sample residue along the N-CA-CB angle."""

        active_names = ('N', 'CA', 'C', 'O', 'CB', 'H', 'HA')
        selection = self.residue.select('name', active_names)
        self.residue.active = False
        self.residue._active[selection] = True
        self.residue.update_clash_mask()
        active = self.residue.active
        angles = np.arange(-self.options.sample_angle_range,
                           self.options.sample_angle_range+0.001,
                           self.options.sample_angle_step)
        new_coor_set = []

        for coor in self._coor_set:
            self.residue.coor = coor
            rotator = CBAngleRotator(self.residue)
            for angle in angles:
                rotator(angle)
                coor = self.residue.coor
                if self.options.remove_conformers_below_cutoff:
                    values = self.xmap.interpolate(coor[active])
                    mask = (self.residue.e[active] != "H")
                    if np.min(values[mask]) < self.options.density_cutoff:
                        continue
                if self.options.external_clash:
                    if self._cd() and self.residue.clashes():
                        continue
                elif self.residue.clashes():
                    continue
                new_coor_set.append(self.residue.coor)
        self._coor_set = new_coor_set
        #print(f"Bond angle sampling generated {len(self._coor_set)} conformers")
        if len(self._coor_set) > 1000:
            print("[WARNING] Large number of conformers have been generated. Run times may be slow."
                  "Please, consider changing sampling parameters and re-running qFit.")


    def _sample_sidechain(self):
        opt = self.options
        start_chi_index = 1
        sampling_window = np.arange(
            -opt.rotamer_neighborhood,
            opt.rotamer_neighborhood + opt.dofs_stepsize,
            opt.dofs_stepsize)
        rotamers = self.residue.rotamers
        rotamers.append([self.residue.get_chi(i) for i in range(1, self.residue.nchi + 1)])

        iteration = 0
        while True:
            chis_to_sample = opt.dofs_per_iteration
            if iteration == 0 and (opt.sample_backbone or opt.sample_angle):
                chis_to_sample = max(1, opt.dofs_per_iteration - 1)
            end_chi_index = min(start_chi_index + chis_to_sample,
                                self.residue.nchi + 1)
            for chi_index in range(start_chi_index, end_chi_index):

                # Set active and passive atoms, since we are iteratively
                # building up the sidechain. This updates the internal
                # clash mask.
                self.residue.active = True
                if chi_index < self.residue.nchi:
                    deactivate = self.residue._rotamers['chi-rotate'][chi_index + 1]
                    selection = self.residue.select('name', deactivate)
                    self.residue._active[selection] = False
                self.residue.update_clash_mask()
                active = self.residue.active

                logger.info(f"Sampling chi: {chi_index} ({self.residue.nchi})")
                new_coor_set = []
                n = 0
                for coor in self._coor_set:
                    n += 1
                    self.residue.coor = coor
                    chis = [self.residue.get_chi(i) for i in range(1, chi_index)]
                    sampled_rotamers = []
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
                        self.residue.set_chi(chi_index, rotamer[chi_index - 1])

                        # The starting chi angles are similar for many
                        # rotamers, make sure we are not sampling double
                        unique = True
                        residue_coor = self.residue.coor
                        for rotamer_coor in sampled_rotamers:
                            if np.allclose(rotamer_coor,
                                           residue_coor,
                                           atol=0.01):
                                unique = False
                                break
                        if not unique:
                            continue
                        sampled_rotamers.append(residue_coor)

                        # Sample around the neighborhood of the rotamer
                        chi_rotator = ChiRotator(self.residue, chi_index)
                        for angle in sampling_window:
                            chi_rotator(angle)
                            coor = self.residue.coor
                            if opt.remove_conformers_below_cutoff:
                                values = self.xmap.interpolate(coor[active])
                                mask = (self.residue.e[active] != "H")
                                if np.min(values[mask]) < opt.density_cutoff:
                                    continue
                            if self.options.external_clash:
                                if not self._cd() and self.residue.clashes() == 0:
                                    new_coor_set.append(self.residue.coor)
                            elif self.residue.clashes() == 0:
                                new_coor_set.append(self.residue.coor)
                self._coor_set = new_coor_set
            logger.info("Nconf: {:d}".format(len(self._coor_set)))
            if not self._coor_set:
                msg = "No conformers could be generated. Check for initial \
                       clashes and density support."
                raise RuntimeError(msg)
            if opt.debug:
                prefix = os.path.join(opt.directory,
                                      f'_conformer_{iteration}.pdb')
                self._write_intermediate_conformers(prefix=prefix)
            # print(f"Side chain sampling generated {len(self._coor_set)} conformers")

            # QP
            logger.debug("Converting densities.")
            self._convert()
            logger.info("Solving QP.")
            self._solve()
            logger.debug("Updating conformers")
            self._update_conformers()
            # self._write_intermediate_conformers(f"qp_{iteration}")
            # MIQP
            self._convert()
            logger.info("Solving MIQP.")
            self._solve(cardinality=opt.cardinality,
                        threshold=opt.threshold)

            self._update_conformers()
            # self._write_intermediate_conformers(f"miqp_{iteration}")
            logger.info("Nconf after MIQP: {:d}".format(len(self._coor_set)))

            # Check if we are done
            if chi_index == self.residue.nchi:
                break
            # Use the next chi angle as starting point, except when we are in
            # the first iteration and have selected backbone sampling and we
            # are sampling more than 1 dof per iteration
            increase_chi = not ((opt.sample_backbone or opt.sample_angle) and
                iteration == 0 and opt.dofs_per_iteration > 1)
            if increase_chi:
                start_chi_index += 1
            iteration += 1

    def tofile(self):
        conformers = self.get_conformers()
        for n, conformer in enumerate(conformers, start=1):
            fname = os.path.join(self.options.directory, f'conformer_{n}.pdb')
            conformer.tofile(fname)
        # Make a multiconformer residue
        nconformers = len(conformers)
        if nconformers < 1:
            msg = "No conformers could be generated. \
             Check for initial clashes."
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
        fname = os.path.join(self.options.directory,
                             f"multiconformer_residue.pdb")
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
        self.bulk_solvent_level = 0.3
        self.fragment_length = None
        self.rmsd_cutoff = 0.01


class QFitSegment(_BaseQFit):
    """Determines consistent protein segments based on occupancy and
       density fit"""

    def __init__(self, structure, xmap, options):
        self.segment = structure
        self.conformer = structure
        self.xmap = xmap
        self.options = options
        self.fragment_length = options.fragment_length
        self.BIC = np.inf
        self._coor_set = [self.conformer.coor]
        self._occupancies = [self.conformer.q]
        self.orderings = []
        self.charseq = []
        self._smax = None
        self._simple = True
        if self.xmap.resolution.high is not None:
            self._smax = 1 / (2 * self.xmap.resolution.high)
            self._simple = False
        elif options.resolution is not None:
            self._smax = 1 / (2 * options.resolution)
            self._simple = False

        self._smin = 0
        self._rmask = 1.5
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
        # Create an empty structure:
        hetatms = self.segment.extract('record', "HETATM")
        print(f'Average number of conformers before qfit_segment run: '
              f'{self.segment.average_conformers():.2f}')
        multiconformers = Structure.fromstructurelike(
                    self.segment.extract('altloc', "Z"))
        segment = []
        for i, rg in enumerate(self.segment.extract('record',
                                                    "ATOM").residue_groups):
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
                        print("f[WARNING] Conformer {altloc} of residue"
                              f"{rg.resi[0]} has more than one coordinate"
                              f" for CA/O atoms.")
                        mask = mask[:2]
                    try:
                        CA_single = np.linalg.norm(CA_pos-conformer.coor[mask][0])
                        CA_single = CA_single <= 0.05
                        O_single = np.linalg.norm(O_pos-conformer.coor[mask][1])
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
                    multiconformers = multiconformers.combine(multi.collapse_backbone(multi.resi[0]))

            else:
                segment.append(multiconformer)

        if len(segment):
            print(f"Running find_paths for segment of length {len(segment)}")
            for path in self.find_paths(segment):
                multiconformers = multiconformers.combine(path)

        print(f'Average number of conformers after qfit_segment run: {multiconformers.average_conformers():.2f}')
        multiconformers = multiconformers.reorder()
        multiconformers = multiconformers.remove_identical_conformers(self.options.rmsd_cutoff)
        print(f'Average number of conformers after removal of identical conformers: {multiconformers.average_conformers():.2f}')
        relab_options = RelabellerOptions()
        relabeller = Relabeller(multiconformers, relab_options)
        multiconformers = relabeller.run()
        multiconformers = multiconformers.combine(hetatms)
        multiconformers = multiconformers.reorder()
        return multiconformers

    def find_paths(self, segment_original):
        segment = segment_original[:]
        fl = self.fragment_length
        possible_conformers = list(map(chr, range(65, 90)))
        possible_conformers = possible_conformers[0:int(
            round(1./self.options.threshold))]
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
                    fragments.append(fragment)

                # We have the fragments, select consistent optimal set
                self._update_transformer(fragments[0])
                self._coor_set = [fragment.coor for fragment in fragments]
                # QP
                self._convert()
                self._solve()
                # Update conformers
                fragments = np.array(fragments)
                mask = self._occupancies >= 0.002
                fragments = fragments[mask]
                self._coor_set = [fragment.coor for fragment in fragments]
                self._occupancies = self._occupancies[mask]
                # self.print_paths(fragments)
                # MIQP
                self._convert()
                self._solve(cardinality=self.options.cardinality,
                            threshold=self.options.threshold,
                            loop_range=[0.3, 0.25, 0.2, 0.16, 0.14, 0.12])
                # Update conformers
                mask = self._occupancies >= 0.002
                for fragment, occ in zip(fragments[mask],
                                         self._occupancies[mask]):
                    fragment.q = occ
                segment.append(fragments[mask])

        for path, altloc in zip(segment[0],possible_conformers ):
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
            print(f"Path {k+1}:\t{ path }\t{fragment.q[-1]}")


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
