import gc
from .qfit import QFitOptions
import multiprocessing as mp
from tqdm import tqdm
import os.path
import os
import sys
import itertools
import copy
import argparse
from .custom_argparsers import ToggleActionFlag, CustomHelpFormatter
import logging
import traceback
import numpy as np
from collections import namedtuple
import time 
import math
import copy
import itertools
from itertools import product
import pickle
from string import ascii_uppercase
from scipy.optimize import least_squares

from .solvers import SolverError, get_qp_solver_class, get_miqp_solver_class
from .water_data_loader import load_water_rotamer_dict
from .qfit import QFitRotamericResidue
from .clash import ClashDetector
from .logtools import setup_logging, log_run_info, poolworker_setup_logging, QueueListener
from . import MapScaler, Structure, XMap
from .structure.rotamers import ROTAMERS
from .samplers import ChiRotator, CBAngleRotator, BondRotator
from .structure.residue import _RotamerResidue
from .transformer import Transformer
from .solvers import QPSolver, MIQPSolver #, SolverError

dict_dist = {
		'Cm' : 3.0,
		'Nm' : 2.4,
		'Om' : 2.4,
		'S' : 2.4,
		'C' : 3.0,
		'N' : 2.4,
		'O' : 2.4}

# Create a namedtuple 'class' (struct) which carries info about an MIQP solution
MIQPSolutionStats = namedtuple(
    "MIQPSolutionStats", ["threshold", "BIC", "rss", "objective", "weights"]
)

# Create a namedtuple 'class' (struct) which carries info about an MIQP solution
MIQPSolutionStats = namedtuple(
    "MIQPSolutionStats", ["threshold", "BIC", "rss", "objective", "weights"]
)


logger = logging.getLogger(__name__)
os.environ["OMP_NUM_THREADS"] = "1"

class QFitWaterOptions:
	def __init__(self):
		super().__init__()
		self.nproc = 1
		self.verbose = True
		self.omit = False
		self.rotamer = None
		self.water = None
		self.pdb = None
		self.em = None
		self.debug = False
		self.directory = "."
		self.structure = None
		self.label = None
		self.map = None
		self.residue = None

		# Density preparation options
		self.density_cutoff = 0.3
		self.subtract = True
		self.padding = 8.0
		self.waters_clash = True
		self.clash_scaling_factor = 0.75
		self.external_clash = False
		self.dofs_per_iteration = 2
		self.dihedral_stepsize = 10
		self.hydro = False
		self.rmsd_cutoff = 0.01

		# MIQP options
		self.qp_solver = None
                self.miqp_solver = None
		self.cardinality = 5
		self.threshold = 0.20
		self.bic_threshold = True
		self.seg_bic_threshold = True

		# Density creation options
		self.map_type = None
		self.resolution = None
		self.resolution_min = None
		self.scattering = "xray"
		self.omit = False
		self.scale = True
		self.scale_rmask = 1.0
		self.randomize_b = False
		self.bulk_solvent_level = 0.3

		# Rotamer sampling
		self.sample_rotamers = True
		self.rotamer_neighborhood = 20  
		self.remove_conformers_below_cutoff = False

	def apply_command_args(self, args):
		for key, value in vars(args).items():
			if hasattr(self, key):
				setattr(self, key, value)
		return self


class QFitWater:
		def __init__(self, conformer, structure, xmap, options): 
			self.xmap = xmap
			self.structure = structure
			self.residue = conformer
			self.base_residue = conformer
			self.options = options
			self._xmap_model = xmap.zeros_like(self.xmap)
			self._xmap_model.set_space_group("P1")
			self._voxel_volume = self.xmap.unit_cell.calc_volume()
			self._voxel_volume /= self.xmap.array.size
			self._smax = None
			self._smin = None
			self._simple = True
			self._bs = [self.residue.b]
			self._coor_set = [self.residue.coor]
			self._rmask = 0.8 
			self._occupancies = []
			self.water_coor_set = []
			self.water_bs = []

		def run(self):
			water = self._run_water_sampling()

		def _sample_sidechain(self):
			opt = self.options
			self._update_transformer(self.residue)
			start_chi_index = 1
			if self.residue.resn[0] != "PRO":
				sampling_window = np.arange(
					-opt.rotamer_neighborhood,
					opt.rotamer_neighborhood + opt.dihedral_stepsize,
					opt.dihedral_stepsize,
				)
			else:
				sampling_window = [0]

			rotamers = self.residue.rotamers
			rotamers.append(
				[self.residue.get_chi(i) for i in range(1, self.residue.nchi + 1)]
			)
			iteration = 0
			while True:
				chis_to_sample = 1
				if iteration == 0:
					chis_to_sample = max(1, opt.dofs_per_iteration - 1)
				end_chi_index = min(start_chi_index + chis_to_sample, self.residue.nchi + 1)
				iter_coor_set = []
				iter_b_set = (
					[]
				)  # track b-factors so that they can be reset along with the coordinates if too many conformers are generated
				for chi_index in range(start_chi_index, end_chi_index):
					# Set active and passive atoms, since we are iteratively
					# building up the sidechain. This updates the internal
					# clash mask.
					self.residue.active = True
					if chi_index < self.residue.nchi:
						current = self.residue._rotamers["chi-rotate"][chi_index]
						deactivate = self.residue._rotamers["chi-rotate"][chi_index + 1]
						selection = self.residue.select("name", deactivate)
						self.residue._active[selection] = False
						bs_atoms = list(set(current) - set(deactivate))
					else:
						bs_atoms = self.residue._rotamers["chi-rotate"][chi_index]

					self.residue.update_clash_mask()
					active = self.residue.active

					print(f"Sampling chi: {chi_index} ({self.residue.nchi})")
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
								if (
									360 - opt.rotamer_neighborhood
									> diff_chi
									> opt.rotamer_neighborhood
								):
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
								
								values = self.xmap.interpolate(coor[active])
								mask = self.residue.e[active] != "H"

								# data = {}
								# for attr in self.residue.data:
								# 		array1 = getattr(self.residue, attr)
								# 		data[attr] = array1[self.residue.active]
								# Structure(data).tofile(f'{TTT}.pdb')
								# TTT+=1
								# Change the condition to check if a certain number of values are below the density cutoff
								if np.sum(values[mask] < 1) > 2:
									keep_coor_set = False
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
										delta = np.array(new_coor_set) - np.array(
											self.residue.coor
										)
										if (
											np.sqrt(
												min(
													np.square((delta))
													.sum(axis=2)
													.sum(axis=1)
												)
											)
											>= 0.01
										):
											new_coor_set.append(self.residue.coor)
											new_bs.append(b)
										else:
											ex += 1
									else:
										#print('adding')
										new_coor_set.append(self.residue.coor)
										new_bs.append(b)
								else:
									ex += 1

					iter_coor_set.append(new_coor_set)
					iter_b_set.append(new_bs)
					self._coor_set = new_coor_set
					self._bs = new_bs

				if len(self._coor_set) > 15000:
					logger.warning(
						f"[{self.identifier}] Too many conformers generated ({len(self._coor_set)}). "
						f"Reverting to a previous iteration of degrees of freedom: item 0. "
						f"n_coords: {[len(cs) for (cs) in iter_coor_set]}"
					)
					self._coor_set = iter_coor_set[0]
					self._bs = iter_b_set[0]

				if not self._coor_set:
					msg = (
						"No conformers could be generated. Check for initial "
						"clashes and density support."
					)
					raise RuntimeError(msg)

				logger.debug(
					f"Side chain sampling generated {len(self._coor_set)} conformers"
				)

				# QP score conformer occupancy
				#self._convert("test")
				#self._solve_qp()
				#self._update_conformers()

				# Check if we are done
				if chi_index == self.residue.nchi:
					break

				# Use the next chi angle as starting point, except when we are in
				# the first iteration and have selected backbone sampling and we
				# are sampling more than 1 dof per iteration
				increase_chi = not (
					iteration == 0
				)
				if increase_chi:
					start_chi_index += 1
				iteration += 1


		def _run_water_sampling(self):
				"""Run qfit water on each residue."""
				#r_pro._init_clash_detection() #look at clashes around the protein
				self.base_residue = copy.deepcopy(self.residue)
				new_coor_set = []
				new_bs = []
				water_coor = []
				self.n = 100 #placeholder -> get last residue of chain
				# Read in dictionary from create_water_rotamer_dictionary
				water_rotamer_dict = load_water_rotamer_dict()

				#sampling residue
				self._sample_sidechain()
				print(len(self._coor_set))
				#self._write_intermediate_conformers(prefix="postsidechain")

				max_attached_waters = 0
				matching_keys = [key for key in water_rotamer_dict.keys() if key[:3] == self.residue.resn[0]]			
				for coor in self._coor_set:
					new_coor_set.append(coor) #append protein only
					new_bs.append(10)
					for key in matching_keys:
						A = water_rotamer_dict[key]['atoms']
						# Extract x, y, z coordinates for each name in A and store it in a 3xN matrix
						A_coordinates = np.array([[atom['x'], atom['y'], atom['z']] for atom in A])
						B = coor
						# Feed coordinates into rigid_transform_3D function
						R, t = self.rigid_transform_3D(A_coordinates.T, B.T)
						new_positions = []
						# Determine the relative coordinate position of each attached_water
						
						for attached_water in water_rotamer_dict[key]['attached_waters']:
							wat_coord = np.array([attached_water['x'], attached_water['y'], attached_water['z']])
							# Apply rotation and translation to the attached_water coordinates
							new_position = (np.dot(R, np.array(wat_coord).T)+t.T)[0]
							new_positions.append(new_position)
					#print(new_positions)
					for r in range(1, len(new_positions)+1):
						for water_position in itertools.combinations(new_positions, r):
							values = self.xmap.interpolate(np.vstack(water_position))
							new_coor = np.vstack((coor, water_position))
							#values = self.xmap.interpolate(new_coor)
							# if np.sum(values < 1) > 0:
							# 	# print('add')
							new_coor_set.append(new_coor)
							new_bs.append(10)
								# else:
								# 	print('dont add')
								# 	print(values)
								# 	print(np.sum(values < 1))
					#print(new_coor_set)

				self._coor_set = new_coor_set
				#print(new_coor_set)
				self._bs = new_bs
				#print('RESIDUE DATA:')
				#print(self.residue.data)
				self._write_intermediate_conformers(prefix="preqp")	
				#self.residue.remove_water()
				self._update_transformer(self.residue)
				self._transformer.reset(full=True)
				self._convert("preqp")
				print('done converting')
				# #self._write_intermediate_conformers(prefix="convertqp")	
				# #print(len(self._coor_set))
				# #print(len(self._bs))
				# self.write_maps()
				self._solve_qp()
				print('postqp')
				self._update_conformers()
				self._write_intermediate_conformers(prefix="postqp")	
				self._convert("postqp")
				self._solve_miqp(threshold=self.options.threshold, cardinality=self.options.cardinality)
				self._update_conformers()
				print(self._occupancies)
				#self._write_intermediate_conformers(prefix="miqp_solution")		
		
		def _write_intermediate_conformers(self, prefix="conformer"):
			for n, coor in enumerate(self._coor_set):
				self.residue = copy.deepcopy(self.base_residue)
				diff = len(coor) - len(self.residue.coor)
				if diff > 0: 
					for i in range(1, diff + 1):
						last_item = coor[-i]
						self.residue.add_water_atom3({'chain': 'A','resi': 3, 'coor': last_item, 'q': 1.0,'b': 10.0, 'altloc':'', 'active':True})	
				fname = os.path.join(f"{prefix}_{n}.pdb")
				data = {}
				for attr in self.residue.data:
					data[attr] = self.residue.data[attr]
				Structure(data).tofile(fname)

		def rigid_transform_3D(self,A, B):
			'''
			* Not my function *
			from : https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
			function for calculating the optimal rotation and transformation matrix for a set of 4 points
			onto another set of 4 points
			Input: expects 3xN matrix of points
			Returns R,t
					R = 3x3 rotation matrix
					t = 3x1 column vector
			'''
			assert A.shape == B.shape

			num_rows, num_cols = A.shape

			if num_rows != 3:
					raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

			num_rows, num_cols = B.shape
			if num_rows != 3:
					raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

			# find mean column wise
			centroid_A = np.mean(A, axis=1)
			centroid_B = np.mean(B, axis=1)

			# ensure centroids are 3x1
			centroid_A = centroid_A.reshape(-1, 1)
			centroid_B = centroid_B.reshape(-1, 1)

			# subtract mean
			Am = A - centroid_A
			Bm = B - centroid_B

			H = Am @ np.transpose(Bm)

			# find rotation
			U, S, Vt = np.linalg.svd(H)
			R = Vt.T @ U.T

			t = -R @ centroid_A + centroid_B
			
			return R, t

		def _convert(self, state): #figure out why 28 atoms on conformer.coor and not 17?
			"""Convert structures to densities and extract relevant values for (MI)QP."""
			self._transformer.reset(full=True) #converting self.xmap.array to zero
			nmodels = len(self._coor_set) 
			#creating a mask for the density map
			#take all conformers and applies a mask to the density 
			n = 1
			for coor in self._coor_set: #for every combination of water molecules
				self.residue = copy.deepcopy(self.base_residue)
				print(len(self.residue.coor))
				diff = len(coor) - len(self.residue.coor)
				if diff > 0:
					for i in range(1, diff + 1):
						last_item = coor[-i]
						self.residue.add_water_atom3({'chain': 'A','resi': 3, 'coor': last_item, 'q': 1.0,'b': 10.0, 'altloc':'', 'active':True})	
				self._update_transformer(self.residue)
				#The mask is a boolean array that indicates which voxels in the density map are within a certain radius (self._rmask) of the current conformer.
				self._transformer.mask(0.3) #self._rmask
			mask = (self._transformer.xmap.array > 0)
			nvalues = mask.sum()
			self._target = self.xmap.array[mask]
			self._transformer.reset(full=True)
			#the mask is stored as a boolean array (mask = self._transformer.xmap.array > 0), where True values represent voxels that are within the radius of at least one conformer.
			# self.xmap.tofile('xmap.ccp4')
			# self._target = self.xmap.array[mask]
			# fname_target = os.path.join(f"target.ccp4")
			# self._target.tofile(fname_target)
			
			self._models = np.zeros((nmodels, nvalues), float)
			for n, coor in enumerate(self._coor_set): #for every combination of water molecules
				self.residue = copy.deepcopy(self.base_residue)
				# Remove all waters first
				# Determine the difference in length between coor and the residue's coor
				diff = len(coor) - len(self.residue.coor)
				if diff > 0:
					for i in range(1, diff + 1):
						last_item = coor[-i]
						self.residue.add_water_atom3({'chain': 'A','resi': 3, 'coor': last_item, 'q': 1.0,'b': 10.0, 'altloc':'', 'active':True})	
				self._update_transformer(self.residue)                       
				self._transformer.density()
				model = self._models[n]
				model[:] = self._transformer.xmap.array[mask]
				model_size = model.size
				np.maximum(model, 0.0, out=model) #self.options.bulk_solvent_level
				self._transformer.reset(full=True)

		def _solve_qp(self):
			# Create and run solver
			print("Solving QP")
        		qp_solver_class = get_qp_solver_class(self.options.qp_solver)
			solver = qp_solver_class(self._target, self._models)
			solver.solve_qp()
			print(solver.weights[solver.weights > 0.05])
			# Update occupancies from solver weights
			self._occupancies = solver.weights
			# Return solver's objective value (|ρ_obs - Σ(ω ρ_calc)|)
			return solver.objective_value
		
		def _solve_miqp(
			self,
			cardinality,
			threshold,
			loop_range=[0.5, 0.33, 0.25, 0.2],
			do_BIC_selection=None,
			segment=None,
		):
			# Set the default (from options) if it hasn't been passed as an argument
			if do_BIC_selection is None:
				do_BIC_selection = self.options.bic_threshold

			# Create solver
			print("Solving MIQP")
			miqp_solver_class = get_miqp_solver_class(self.options.miqp_solver)
			solver = miqp_solver_class(self._target, self._models)

			# Threshold selection by BIC:
			if do_BIC_selection:
				# Iteratively test decreasing values of the threshold parameter tdmin (threshold)
				# to determine if the better fit (RSS) justifies the use of a more complex model (k)
				miqp_solutions = []
				for threshold in loop_range:
					solver.solve_miqp(cardinality=None, threshold=threshold)
					rss = solver.objective_value * self._voxel_volume
					n = len(self._target)
					n = len(self._models)

					natoms = self._coor_set[0].shape[0]
					nconfs = np.sum(solver.weights >= 0.002)
					model_params_per_atom = 1 
					k = (
						model_params_per_atom * natoms * nconfs #2
					)  # 0.95 hyperparameter in put in here since we are almost always over penalizing
					if segment is not None:
						k = nconfs  # for segment, we only care about the number of conformations come out of MIQP. Considering atoms penalizes this too much
					BIC = n * np.log(rss / n) + k * np.log(n)
					solution = MIQPSolutionStats(
						threshold=threshold,
						BIC=BIC,
						rss=rss,
						objective=solver.objective_value.copy(),
						weights=solver.weights.copy(),
					)
					miqp_solutions.append(solution)

				# Update occupancies from solver weights
				print(miqp_solutions)
				miqp_solution_lowest_bic = min(miqp_solutions, key=lambda sol: sol.BIC)
				print(miqp_solution_lowest_bic)
				self._occupancies = miqp_solution_lowest_bic.weights
				# Return solver's objective value (|ρ_obs - Σ(ω ρ_calc)|)
				return miqp_solution_lowest_bic.objective

			else:
				# Run solver with specified parameters
				solver.solve(cardinality=cardinality, threshold=threshold)

				# Update occupancies from solver weights
				self._occupancies = solver.weights

				# Return solver's objective value (|ρ_obs - Σ(ω ρ_calc)|)
				return solver.obj_value

		def _update_conformers(self, cutoff=0.02):
				logger.debug("Updating conformers based on occupancy")
				# Check that all arrays match dimensions.
				assert len(self._occupancies) == len(self._coor_set) == len(self._bs)
				filterarray = (self._occupancies >= cutoff)
				self._occupancies = self._occupancies[filterarray]
				self._coor_set = list(itertools.compress(self._coor_set, filterarray))
				self._bs = list(itertools.compress(self._bs, filterarray))
				print(f"Remaining valid conformations: {len(self._coor_set)}")

		def _update_transformer(self, structure):
				self.residue = structure
				self._transformer = Transformer(
												structure, self._xmap_model,
												smax=self._smax, smin=self._smin,
												simple=self._simple,
												em=self.options.em
				)
				self._transformer.initialize()

		def write_maps(self):
			"""Write out model and difference map."""
			if np.allclose(self.xmap.origin, 0):
				ext = "ccp4"
			else:
				ext = "mrc"
			print(type(self._transformer.xmap))
			print(type(self.xmap))
			print(type(self._target))
			#self._transformer.xmap.array = self.xmap.array
			# for i, (q, coor, b) in enumerate(zip(self._occupancies, self._coor_set, self._bs)):
			# 	self.residue.q = q
			# 	self.residue.coor = coor
			# 	self.residue.b = b
			# 	self._transformer.density()
			# 	fname = os.path.join(f"model_i.{ext}")
			# 	self._transformer.xmap.tofile(fname)
			# 	self._transformer.reset(full=True)
			# Create a new CCP4 map
			# Convert numpy array to gemmi.FloatGrid
			# ccp4_map = gemmi.Ccp4Map()
			# # Create a new FloatGrid
			# grid = gemmi.FloatGrid(self._target)

			# # Fill the grid with data from the numpy array
			# grid.spacegroup = gemmi.find_spacegroup_by_name('P 1')
			# for index, value in np.ndenumerate(self._target):
			# 	grid.set_value(index, value)

			# # Assign the grid to the ccp4_map
			# ccp4_map.grid = grid
			# # Set the map properties
			# ccp4_map.update_ccp4_header(2, True)

			# # Write the map to a file
			# ccp4_map.write_ccp4_map('target.ccp4')
			

		def _place_waters(self, wat_loc, altloc, resn):
			"""create new residue structure with residue atoms & new water atoms
									take OG residue, output new residue
			"""
			water = self.water
			sidechain = self.pro_alt.select('name', (['N', 'CA', 'C', 'O']), '!=')
			for n, coor in enumerate(wat_loc):
				water.resi = self.n #giving each water molecule its own resi
				water.chain = 'S'
				water.coor = coor
				if np.unique(self.pro_alt.extract('resn', resn, '==')).all() == 1.0:
						water.q = 1.0
						water.altloc = ''
				else:
						water.q = np.unique(self.pro_alt.extract('resn', resn, '==').q)[0]
						water.altloc = altloc
				water.b = np.mean(self.pro_alt.b[sidechain])*1.5 #make b-factor higher
				if n == 0:
								residue = self.pro_alt.extract('resn', resn, '==').combine(water)
				else:
						residue = residue.combine(water)
			return residue.coor, residue.b 

		def get_water_conformers(self):
				conformers = []
				for i, (q, coor, b) in enumerate(zip(self._occupancies, self._coor_set, self._bs)):
					conformer = copy.deepcopy(self.base_residue)
					diff = len(coor) - len(conformer.coor)
					if diff > 0:
						for j in range(1, diff + 1):
							coord = coor[-j]
							conformer.add_water_atom3({'chain': 'A','resi': 3, 'coor': coord, 'q': q,'b': b, 'altloc': chr(65 + i), 'active':True})	
					conformer.q = q
					conformer.coor = coor
					conformer.b = b
					conformers.append(conformer)
					print(conformers)
				return conformers


		def _run_water_clash(self, water):
				self._cd = ClashDetector(water, self.structure, scaling_factor=self.options.clash_scaling_factor)
				if not self._cd():
								return True
				else:
								return False

		def calc_chi1(self, v1, v2, v3, v4):
						b1 = v1.flatten() - v2.flatten()
						b2 = v2.flatten() - v3.flatten()
						b3 = v3.flatten() - v4.flatten()

						n1 = np.cross(b1, b2)/np.linalg.norm(np.cross(b1, b2))
						n2 = np.cross(b2, b3)/np.linalg.norm(np.cross(b2, b3))
						b2 = b2/np.linalg.norm(b2)

						x = np.dot(n1, n2)
						y = np.dot(np.cross(n1, b2), n2)

						radians = math.atan2(y, x)
						return math.degrees(radians)

		def new_dihedral(self,p):
				"""
				Function for calculated the dihedral of a given set of 4 points.
				(not my function)
				Parameters 
				----------
				p : nd.array, shape=(4, 3)
						4 points you want to calculate a dihedral from
				Returns
				-------
				dih_ang : float
						calculated dihedral angle
				"""
				p0 = p[0]
				p1 = p[1]
				p2 = p[2]
				p3 = p[3]

				b0 = -1.0*(p1 - p0)
				b1 = p2 - p1
				b2 = p3 - p2
				b1 /= np.linalg.norm(b1)

				v = b0 - np.dot(b0, b1)*b1
				w = b2 - np.dot(b2, b1)*b1
				x = np.dot(v, w)
				y = np.dot(np.cross(b1, v), w)
				dih_ang = np.degrees(np.arctan2(y, x))
				return dih_ang
		
		def choose_rot(self, dihedral):
				if dihedral < 0:
								rot = 360 + dihedral
				else:
								rot = dihedral
				if 0 <= rot < 120:
										rotamer = 'g+'
				elif 120 <= rot < 240:
										rotamer = 't'
				elif 240 <= rot < 360:
										rotamer = 'g-'
				return rotamer

		def least_squares(self, P1, P2, P3, P4, P5, dist_1, dist_2, dist_3, dist_4, dist_5):
				def equations5(guess):
						x, y, z = guess
						return(
						(x - x1)**2 + (y - y1)**2 + (z - z1)**2 - (dist_1)**2,
						(x - x2)**2 + (y - y2)**2 + (z - z2)**2 - (dist_2)**2,
						(x - x3)**2 + (y - y3)**2 + (z - z3)**2 - (dist_3)**2,
						(x - x4)**2 + (y - y4)**2 + (z - z4)**2 - (dist_4)**2,
						(x - x5)**2 + (y - y5)**2 + (z - z5)**2 - (dist_5)**2
						)

				x1, y1, z1 = [P1.flatten().tolist()[i] for i in (0, 1, 2)]
				x2, y2, z2 = [P2.flatten().tolist()[i] for i in (0, 1, 2)]
				x3, y3, z3 = [P3.flatten().tolist()[i] for i in (0, 1, 2)]
				x4, y4, z4 = [P4.flatten().tolist()[i] for i in (0, 1, 2)]
				x5, y5, z5 = [P5.flatten().tolist()[i] for i in (0, 1, 2)]
				x_g = np.array(np.mean(P1.flatten()))
				y_g = np.array(np.mean(P2.flatten()))
				z_g = np.array(np.mean(P3.flatten()))
				initial_guess = (x_g, y_g, z_g)
				results_5 = least_squares(equations5, initial_guess)
				dist_err = sum([abs(f) for f in results_5.fun])/len(results_5.fun) #avg sum of absolute val of residuals
				x, y, z = results_5.x
				return results_5.x.reshape(3, 1).T

		def least_squares_gly(self, P1, P2, P3, P4, dist_1, dist_2, dist_3, dist_4):
				def equations5(guess):
						x, y, z = guess
						return(
						(x - x1)**2 + (y - y1)**2 + (z - z1)**2 - (dist_1)**2,
						(x - x2)**2 + (y - y2)**2 + (z - z2)**2 - (dist_2)**2,
						(x - x3)**2 + (y - y3)**2 + (z - z3)**2 - (dist_3)**2,
						(x - x4)**2 + (y - y4)**2 + (z - z4)**2 - (dist_4)**2,
						)

				x1, y1, z1 = [P1.flatten().tolist()[i] for i in (0, 1, 2)]
				x2, y2, z2 = [P2.flatten().tolist()[i] for i in (0, 1, 2)]
				x3, y3, z3 = [P3.flatten().tolist()[i] for i in (0, 1, 2)]
				x4, y4, z4 = [P4.flatten().tolist()[i] for i in (0, 1, 2)]
				x_g = np.array(np.mean(P1.flatten()))
				y_g = np.array(np.mean(P2.flatten()))
				z_g = np.array(np.mean(P3.flatten()))
				initial_guess = (x_g, y_g, z_g)
				results_5 = least_squares(equations5, initial_guess)
				dist_err = sum([abs(f) for f in results_5.fun])/len(results_5.fun) #avg sum of absolute val of residuals
				x, y, z = results_5.x
				return results_5.x.reshape(3, 1).T
