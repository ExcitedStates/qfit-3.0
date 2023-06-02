import gc
from .qfit import QFitOptions
import multiprocessing as mp
from tqdm import tqdm
import os.path
import os
import sys
import itertools
import argparse
from .custom_argparsers import ToggleActionFlag, CustomHelpFormatter
import logging
import traceback
import numpy as np
import time 
import math
import copy
import itertools
import pickle
from string import ascii_uppercase
from scipy.optimize import least_squares

from .clash import ClashDetector
from .logtools import setup_logging, log_run_info, poolworker_setup_logging, QueueListener
from . import MapScaler, Structure, XMap
from .structure.rotamers import ROTAMERS
from .transformer import Transformer
from .solvers import QPSolver, MIQPSolver

dict_dist = {
		'Cm' : 3.0,
		'Nm' : 2.4,
		'Om' : 2.4,
		'S' : 2.4,
		'C' : 3.0,
		'N' : 2.4,
		'O' : 2.4}

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
        self.cplex = True
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

    def apply_command_args(self, args):
        for key, value in vars(args).items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class QFitWater_Residue:
	def __init__(self, residue, structure, xmap, water, num_resi, options):
		self.xmap = xmap
		self.structure = structure
		self.residue = residue
		self.options = options
		self.water = water
		self.n = num_resi
		self._xmap_model = xmap.zeros_like(self.xmap)
		self._xmap_model.set_space_group("P1")
		self._voxel_volume = self.xmap.unit_cell.calc_volume()
		self._voxel_volume /= self.xmap.array.size
		self._smax = None
		self._smin = None
		self._simple = True
		self._rmask = 0.8 
		self._occupancies =[]


	def run(self):
		water = self._run_water_sampling() 


	def _run_water_sampling(self):
		"""Run qfit water on each residue."""
		r_pro = self.residue.extract('resn', 'HOH', '!=') #extract just the residues
		r_pro._init_clash_detection() #look at clashes around the protein
		print(self.residue.name)
		self._update_transformer(self.residue) #this should now include water molecules
		self.water_holder_coor = np.empty((1, 3))
		self.water_holder_coor[:] = np.nan
		new_coor_set = []
		new_bs = []
		occ = []
		all_water_coor = []
		water_coor = []
		self.n = 100 #placeholder -> get last residue of chain
		altlocs = np.unique(r_pro.altloc)
		# Read in dictionary from create_water_rotamer_dictionary
		with open("/Users/stephaniewanko/Downloads/qfit-3.0/src/qfit/water_rotamer_library.json", "rb") as f:
			water_rotamer_dict = pickle.load(f)

		# Infilling code
		for resn, wat_center_coords in water_rotamer_dict.items():
			if self.residue.resn[0] == resn:
				# Determine the number of unique coordinates in water_cluster_coords
				water_cluster_coords = np.array(wat_center_coords)
				unique_coords = np.unique(water_cluster_coords, axis=0)
				num_unique_coords = len(unique_coords)
		return num_unique_coords

				# Continue with the rest of the code
				
				# Extract relevant information from the dictionary
		
		
									
		
		


		#write out multiconformerse

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

	
	def rigid_transform_3D(A, B):
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

	def choose_rotamer(self, resn, r_pro, a):
		chis = DICT4A[resn]
		for i in range(len(chis)):
			atoms = list(chis.keys())[i]
			dihedral = self.calc_chi1(r_pro.extract(f'name {atoms[0]}').coor, r_pro.extract(f'name {atoms[1]}').coor, r_pro.extract(f'name {atoms[2]}').coor, r_pro.extract(f'name {atoms[3]}').coor)					
			rotamer = self.choose_rot(dihedral, r_pro)
			return rotamer 


	def _convert(self): #figure out why 28 atoms on conformer.coor and not 17?
		"""Convert structures to densities and extract relevant values for (MI)QP."""
		#print("Converting conformers to density")
		self._transformer.reset(full=True) #converting self.xmap.array to zero
		for n, coor in enumerate(self._coor_set):
			self.conformer.coor = coor
			self._transformer.mask(1.5) #self._rmask
		mask = (self._transformer.xmap.array > 0)
		self._transformer.reset(full=True)

		nvalues = mask.sum()
		self._target = self.xmap.array[mask]
		#print("Density")
		nmodels = len(self._coor_set)
		self._models = np.zeros((nmodels, nvalues), float)
		for n, coor in enumerate(self._coor_set):
						self.conformer.coor = coor
						self.conformer.b = self._bs[n]
						self._update_transformer(self.conformer)
						self._transformer.density()
						model = self._models[n]
						model[:] = self._transformer.xmap.array[mask]
						np.maximum(model, 0.0, out=model) #self.options.bulk_solvent_level
						
						if np.sum(model) > 0.0:
								print(np.sum(model))
								#print(n)
								#print(coor)
						self._transformer.reset(full=True)


	def _solve(self, cardinality=None, threshold=None,
		loop_range=[0.5, 0.4, 0.33, 0.3, 0.25, 0.2, 0.1]):
		do_qp = cardinality is threshold is None
		if do_qp:
			logger.info("Solving QP")
			solver = QPSolver(self._target, self._models, use_cplex=self.options.cplex)
			solver()
		else:
			logger.info("Solving MIQP")
			solver = MIQPSolver(self._target, self._models, use_cplex=self.options.cplex)
			solver(cardinality=cardinality, threshold=threshold)
										

		# Update occupancies from solver weights
		self._occupancies = solver.weights

		# logger.info(f"Residual under footprint: {residual:.4f}")
		# residual = 0
		return solver.obj_value

	def _update_conformers(self, cutoff=0.002):
			logger.debug("Updating conformers based on occupancy")

			# Check that all arrays match dimensions.
			assert len(self._occupancies) == len(self._coor_set) == len(self._bs)

			filterarray = (self._occupancies >= cutoff)
			self._occupancies = self._occupancies[filterarray]
			self._coor_set = list(itertools.compress(self._coor_set, filterarray))
			self._bs = list(itertools.compress(self._bs, filterarray))
			
			for coor in self._coor_set:
					print('update')
					print(coor)
			print(f"Remaining valid conformations: {len(self._coor_set)}")

	def _update_transformer(self, structure):
		self.conformer = structure
		self._transformer = Transformer(
						structure, self._xmap_model,
						smax=self._smax, smin=self._smin,
						simple=self._simple,
						scattering=self.options.scattering,
		)
		self._transformer.initialize()

	def get_conformers(self):
		conformers = []
		for q, coor, b in zip(self._occupancies, self._coor_set, self._bs):
						conformer = self.base_residue.copy()
						conformer.q = q
						conformer.coor = coor
						conformer.b = b
						conformers.append(conformer)
		return conformers

	def _write_water_locations(self, water_coor, prefix):
		conformers = []
		n = 1
		for coor in water_coor:
				conformer = self.water
				conformer.coor = coor
				conformer.q = 1.0
				conformer.b = 10
				conformer.resi = n
				if n == 1:
						final_conf = conformer
				else:
						final_conf = final_conf.combine(conformer)
				n += 1
		final_conf.tofile(f"{prefix}.pdb")
	
	def _write_intermediate_conformers(self, prefix="_conformer"):
			conformers = []
			conformer = self.base_residue.copy()
			if len(self._occupancies) == 0:
					for i, coor in enumerate(self._coor_set):
							conformer = self.base_residue.copy()
							conformer.q = 1.0
							conformer.coor = coor
							conformer.b = self._bs[i]
							for c in conformer.extract('resn', 'HOH', '==').atomid:
								conformer.extract('atomid', c, '==').resi = self.n
								self.n += 1
							#conformer.resi = self.n 
							conformers.append(conformer)
							#self.n += 1
			else:
				conformers = []
				for i, coor in enumerate(self._coor_set):   
							conformer = self.base_residue.copy()
							conformer.q = self._occupancies[i]
							conformer.coor = coor
							conformer.b = self._bs[i]
							conformer.resi = self.n 
							conformers.append(conformer)
							self.n += 1
			for i in range(len(conformers)):
						conf = Structure.fromstructurelike(conformers[i])
						for a in conf.atomid:
							if np.isnan(np.sum(conf.extract('atomid', a, '==').coor)): 
								continue
							else:
									try: 
										tmp = tmp.combine(conf.extract('atomid', a, '=='))
									except Exception:
										tmp = conf.extract('atomid', a, '==')
						multiconf = tmp
						del tmp
						if i < 26: 
									multiconf.altloc = ascii_uppercase[i]
						elif i < 52:
									multiconf.resi = multiconf.resi + 1
									multiconf.altloc = ascii_uppercase[i-26]
						else:
									continue
									#multiconf.resi = multiconf.resi + 1
									#multiconf.altloc = ascii_uppercase[i-27]
						if i == 0:
									final_conf = multiconf
						else:
									final_conf = final_conf.combine(multiconf)
			fname = os.path.join(self.options.directory, f"{prefix}.pdb")
			final_conf.tofile(fname)
