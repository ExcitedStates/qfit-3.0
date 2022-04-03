import gc
from .qfit import QFitRotamericResidueOptions
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
from string import ascii_uppercase
from scipy.optimize import least_squares

from .clash import ClashDetector
from .logtools import setup_logging, log_run_info, poolworker_setup_logging, QueueListener
from . import MapScaler, Structure, XMap
from .structure.rotamers import ROTAMERS
from .structure.WATER_LOCS_5 import WATERS
from .structure.chi1 import chi_atoms
from .transformer import Transformer
from .solvers import QPSolver, MIQPSolver



logger = logging.getLogger(__name__)
os.environ["OMP_NUM_THREADS"] = "1"

class QFitWaterOptions(QFitRotamericResidueOptions):
				def __init__(self):
								super().__init__()
								self.nproc = 1
								self.verbose = True
								self.omit = False
								self.rotamer = None
								self.water = None
								self.pdb = None


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
								r_pro = self.residue.extract('resn', 'HOH', '!=')
								r_pro._init_clash_detection()
								self._update_transformer(self.residue) #this should now include water molecules
								#self.residue._init_clash_detection()
								self.water_holder_coor = np.empty((1, 3))
								self.water_holder_coor[:] = np.nan
								new_coor_set = []
								new_bs = []
								occ = []
								all_water_coor = []
								water_coor = []
								self.n = 100
								altlocs = np.unique(r_pro.altloc)
								if len(altlocs) > 1:
												 #full occupancy portion of the residue
												 pro_full = r_pro.extract('q', 1.0, '==') 
												 for a in altlocs:
														water_coor = []
														if a == '': continue # only look at 'proper' altloc
														self.pro_alt = pro_full.combine(r_pro.extract('altloc', a, '=='))
														if self.residue.resn[0] in ('ALA', 'GLY'): 
																rotamer = 'all'
														else:
														 rotamer = self.choose_rotamer(r_pro.resn[0], self.pro_alt, a)

														#get distance of water molecules from protein atoms
														close_atoms = WATERS[self.pro_alt.resn[0]][rotamer]
														#create base residue that matches number of close atoms
														self.water.coor = self.water_holder_coor
														for i in range(len(close_atoms)-1):
																self.water.atomid = i + np.max(self.pro_alt.atomid) + 2
																water = copy.deepcopy(self.water)
																water.atomid = i + np.max(self.pro_alt.atomid) + 1
																if i == 0:
																	self.all_waters = water.combine(self.water)
																else:
																	self.all_waters = self.all_waters.combine(self.water)
														self.base_residue = self.pro_alt.combine(self.all_waters)
														new_coor_set.append(self.base_residue.coor)
														new_bs.append(self.base_residue.b)
														occ.append(self.base_residue.q)
														for i in range(0, len(close_atoms)):
																atom = list(close_atoms[i+1].keys())
																dist = list(close_atoms[i+1].values())
																if self.residue.resn[0] == 'GLY':
																		wat_loc = self.least_squares_gly(self.pro_alt.extract('name', atom[0],'==').coor, self.pro_alt.extract('name', atom[1],'==').coor, self.pro_alt.extract('name', atom[2],'==').coor, self.pro_alt.extract('name', atom[3],'==').coor, dist[0], dist[1], dist[2], dist[3])
																else:
																		wat_loc = self.least_squares(self.pro_alt.extract('name', atom[0],'==').coor, self.pro_alt.extract('name', atom[1],'==').coor, self.pro_alt.extract('name', atom[2],'==').coor, self.pro_alt.extract('name', atom[3],'==').coor, self.pro_alt.extract('name', atom[4],'==').coor, dist[0], dist[1], dist[2], dist[3], dist[4])
																if wat_loc == 'None': print('none')#continue

																#is new water location supported by density
																values = self.xmap.interpolate(wat_loc)
																if np.min(values) < 0.3: 
																		continue
																else:
																		self.water.coor = wat_loc
																		if self._run_water_clash(self.water):
																				water_coor.append(wat_loc)

														
														for i in range(1, len(water_coor)+1):
																wat = []
																wat = list(map(list, itertools.combinations(water_coor, i)))
																for i, coor in enumerate(wat):
																	water_place = []
																	for i in range(len(coor)):
																		water_place.append(coor[i])
																	for l in range(len(close_atoms)-len(water_place)):
																		water_place.append(self.water_holder_coor)
																	coor, b = self._place_waters(water_place, a, r_pro.resn[0]) #place all water molecules along with residue!
																	#print('place:')
																	#print(coor)
																	new_coor_set.append(coor) 
																	new_bs.append(b)
								else:
												self.base_residue = r_pro.combine(self.water)
												prot_only_coor = np.concatenate((r_pro.coor, self.water_holder_coor))
												new_coor_set.append(prot_only_coor)
												new_bs.append(self.base_residue.b)
												occ.append(self.base_residue.q)
												if self.residue.resn[0] in ('ALA', 'GLY'): 
														rotamer = 'all'
												else:
														rotamer = self.choose_rotamer(self.residue.resn[0], r_pro, '')
												close_atoms = WATERS[self.residue.resn[0]][rotamer]

												for i in range(0, len(close_atoms)):
														atom = list(close_atoms[i+1].keys())
														dist = list(close_atoms[i+1].values())
														if self.residue.resn[0] == 'GLY':
																		wat_loc = self.least_squares_gly(r_pro.extract('name', atom[0],'==').coor, r_pro.extract('name', atom[1],'==').coor, r_pro.extract('name', atom[2],'==').coor, r_pro.extract('name', atom[3],'==').coor, dist[0], dist[1], dist[2], dist[3])
														else:
																		wat_loc = self.least_squares(r_pro.extract('name', atom[0],'==').coor, r_pro.extract('name', atom[1],'==').coor, r_pro.extract('name', atom[2],'==').coor, r_pro.extract('name', atom[3],'==').coor, r_pro.extract('name', atom[4],'==').coor, dist[0], dist[1], dist[2], dist[3], dist[4])
														water_coor.append(wat_loc)
														#is new water location supported by density
														values = self.xmap.interpolate(wat_loc)
														if np.min(values) < 0.3: #density cutoff value
																continue
														else:
																self.water.coor = wat_loc
																if self._run_water_clash(self.water):
																#place all water molecules along with residue!  
																		coor, b = self._place_waters(wat_loc, '', r_pro.resn[0])
																		new_coor_set.append(coor) 
																		new_bs.append(b)
																else:
																	print('clash') 

								self.conformer = self.base_residue
								self.conformer.q = 1.0
								self._coor_set = new_coor_set
								#for i in self._coor_set:
								#	print('new coor')
								#	print(i)
								self._bs = new_bs
								#QP
								#print(f"Remaining valid conformations: {len(self._coor_set)}")
								#self._write_water_locations(all_water_coor, prefix=f"{r_pro.resi[0]}_water_sample")
								self._write_intermediate_conformers(prefix=f"{r_pro.resi[0]}_sample")
								print(f"Remaining valid conformations: {len(self._coor_set)}")
								self._convert()
								#for i in self._coor_set:
								#	print('new coor 2')
								#	print(i)
								self._solve()
								self._update_conformers()
								print(f"Remaining valid conformations (post QP): {len(self._coor_set)}")
								print(len(self._coor_set))
								self._write_intermediate_conformers(prefix=f"{r_pro.resi[0]}_qp_solution")

								self._convert()
								self._solve(threshold=self.options.threshold,
																						 cardinality=self.options.cardinality) #, loop_range=[0.34, 0.25, 0.2, 0.16, 0.14]
								self._update_conformers()
								self._write_intermediate_conformers(prefix=f"{r_pro.resi[0]}_miqp_solution")



								#write out multiconformer residues
								conformers = self.get_conformers()
								nconformers = len(conformers)
								if nconformers < 1:
												msg = ("No conformers could be generated. "
																		 "Check for initial clashes.")
												raise RuntimeError(msg)

								d_pro_coor = {}
								pro_coor = []
								if nconformers == 1:
												#determine if HOH location is nan
												conformer = Structure.fromstructurelike(conformers[0])
												for i in conformer.atomid:
														if np.isnan(np.sum(conformer.extract('atomid', a, '==').coor)): continue
														else:
															try:
																mc_residue.combine(conformer.extract('atomid', a, '=='))
															except Exception:
																mc_residue = conformer.extract('atomid', a, '==')
												for w in mc_residue.extract('resn', 'HOH', '=='):
														mc_residue.extract('resn', 'HOH', '==').resi = self.n
														self.n +=1
												mc_residue.altloc = ''
								
								else:
												a = 0
												water_nan = False
												for i in range(len(conformers)):
														conf = Structure.fromstructurelike(conformers[i])
														conformer = [] #create holder
														for i in conf.atomid:
															if np.isnan(np.sum(conf.extract('atomid', i, '==').coor)): continue
															else:
																try:
																	conformer = conformer.combine(conf.extract('atomid', i, '=='))
																except Exception:
																	conformer = conf.extract('atomid', i, '==')
														if not pro_coor: #if no protein has been placed yet
															 conformer.altloc = ascii_uppercase[a]
															 d_pro_coor[ascii_uppercase[a]] = conformer.extract('resn', 'HOH', '!=').coor
															 pro_coor.append(conformer.extract('resn', 'HOH', '!=').coor)
															 a += 1 
															 mc_residue = conformer
														else:
															 delta = []
															 for i in pro_coor:
																	delta.append(np.linalg.norm(i - conformer.extract('resn', 'HOH', '!=').coor, axis=1))#determine if protein coor already exists
															 if np.sum(delta) > 0: #protein coor is new, get new altloc and append
																	mc_residue = mc_residue.combine(conformer.extract('resn', 'HOH','!='))
																	d_pro_coor[ascii_uppercase[a]] = conformer.extract('resn', 'HOH', '!=').coor
																	pro_coor.append(conformer.extract('resn', 'HOH', '!=').coor)


																	wat = conformer.extract('resn', 'HOH', '==')
																	print('wat:')
																	print(wat.coor)
																	if len(wat.coor) > 0:
																		#if len(mc_residue.extract(f"resn HOH and altloc {a}").coor) == 0:
																		#		mc_residue = mc_residue.combine(wat.extract('atomid', wat.atomid[0], '=='))
																		for n1, c1 in enumerate(wat.coor):
																			if np.any(np.linalg.norm(c1- mc_residue.extract(f"resn HOH").coor, axis=1) < 1.2):
																					print(c1)
																					print(mc_residue.extract(f"resn HOH").coor)
																					atom1 = wat.atomid[n1]
																					#atom2 = mc_residue.extract(f"resn HOH").atomid[n2]
																					#find closest
																					delta2 = 100
																					for n in mc_residue.extract(f"resn HOH").atomid:
																							tmp = np.linalg.norm(c1 - mc_residue.extract("atomid", n, "==").coor)
																							if tmp < delta2:
																								 delta2 == tmp
																								 atom2 = n
																					mc_residue.extract("atomid", atom2, "==").q += wat.extract('atomid', atom1, '==').q[0]
																			else:
																					atom1 = wat.atomid[n1]
																					wat.extract('atomid', atom1, '==').resi = self.n
																					print('resi:')
																					print(self.n)
																					print(wat.extract('atomid', atom1, '==').resi)
																					self.n += 1
																					wat.extract('atomid', atom1, '==').altloc = ascii_uppercase[a]
																					mc_residue = mc_residue.combine(wat.extract('atomid', atom1, '=='))
																					print(mc_residue.resi)
																	a += 1
															 else:
																		 alt_conf = ''
																		 for alt, xcoor in d_pro_coor.items():
																					if np.array_equal(conformer.extract('resn', 'HOH', '!=').coor, coor):
																							alt_conf = alt
																							pro_occ = conformer.extract('resn', 'HOH', '!=').q[0]
																							#collapsing the protein part of the residue
																							mc_residue.extract(f'altloc {alt} and resn {r_pro.resn[0]}').q += pro_occ 
																							#now look at water
																							wat = conformer.extract('resn', 'HOH', '==')
																							if not len(wat.coor) == 0:			
																								for n1, c1 in enumerate(wat.coor):
																										if np.any(np.linalg.norm(c1- mc_residue.extract(f"resn HOH").coor, axis=1) < 1.2):
																											 print(c1)
																											 print(mc_residue.extract(f"resn HOH").coor)
																											 atom1 = wat.atomid[n1]
																											 delta2 = 100
																											 for n in mc_residue.extract(f"resn HOH").atomid:
																													tmp = np.linalg.norm(c1 - mc_residue.extract("atomid", n, "==").coor)
																													if tmp < delta2:
																													 delta2 == tmp
																													 atom2 = n
																											 mc_residue.extract("atomid", atom2, "==").q += wat.extract('atomid', atom1, '==').q[0]
																										else:
																											 atom1 = wat.atomid[n1]
																											 wat.extract('atomid', atom1, '==').resi = self.n
																											 self.n += 1
																											 wat.extract('atomid', atom1, '==').altloc = ascii_uppercase[a]
																											 mc_residue = mc_residue.combine(wat.extract('atomid', atom1, '=='))
								#mc_residue = mc_residue.collapse_backbone(mc_residue.resi[0], mc_residue.chain[0])
								mc_residue = mc_residue.reorder()

								if len(np.unique(mc_residue.altloc)) == 1:
												mc_residue.altloc = ''
								fname = os.path.join(self.options.directory,
																												 f"{self.residue.resi[0]}_resi_waternew.pdb")
								mc_residue.tofile(fname)


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


				def choose_rot(self, dihedral, r):
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

				def choose_rotamer(self, resn, r_pro,a):
								chi1 = chi_atoms[resn]
								dihedral = self.calc_chi1(r_pro.extract(f'name {chi1[0]}').coor, r_pro.extract(f'name {chi1[1]}').coor, r_pro.extract(f'name {chi1[2]}').coor, r_pro.extract(f'name {chi1[3]}').coor)
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
								# Create and run QP or MIQP solver
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
								"""Removes conformers with occupancy lower than cutoff.

								Args:
												cutoff (float, optional): Lowest acceptable occupancy for a conformer.
																Cutoff should be in range (0 < cutoff < 1).
								"""
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
