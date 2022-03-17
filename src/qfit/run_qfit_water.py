import gc
from .qfit_water import QFitWaterOptions, QFitWater_Residue
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

def build_argparser():
	p = argparse.ArgumentParser(formatter_class=CustomHelpFormatter,
								description=__doc__)
	p.add_argument("map", type=str,
					 help="Density map in CCP4 or MRC format, or an MTZ file "
						"containing reflections and phases. For MTZ files "
						"use the --label options to specify columns to read.")
	p.add_argument("structure",
					 help="PDB-file containing multiconformer structure.")

	# Map input options
	p.add_argument("-l", "--label", default="FWT,PHWT",
					 metavar="<F,PHI>",
					 help="MTZ column labels to build density")
	p.add_argument('-r', "--resolution", default=None,
					 metavar="<float>", type=float,
					 help="Map resolution (Å) (only use when providing CCP4 map files)")
	p.add_argument("-m", "--resolution-min", default=None,
					 metavar="<float>", type=float,
					 help="Lower resolution bound (Å) (only use when providing CCP4 map files)")
	p.add_argument("-z", "--scattering", choices=["xray", "electron"], default="xray",
					 help="Scattering type")
	p.add_argument('-o', '--omit', action="store_true",
					 help="Treat map file as an OMIT map in map scaling routines")

	# Map prep options
	p.add_argument("--scale", action=ToggleActionFlag, dest="scale", default=True,
					 help="Scale density")
	p.add_argument("-sv", "--scale-rmask", dest="scale_rmask", default=0.8,
					 metavar="<float>", type=float,
					 help="Scaling factor for soft-clash mask radius")
	p.add_argument("-dc", "--density-cutoff", default=0.3,
					 metavar="<float>", type=float,
					 help="Density values below this value are set to <density-cutoff-value>")
	p.add_argument("-dv", "--density-cutoff-value", default=-1,
					 metavar="<float>", type=float,
					 help="Density values below <density-cutoff> are set to this value")
	p.add_argument("--subtract", action=ToggleActionFlag, dest="subtract", default=True,
					 help="Subtract Fcalc of neighboring residues when running qFit")
	p.add_argument("--padding", default=10.0,
					 metavar="<float>", type=float,
					 help="Padding size for map creation")

	
	#arguments
	p.add_argument('-cf', "--clash-scaling-factor", default=0.7,
					 metavar="<float>", type=float,
					 help="Set clash scaling factor")
	p.add_argument('-ec', "--external-clash", action="store_true", dest="external_clash",
					 help="Enable external clash detection during sampling")
	p.add_argument("-bs", "--bulk-solvent-level", default=0.0,
					 metavar="<float>", type=float,
					 help="Bulk solvent level in absolute values")
	p.add_argument("-c", "--cardinality", default=10,
					 metavar="<int>", type=int,
					 help="Cardinality constraint used during MIQP")
	p.add_argument("-t", "--threshold", default=0.1,
					 metavar="<float>", type=float,
					 help="Threshold constraint used during MIQP")
	p.add_argument("-hy", "--hydro", action="store_true", dest="hydro",
					 help="Include hydrogens during calculations")
	p.add_argument("--threshold-selection", dest="bic_threshold", action=ToggleActionFlag, default=True,
					 help="Use BIC to select the most parsimonious MIQP threshold")
	p.add_argument("-p", "--nproc", type=int, default=1, metavar="<int>",
					 help="Number of processors to use")

	# Global options
	p.add_argument("--random-seed", dest="random_seed",
					 metavar="<int>", type=int,
					 help="Seed value for PRNG")

	# Output options
	p.add_argument("-d", "--directory", default='.',
					 metavar="<dir>", type=os.path.abspath,
					 help="Directory to store results")
	p.add_argument("-v", "--verbose", action="store_true",
					 help="Be verbose")
	p.add_argument("--debug", action="store_true",
					 help="Log as much information as possible")
	p.add_argument("--write-intermediate-conformers", action="store_true",
					 help="Write intermediate structures to file (useful with debugging)")
	p.add_argument("--pdb", help="Name of the input PDB")

	return p

class QFitWater:
	def __init__(self, structure, xmap, options):
		self.xmap = xmap
		self.structure = structure
		self.options = options
		self.nproc = 1
		self.pdb = None

	def run(self):
		if self.options.pdb is not None:
			self.pdb = self.options.pdb + '_'
		else:
			self.pdb = ''
		self.water = self.structure.extract('resn', 'HOH', '==')#create blank water object
		self.water = self.water.extract('resi', self.water.resi[0], '==')
		if len(self.water.altloc) > 1:
			self.water = self.water.extract('altloc', self.water.altloc[0], '==')
			if len(self.water.resi) > 1:
				logger.error("Duplicate water molecules found. Please run remove_duplicates")
				return 

		#subset out protein that is full occupancy
		self.protein = self.structure.extract('resn', 'HOH', '!=')
		self.full_occ = self.protein.extract('q', 1.0, '==') 
		self.water_holder = copy.deepcopy(self.water)

		#for labeling water molecule numbers
		n = len(list(np.unique(self.structure.extract('resn', 'HOH', '!=').resi))) + 1
		#get a list of all residues
		residues = list(np.unique(self.structure.extract('record', 'ATOM', '==').resi)) #need to figure out with multiple chains

		for res in residues:
			print(res)
			residue = self.structure.extract('resi', res, '==')
			full_occ = self.full_occ
			xmap_reduced = self.xmap.extract(residue.coor, padding=self.options.padding)
			#xmap_reduced.tofile('reduced.ccp4')
			residue = residue.combine(self.water_holder)
			qfit = QFitWater_Residue(residue, full_occ, xmap_reduced, self.water_holder, n, self.options)
			qfit.run() 
			del xmap_reduced
			del qfit

		#now that all the individual residues have run...
		# Combine all multiconformer residues into one structure
		
		for res in residues:
			directory = os.path.join(self.options.directory)
			fname = os.path.join(directory, f'{res}_resi_waternew.pdb')
				#if not os.path.exists(fname): continue
			residue_multiconformer = Structure.fromfile(fname)
			for water in residue_multiconformer.extract('resn', 'HOH', '==').resi:
				  residue_multiconformer.extract('resi', water, '==').resi = n
				  n += 1 
			try:
					multiconformer = multiconformer.combine(residue_multiconformer)
			except:
					multiconformer = residue_multiconformer

		fname = os.path.join(self.options.directory, "multiconformer_model_water.pdb")
		multiconformer = multiconformer.reorder()
		multiconformer.tofile(fname, self.structure.scale, self.structure.cryst_info)
		
def prepare_qfit_water(options):

	# Load structure and prepare it
	structure = Structure.fromfile(options.structure)
	structure = structure.extract('e', 'H', '!=')

	# Load map and prepare it
	xmap = XMap.fromfile(
		options.map, resolution=options.resolution, label=options.label
	)
	xmap = xmap.canonical_unit_cell()
	if options.scale is True:
		scaler = MapScaler(xmap, scattering=options.scattering)
		radius = 1.5
		reso = None
		if xmap.resolution.high is not None:
			reso = xmap.resolution.high
		elif options.resolution is not None:
			reso = options.resolution
		if reso is not None:
			radius = 0.5 + reso / 3.0
		scaler.scale(structure, radius=options.scale_rmask*radius)

	return QFitWater(structure, xmap, options)


def main():
	"""Default entrypoint for qfit_protein."""

	p = build_argparser()
	args = p.parse_args(args=None)
	try:
		os.mkdir(args.directory)
	except OSError:
		pass

	# Apply the arguments to options
	options = QFitWaterOptions()
	options.apply_command_args(args)

	# Setup logger
	setup_logging(options=options)
	log_run_info(options, logger)

	# Build a QFitWater job

	qfit = prepare_qfit_water(options)
	qfit.run()
		
