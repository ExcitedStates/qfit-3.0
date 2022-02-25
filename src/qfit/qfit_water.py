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
from string import ascii_uppercase

from .clash import ClashDetector
from .logtools import setup_logging, log_run_info, poolworker_setup_logging, QueueListener
from . import MapScaler, Structure, XMap
from .structure.rotamers import ROTAMERS
from .structure.waters import WATERS
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
    p.add_argument("-sv", "--scale-rmask", dest="scale_rmask", default=1.0,
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
    p.add_argument("--padding", default=8.0,
                   metavar="<float>", type=float,
                   help="Padding size for map creation")
    p.add_argument("--waters-clash", action=ToggleActionFlag, dest="waters_clash", default=True,
                   help="Consider waters for soft clash detection")

    
    #arguments
    p.add_argument('-cf', "--clash-scaling-factor", default=0.75,
                   metavar="<float>", type=float,
                   help="Set clash scaling factor")
    p.add_argument('-ec', "--external-clash", action="store_true", dest="external_clash",
                   help="Enable external clash detection during sampling")
    p.add_argument("-bs", "--bulk-solvent-level", default=0.1,
                   metavar="<float>", type=float,
                   help="Bulk solvent level in absolute values")
    p.add_argument("-c", "--cardinality", default=5,
                   metavar="<int>", type=int,
                   help="Cardinality constraint used during MIQP")
    p.add_argument("-t", "--threshold", default=0.2,
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


class QFitWaterOptions(QFitRotamericResidueOptions):
    def __init__(self):
        super().__init__()
        self.nproc = 1
        self.verbose = True
        self.omit = False
        self.checkpoint = False
        self.pdb = None
        self.rotamer = None
        self.water = None


class QFitWater:
    def __init__(self, structure, xmap, options):
        self.xmap = xmap
        self.structure = structure
        self.options = options
        self._xmap_model = xmap.zeros_like(self.xmap)
        self._xmap_model.set_space_group("P1")
        self._smax = None
        self._smin = None
        self._simple = True
        self._rmask = 1.5 


    def run(self):
        if self.options.pdb is not None:
            self.pdb = self.options.pdb + '_'
        else:
            self.pdb = ''
        self.water = self.structure.extract('resn', 'HOH', '==')#create blank water object
        self.water = self.water.extract('resi', self.water.resi[0], '==')
        self.protein = self.structure.extract('record', 'ATOM', '==')

        #for labeling water molecule numbers
        self.n = len(list(np.unique(self.structure.extract('resn', 'HOH', '!=').resi))) + 1
        #get a list of all residues
        residues = list(np.unique(self.structure.extract('record', 'ATOM', '==').resi)) #need to figure out with multiple chains
        
        for r in residues:
          print(r)
          # These lists will be used to combine coor_sets output for
          self._all_coor_set = []
          self._all_bs = []
          #subset map
          #This funciton will run each residue seperately
          self.residue = self.structure.extract('resi', r, '==')
          self.residue = self.residue.combine(self.water)
          xmap = self.xmap.extract(self.residue.coor, padding=8)
          self._bs = self.residue.b
          water = self._run_water_sampling(xmap) 
        
        #now that all the individual residues have run...
        # Combine all multiconformer residues into one structure
          resid = r
          directory = os.path.join(self.options.directory)
          fname = os.path.join(directory, f'{resid}_resi_waternew.pdb')
          if not os.path.exists(fname):
                continue
          residue_multiconformer = Structure.fromfile(fname)
          try:
              multiconformer = multiconformer.combine(residue_multiconformer)
          except:
              multiconformer = residue_multiconformer

        fname = os.path.join(self.options.directory,
                             "multiconformer_model_water.pdb")
        multiconformer.tofile(fname, self.structure.scale, self.structure.cryst_info)
        return multiconformer
        #self._run_water_clash() #this will be segment-ish and deal with clashes
   

    def _run_water_sampling(self, xmap):
        """Run qfit water on each residue."""
        r_pro = self.residue.extract('resn', 'HOH', '!=')
        self.residue._init_clash_detection()
        self._update_transformer(self.residue) #this should now include water molecules
        self._starting_coor_set = [self.residue.coor.copy()]
        self._starting_bs = [self.residue.b.copy()]
        self._coor_set = list(self._starting_coor_set)
        new_coor_set = []
        new_bs = []

        altlocs = np.unique(r_pro.altloc)
        if len(altlocs) > 1:
             #full occupancy portion of the residue
             pro_full = r_pro.extract('q', 1.0, '==') 
             for a in altlocs:
              if a == '': continue # only look at 'proper' altloc
              pro_alt = pro_full.combine(r_pro.extract('altloc', a, '=='))
              if self.residue.resn[0] in ('ALA', 'GLY'): 
                rotamer = 'all'
              else:
               rotamer = self.choose_rotamer(r_pro.resn[0], pro_alt, a)

              #get distance of water molecules from protein atoms
              close_atoms = WATERS[pro_alt.resn[0]][rotamer]
              for i in range(0, len(close_atoms)):
                atom = list(close_atoms[i][i+1].keys())
                dist = list(close_atoms[i][i+1].values())
                wat_loc = self.trilaterate(pro_alt.extract(f'name {atom[0]}').coor, pro_alt.extract(f'name {atom[1]}').coor, pro_alt.extract(f'name {atom[2]}').coor, dist[0], dist[1], dist[2]) 
                if wat_loc == 'None': continue
              
              #is new water location supported by density
                values = xmap.interpolate(wat_loc)
                if np.min(values) < 0.3: 
                  continue
                else:
                  #clash
                  #new_coor_set.append(new_coor)
                  #new_bs.append(b) 
                  self._place_waters(wat_loc, a, r_pro.resn[0]) #place all water molecules along with residue!  
        else:
            if self.residue.resn[0] in ('ALA', 'GLY'): 
              rotamer = 'all'
            else:
              rotamer = self.choose_rotamer(self.residue.resn[0], r_pro, '')
            close_atoms = WATERS[self.residue.resn[0]][rotamer]
            for i in range(0, len(close_atoms)):
              atom = list(close_atoms[i][i+1].keys())
              dist = list(close_atoms[i][i+1].values())
              wat_loc = self.trilaterate(r_pro.extract('name', atom[0],'==').coor, r_pro.extract('name', atom[1],'==').coor, r_pro.extract('name', atom[2],'==').coor, dist[0], dist[1], dist[2]) 
              #is new water location supported by density
              values = xmap.interpolate(wat_loc)
              if np.min(values) < 0.3: #density cutoff value
                continue
              else:
                if self._run_water_clash(self.protein, self.water):
                   #new_coor_set.append(water_loc)
                   #new_bs.append(np.mean(self.residue.b))
                #place all water molecules along with residue!  
                  self._place_waters(wat_loc, '', r_pro.resn[0])
        #now we need to remove/adjust overlapping water molecules

        #QP
        print('initial')
        self.conformer = self.residue
        self._coor_set = self.conformer.coor
        self._bs = self.residue.b
        self._occupancies = self.residue.q
        # logger.debug("Converting densities within run.")
        self._convert()
        # print('targets:')
        # print(self._target)
        # print('models:')
        # print(self._models)
        # print('solve:')
        self._solve()

        # print(self.residue.coor)
        # logger.debug("Updating conformers within run.")
        self._update_conformers()
        #write out multiconformer residues
        conformers = self.get_conformers()
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
        fname = os.path.join(self.options.directory,
                             f"{self.residue.resi[0]}_resi_waternew.pdb")
        mc_residue.tofile(fname)

        #residue.tofile(str(self.residue.resi[0]) + '_resi_waternew.pdb')

    def _run_water_clash(self, protein, water):
        full_occ = protein.extract('q', 1.0, '==')  #subset out protein that is full occupancy
        self._cd = ClashDetector(water, full_occ, scaling_factor=self.options.clash_scaling_factor)
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


    def trilaterate(self,P1,P2,P3,r1,r2,r3): 
        # find location of the water molecule                      
        temp1 = P2.flatten()- P1.flatten()
        e_x = temp1/np.linalg.norm(temp1)                              
        temp2 = P3.flatten() - P1.flatten()                                      
        i = np.dot(e_x,temp2)                                   
        temp3 = temp2 - i*e_x                               
        e_y = temp3/np.linalg.norm(temp3)                              
        e_z = np.cross(e_x,e_y)                                 
        d = np.linalg.norm(P2.flatten()-P1.flatten())                                      
        j = np.dot(e_y,temp2)                                   
        x = (r1*r1 - r2*r2 + d*d) / (2*d)                    
        y = (r1*r1 - r3*r3 -2*i*x + i*i + j*j) / (2*j)       
        temp4 = r1*r1 - x*x - y*y                            
        z = np.sqrt(temp4)                                      
        p_12_a = P1.flatten() + x*e_x + y*e_y + z*e_z                  
        p_12_b = P1.flatten() + x*e_x + y*e_y - z*e_z                  
        return p_12_a.reshape(3, 1).T #,p_12_b


    def _place_waters(self, wat_loc, altloc, resn):
        """create new residue structure with residue atoms & new water atoms
           take OG residue, output new residue
        """
        self.water.resi = self.n #giving each water molecule its own resi
        self.water.chain = 'S'
        if np.unique(self.residue.extract('resn', resn, '==')).all() == 1.0:
          self.water.q = 1.0
        else:
          self.water.q = np.unique(self.residue.extract('altloc', altloc, '==').q)[0]
          self.water.altloc = altloc
        self.water.coor = wat_loc
        self.water.b = np.mean(self.residue.b) #amend in the future
        self.residue = self.residue.combine(self.water)
        self.n += 1 

    def choose_rotamer(self, resn, r_pro, a):
        chi1 = chi_atoms[resn]

        dihedral = self.calc_chi1(r_pro.extract(f'name {chi1[0]}').coor, r_pro.extract(f'name {chi1[1]}').coor, r_pro.extract(f'name {chi1[2]}').coor, r_pro.extract(f'name {chi1[3]}').coor)
        rotamer = self.choose_rot(dihedral, r_pro)
        return rotamer 


    def _convert(self): #figure out why 28 atoms on conformer.coor and not 17?
        #28 would include alt loc + water
        """Convert structures to densities and extract relevant values for (MI)QP."""
        print("Converting conformers to density")
        print("Masking")
        self._transformer.reset(full=True) #converting self.xmap.array to zero
        #for n, coor in enumerate(self._coor_set):
        for n, coor in enumerate(self._coor_set):
            print('len self.conformer.coor')
            print(len(self.conformer.coor))
            #print(self.conformer.coor)
            self.conformer.coor = coor
            self._transformer.mask(self._rmask)
        #self.conformer.coor = self._coor_set
        #self._transformer.mask(self._rmask)
        mask = (self._transformer.xmap.array > 0)
        self._transformer.reset(full=True)

        nvalues = mask.sum()
        #print(nvalues)
        self._target = self.xmap.array[mask]
        nmodels = len(self._coor_set)
        self._models = np.zeros((nmodels, nvalues), float)
        #for n, coor in enumerate(self._coor_set):
        self.conformer.coor = self._coor_set
        print('conformer')
        print(self.conformer.coor)
        self.conformer.b = self._bs
        self._transformer.density()
        model = self._models
        model[:] = self._transformer.xmap.array[mask]
            #print(np.maximum(model, self.options.bulk_solvent_level, out=model))
        np.maximum(model, self.options.bulk_solvent_level, out=model)
        self._transformer.reset(full=True)

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
            else:
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

        # Filter all arrays & lists based on self._occupancies
        print(self._occupancies)
        filterarray = (self._occupancies >= cutoff)
        self._occupancies = self._occupancies[filterarray]
        self._coor_set = list(itertools.compress(self._coor_set, filterarray))
        self._bs = list(itertools.compress(self._bs, filterarray))

        logger.debug(f"Remaining valid conformations: {len(self._coor_set)}")

    def _update_transformer(self, structure):
        self.conformer = structure
        self._transformer = Transformer(
            structure, self._xmap_model,
            smax=self._smax, smin=self._smin,
            simple=self._simple,
            scattering=self.options.scattering,
        )
        print("[_BaseQFit._update_transformer]: Initializing radial density lookup table.")
        self._transformer.initialize()

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


def prepare_qfit_water(options):

    # Load structure and prepare it
    structure = Structure.fromfile(options.structure)
    if not options.hydro:
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
    qfit = prepare_qfit_water(options=options)

    multiconformer = qfit.run()
