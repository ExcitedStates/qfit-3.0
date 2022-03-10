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
    p.add_argument("-sv", "--scale-rmask", dest="scale_rmask", default=1.0,
                   metavar="<float>", type=float,
                   help="Scaling factor for soft-clash mask radius")
    p.add_argument("-dc", "--density-cutoff", default=0.2,
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
    p.add_argument("-bs", "--bulk-solvent-level", default=0.1,
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
        self._voxel_volume = self.xmap.unit_cell.calc_volume()
        self._voxel_volume /= self.xmap.array.size
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
        if len(self.water.altloc) > 1:
           self.water = self.water.extract('altloc', self.water.altloc[0], '==')
           if len(self.water.resi) > 1:
              logger.error("Duplicate water molecules found. Please run remove_duplicates")
              return 
        self.protein = self.structure.extract('resn', 'HOH', '!=')
        self.full_occ = self.protein.extract('q', 1.0, '==')  #subset out protein that is full occupancy
        
        #for labeling water molecule numbers
        self.n = len(list(np.unique(self.structure.extract('resn', 'HOH', '!=').resi))) + 1
        #get a list of all residues
        residues = list(np.unique(self.structure.extract('record', 'ATOM', '==').resi)) #need to figure out with multiple chains
        
        for r in residues:
          print(r)
          self.residue = self.structure.extract('resi', r, '==')
          self.residue = self.residue.combine(self.water)
          
          self.water_holder = copy.deepcopy(self.water) 
          self.water_holder_coor = np.empty((1, 3))
          self.water_holder_coor[:] = np.nan

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
        multiconformer = multiconformer.reorder()
        multiconformer.tofile(fname, self.structure.scale, self.structure.cryst_info)
        return multiconformer
   

    def _run_water_sampling(self, xmap):
        """Run qfit water on each residue."""
        r_pro = self.residue.extract('resn', 'HOH', '!=')
        r_pro._init_clash_detection()
        #self.residue._init_clash_detection()
        self._update_transformer(self.residue) #this should now include water molecules
        self._bs = []
        self._coor_set = []

        altlocs = np.unique(r_pro.altloc)
        if len(altlocs) > 1:
             #full occupancy portion of the residue
             pro_full = r_pro.extract('q', 1.0, '==') 
             for a in altlocs:
              if a == '': continue # only look at 'proper' altloc
              self.pro_alt = pro_full.combine(r_pro.extract('altloc', a, '=='))
              self.base_residue = self.pro_alt.combine(self.water)
              prot_only_coor = np.concatenate((self.pro_alt.coor, self.water_holder_coor))
              
              self._coor_set.append(prot_only_coor)
              self._bs.append(self.base_residue.b)
              if self.residue.resn[0] in ('ALA', 'GLY'): 
                rotamer = 'all'
              else:
               rotamer = self.choose_rotamer(r_pro.resn[0], self.pro_alt, a)

              #get distance of water molecules from protein atoms
              close_atoms = WATERS[self.pro_alt.resn[0]][rotamer]
              for i in range(0, len(close_atoms)):
                atom = list(close_atoms[i+1].keys())
                dist = list(close_atoms[i+1].values()) #[i+1]
                wat_loc = self.least_squares(r_pro.extract('name', atom[0],'==').coor, r_pro.extract('name', atom[1],'==').coor, r_pro.extract('name', atom[2],'==').coor, r_pro.extract('name', atom[3],'==').coor, r_pro.extract('name', atom[4],'==').coor, dist[0], dist[1], dist[2], dist[3], dist[4])
                if wat_loc == 'None': continue
              
              #is new water location supported by density
                values = xmap.interpolate(wat_loc)
                if np.min(values) < 0.3: 
                  continue
                else:
                  if self._run_water_clash(self.water):
                    self._place_waters(wat_loc, a, r_pro.resn[0]) #place all water molecules along with residue!  
        else:
            self.base_residue = r_pro.combine(self.water)
            prot_only = r_pro.combine(self.water_holder)
            prot_only_coor = np.concatenate((r_pro.coor, self.water_holder_coor))
            self._coor_set.append(prot_only_coor)
            self._bs.append(self.base_residue.b)
            if self.residue.resn[0] in ('ALA', 'GLY'): 
              rotamer = 'all'
            else:
              rotamer = self.choose_rotamer(self.residue.resn[0], r_pro, '')
            close_atoms = WATERS[self.residue.resn[0]][rotamer]
            #always add just protein to self.coor_set

            #new_coor_set.append()
            for i in range(0, len(close_atoms)):
              atom = list(close_atoms[i+1].keys())
              dist = list(close_atoms[i+1].values())
              wat_loc = self.least_squares(r_pro.extract('name', atom[0],'==').coor, r_pro.extract('name', atom[1],'==').coor, r_pro.extract('name', atom[2],'==').coor, r_pro.extract('name', atom[3],'==').coor, r_pro.extract('name', atom[4],'==').coor, dist[0], dist[1], dist[2], dist[3], dist[4]) 
              #is new water location supported by density
              values = xmap.interpolate(wat_loc)
              if np.min(values) < 0.3: #density cutoff value
                continue
              else:
                if self._run_water_clash(self.water):
                #place all water molecules along with residue!  
                  self._place_waters(wat_loc, '', r_pro.resn[0])
        #now we need to remove/adjust overlapping water molecules

        #QP
        self.conformer = self.base_residue
        self._occupancies = self.base_residue.q
        self._write_intermediate_conformers(prefix=f"{r_pro.resi[0]}_sample")
        self._convert()
        self._solve()
        self._update_conformers()
        self._write_intermediate_conformers(prefix=f"{r_pro.resi[0]}_qp_solution")

        # MIQP score conformer occupancy
        #self._convert()
        #self._solve(threshold=self.options.threshold,
                        #cardinality=self.options.cardinality, loop_range=[0.34, 0.25, 0.2, 0.16, 0.14])
        #self._update_conformers()


        #write out multiconformer residues
        conformers = self.get_conformers()
        nconformers = len(conformers)
        if nconformers < 1:
            msg = ("No conformers could be generated. "
                   "Check for initial clashes.")
            raise RuntimeError(msg)

        pro_coor = []
        if nconformers == 1:
            #determine if HOH location is nan
            mc_residue = Structure.fromstructurelike(conformers[0])
            if np.isnan(np.sum(mc_residue.extract('resn', 'HOH', '==').coor)):
               mc_residue = mc_residue.extract('resn', 'HOH', '!=')
            mc_residue.altloc = ''
        
        else:
            #split protein and water
            #first determine if HOH == nan
            for altloc, conformer in zip(ascii_uppercase, conformers):
                #if the water value is NaN
                if np.isnan(np.sum(conformer.extract('resn', 'HOH', '==').coor)): 
                    #only select the protein
                    conformer = conformer.extract('resn', 'HOH', '!=') 
                else:
                   pro_conf = conformer.extract('resn', 'HOH', '!=')
                   wat_conf = conformer.extract('resn', 'HOH', '!=')
                if len(pro_coor) == 0: #if no protein has been placed yet
                   mc_residue = pro_conf
                   wat_conf.n = self.n 
                   self.n += 1 
                   mc_residue.combine(wat_conf)
                   pro_coor.append(pro_conf.coor)

                else:
                  delta = np.array(pro_coor) - np.array(pro_conf.coor) #determine if protein coor already exists
                  if np.sum(delta) > 0: #protein coor is new, get new altloc and append
                     #conformer.altloc = altloc
                     wat_conf.n = self.n
                     mc_residue = mc_residue.combine(pro_conf)
                     mc_residue = mc_residue.combine(wat_conf)
                     pro_coor.append(pro_conf.coor)
                  else:

                     #determine which alt loc the protein is closest to 
                     #print(mc_residue.altloc[dist_pdb == np.amin(dist_pdb)])
                      #only add water
                      wat_conf.resi = self.n
                      self.n += 1
                      mc_residue = mc_residue.combine(wat_conf)

        mc_residue = mc_residue.reorder()
        fname = os.path.join(self.options.directory,
                             f"{self.residue.resi[0]}_resi_waternew.pdb")
        mc_residue.tofile(fname)

        #residue.tofile(str(self.residue.resi[0]) + '_resi_waternew.pdb')

    def _run_water_clash(self, water):
        self._cd = ClashDetector(water, self.full_occ, scaling_factor=self.options.clash_scaling_factor)
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
            (x - x4)**2 + (y - y4)**2 + (z - z4)**2 - (dist_5)**2
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
      
    def trilaterate_3D(self,p1,p2,p3,p4,r1,r2,r3,r4):
      e_x=(p2.flatten()-p1.flatten())/np.linalg.norm(p2.flatten()-p1.flatten())
      i=np.dot(e_x,(p3.flatten()-p1.flatten()))
      print(i.shape)
      e_y=(p3.flatten()-p1.flatten()-(i*e_x))/(np.linalg.norm(p3.flatten()-p1.flatten()-(i*e_x)))
      print(e_y.shape)
      print(e_x.shape)
      e_z=np.cross(e_x,e_y)
      d=np.linalg.norm(p2.flatten()-p1.flatten())
      j=np.dot(e_y,(p3.flatten()-p1.flatten()))
      x=((r1**2)-(r2**2)+(d**2))/(2*d)
      y=(((r1**2)-(r3**2)+(i**2)+(j**2))/(2*j))-((i/j)*(x))
      z1=np.sqrt(r1**2-x**2-y**2)
      z2=np.sqrt(r1**2-x**2-y**2)*(-1)
      ans1=p1+(x*e_x)+(y*e_y)+(z1*e_z)
      ans2=p1+(x*e_x)+(y*e_y)+(z2*e_z)
      dist1=np.linalg.norm(p4-ans1)
      dist2=np.linalg.norm(p4-ans2)
      print('returning')
      if np.abs(r4-dist1)<np.abs(r4-dist2):
        return ans1.reshape(3, 1).T
      else: 
        return ans2.reshape(3, 1).T


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
        water = self.base_residue.extract('resn', 'HOH', '==')
        water.resi = self.n #giving each water molecule its own resi
        water.chain = 'S'
        if np.unique(self.base_residue.extract('resn', resn, '==')).all() == 1.0:
          water.q = 1.0
          water.altloc = ''
        else:
          water.q = np.unique(self.base_residue.q)[0]
          water.altloc = altloc
        water.coor = wat_loc
        water.b = np.mean(self.base_residue.b)*1.2 #make b-factor higher
        residue = self.base_residue.extract('resn', resn, '==').combine(water)
        self._coor_set.append(residue.coor)
        self._bs.append(residue.b)
        self.n += 1 

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
            self._transformer.mask(self._rmask)
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

        logger.debug(f"Remaining valid conformations: {len(self._coor_set)}")

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

    def _write_intermediate_conformers(self, prefix="_conformer"):
        for n, coor in enumerate(self._coor_set):
            self.conformer.coor = coor
            fname = os.path.join(self.options.directory, f"{prefix}_{n}.pdb")

            data = {}
            for attr in self.conformer.data:
                array1 = getattr(self.conformer, attr)
                data[attr] = array1[self.conformer.active]
            Structure(data).tofile(fname)


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
    qfit = prepare_qfit_water(options=options)

    multiconformer = qfit.run()
