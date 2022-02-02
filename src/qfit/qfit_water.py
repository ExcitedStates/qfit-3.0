import gc
from .qfit import QFitRotamericResidue, QFitRotamericResidueOptions
from .qfit import QFitSegment, QFitSegmentOptions
import multiprocessing as mp
from tqdm import tqdm
import os.path
import os
import sys
import argparse
from .custom_argparsers import ToggleActionFlag, CustomHelpFormatter
import logging
import traceback
import pandas as pd
import numpy as np
import time 
import math
#import ray


from .logtools import setup_logging, log_run_info, poolworker_setup_logging, QueueListener
from . import MapScaler, Structure, XMap
from .structure.rotamers import ROTAMERS
from .structure.waters import WATERS
from .structure.chi1 import chi_atoms



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
    p.add_argument("-bs", "--bulk-solvent-level", default=0.3,
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


class QFitWaterOptions(QFitRotamericResidueOptions, QFitSegmentOptions):
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

    def run(self):
        if self.options.pdb is not None:
            self.pdb = self.options.pdb + '_'
        else:
            self.pdb = ''
        self.water = self.structure.extract('resn', 'HOH', '==')#create blank water object
        self.water = self.water.extract('resi', self.water.resi[0], '==')

        #get a list of all residues
        residues = list(np.unique(self.structure.extract('record', 'ATOM', '==').resi)) #need to figure out with multiple chains
        
        for r in residues:
          print(r)
          #subset map
          xmap = self.xmap.extract(self.structure.extract('resi', r, '==').coor, padding=5)
          #This funciton will run each residue seperately
          multiconformer = self._run_water_sampling(self.structure.extract('resi', r, '=='), xmap) 
        #self._run_water_clash() #this will be segment-ish and deal with clashes
        return multiconformer
    

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
        #if np.unique(r.resn)[0] in ('PRO', 'ALA'):
        #   rotamer = 'all'
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


    def _place_waters(self, residue, wat_loc, altloc, resn):
        """create new residue structure with residue atoms & new water atoms
           take OG residue, output new residue
        """
        water_new = self.water.copy() #create blank water structure
        water_new.resi = 0
        water_new.chain = 'S'
        if np.unique(residue.extract('resn', resn, '==')).all() == 1.0:
          water_new.q = 1.0
        else:
          water_new.q = np.unique(residue.extract('altloc', altloc, '==').q)[0]
          water_new.altloc = altloc
        water_new.coor = wat_loc
        water_new.b = np.mean(residue.b) #amend in the future
        r = residue.combine(water_new)
        r.tofile(str(residue.resi[0]) + '_resi_waternew.pdb')
        return r

    def choose_rotamer(self, resn, r_pro, a):
        chi1 = chi_atoms[resn]
        atom_altlocs = ['altloc_0', 'altloc_1', 'altloc_2', 'altloc_3']
        for i in range(len(chi1)): # we need to label atoms that don't have altloc (ie backbones)
          if len(np.unique(r_pro.extract('name', chi1[i], '==').altloc)) == 1:
             atom_altlocs[i] = '""'
          else: 
             atom_altlocs[i] = a
        dihedral = self.calc_chi1(r_pro.extract(f'name {chi1[0]} and altloc {atom_altlocs[0]}').coor, r_pro.extract(f'name {chi1[1]} and altloc {atom_altlocs[1]}').coor, r_pro.extract(f'name {chi1[2]} and altloc {atom_altlocs[2]}').coor, r_pro.extract(f'name {chi1[3]} and altloc {atom_altlocs[3]}').coor)
        rotamer = self.choose_rot(dihedral, r_pro)
        return rotamer 

    def _run_water_sampling(self, r, xmap):
        """Run qfit water on each residue."""
        r_pro = r.extract('resn', 'HOH', '!=')
        altlocs = np.unique(r_pro.altloc)
        if len(altlocs) > 1:
             for a in altlocs:
              if a == '': continue # only look at 'proper' altloc
              #get rotamer 
              if r.resn[0] in ('PRO', 'ALA'): 
                rotamer = 'all'
              else:
               rotamer = self.choose_rotamer(r.resn[0], r_pro, a)

              #get distance of water molecules from protein atoms
              close_atoms = WATERS[r_pro.resn[0]][rotamer]
              for i in range(0, len(close_atoms)):
                atom = list(close_atoms[i][i+1].keys())
                dist = list(close_atoms[i][i+1].values())
                atom_altlocs = ['altlocs_0', 'altloc_1', 'altloc_2']
                for i in range(len(atom)): # we need to label atoms that don't have altloc (ie backbones)
                  if len(np.unique(r_pro.extract('name', atom[i], '==').altloc)) == 1:
                   atom_altlocs[i] = '""'
                  else: 
                   atom_altlocs[i] = a
                #print(r_pro)
                #print(r_pro.extract(f'name {atom[0]} and altloc {atom_altlocs[0]}').coor)
                wat_loc = self.trilaterate(r_pro.extract(f'name {atom[0]} and altloc {atom_altlocs[0]}').coor, r_pro.extract(f'name {atom[1]} and altloc {atom_altlocs[1]}').coor, r_pro.extract(f'name {atom[2]} and altloc {atom_altlocs[2]}').coor, dist[0], dist[1], dist[2]) 
                if wat_loc == 'None': continue
              
              #is new water location supported by density
                values = xmap.interpolate(wat_loc)
                if np.min(values) < 0.3: 
                  continue
                else: 
                  r = self._place_waters(r, wat_loc, a, r_pro.resn[0]) #place all water molecules along with residue!  
        else:
            if r.resn[0] in ('PRO', 'ALA'): 
              rotamer = 'all'
            else:
              rotamer = self.choose_rotamer(r.resn[0], r_pro, '')
            close_atoms = WATERS[r.resn[0]][rotamer]
            for i in range(0, len(close_atoms)):
              atom = list(close_atoms[i][i+1].keys())
              dist = list(close_atoms[i][i+1].values())
              wat_loc = self.trilaterate(r_pro.extract('name', atom[0],'==').coor, r_pro.extract('name', atom[1],'==').coor, r_pro.extract('name', atom[2],'==').coor, dist[0], dist[1], dist[2]) 
              #is new water location supported by density
              values = xmap.interpolate(wat_loc)
              if np.min(values) < 0.3: #density cutoff value
                continue
              else:
                #place all water molecules along with residue!  
                r = self._place_waters(r, wat_loc, '', r.resn[0])

        # should we be playing aorund with different density cutoffs at this point.




            #fcalc based on oxygen water (0.1) * occ & delete if density not supported
            #else, keep it
            #ouptut individual residue & water

        #combine backbone water molecules & residues + water molecules
          #for each water molecule, if does not clash:
            #confirm H-bonding patterns
            #if it does clash:
              #if clash with fully occupied protein atom: ????
              #if clash with partially occupied protein atom: 
                #adjust occupancy of water molecule
                #assess H-bonding patterns
                  #if acceptable, keep, else throw out



        



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
