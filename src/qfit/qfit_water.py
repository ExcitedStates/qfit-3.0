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
import ray


from .logtools import setup_logging, log_run_info, poolworker_setup_logging, QueueListener
from . import MapScaler, Structure, XMap
from .structure.rotamers import ROTAMERS



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
    p.add_argument("-pad", "--padding", default=8.0,
                   metavar="<float>", type=float,
                   help="Padding size for map creation")
    p.add_argument("--waters-clash", action=ToggleActionFlag, dest="waters_clash", default=True,
                   help="Consider waters for soft clash detection")

    
    #arguments
    p.add_argument('-cf', "--clash-scaling-factor", default=0.75,
                   metavar="<float>", type=float,
                   help="Set clash scaling factor")
    p.add_argument("-pad", "--padding", default=10.0,
                   metavar="<float>", type=float,
                   help="Padding size for map creation")
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


class QFitProteinOptions(QFitRotamericResidueOptions, QFitSegmentOptions):
    def __init__(self):
        super().__init__()
        self.nproc = 1
        self.verbose = True
        self.omit = False
        self.checkpoint = False
        self.pdb = None

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
        multiconformer = self._run_water_sampling()
        multiconformer = self._run_water_trum(structure)
        return multiconformer


    def _run_water_sampling(self):
        """Run qfit independently over all residues."""
        
        
        #delete all existing water molecules
        stru_nowater = self.structure.extract('resn', 'HOH', '!=')

        #seperate protein model into components
        protein = self.structure.extract('record', 'ATOM', '==')
        backbone = self.structure.extact(f"name CA or name C or name O or name N")
        backbone.tofile('backbone_test.pdb')

        #potentially just do with rotamer library
        #sample water at ideal backbone atoms 
          #for oxygen and nitrogen, place waters where they should be
            #delete based on density (we need info from SWIM as to what this cutoff should be, or should it be a sample of 0.2-1 occ * fcalc oxygen) 
            #if clash with fully occupied protein or ligand remove
            #only store water molecule positions
        #for protein of full or A occ protein
        protein_A = self.structure.extract(f"record ATOM and altloc A or altloc '' ")
        
        #split & sample

        #testing with one residue only
        residues = protein_A.extract(f"resi 1 and chain A")

        #remove if residue type is not a sidechain
        #residues = list(protein_A.residues)
        print(residues)

        for residue in residues: #? Is this going to include backbone atoms?
          if r.type != 'rotamer-residue':
            return
          xmap_reduced = xmap.extract(residue.coor, padding=options.padding) #padding is 10A default currently
          #based on residue type look up positions of water molecules based on WatAA
          #for water in water_residue[residue.resn[0]] #dictionary of water of each residue. 
            #place water molecule 
              # we will need the distance from two atoms 
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
    """Loads files to build a QFitWater job."""

    # Load structure and prepare it
    structure = Structure.fromfile(options.structure).reorder()
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
    options = QFitProteinOptions()
    options.apply_command_args(args)

    # Setup logger
    setup_logging(options=options)
    log_run_info(options, logger)

    # Build a QFitProtein job
    qfit = prepare_qfit_water(options=options)

    # Run the QFitProtein job
    time0 = time.time()
    multiconformer = qfit.run()
    logger.info(f"Total time: {time.time() - time0}s")
