"""Automatically build a multiconformer residue."""

import argparse
from .custom_argparsers import ToggleActionFlag, CustomHelpFormatter, ValidateMapFileArgument, ValidateStructureFileArgument
import logging
import os
import sys
import time
import numpy as np
from string import ascii_uppercase
from .logtools import setup_logging, log_run_info
from . import MapScaler, Structure, XMap
from .qfit import QFitOptions
from . import QFitRotamericResidue
from .structure import residue_type

logger = logging.getLogger(__name__)
os.environ["OMP_NUM_THREADS"] = "1"

def build_argparser():
    p = argparse.ArgumentParser(formatter_class=CustomHelpFormatter,
                                description=__doc__)
    p.add_argument("map",
                   help="Density map in CCP4 or MRC format, or an MTZ file "
                        "containing reflections and phases. For MTZ files "
                        "use the --label options to specify columns to read.", type=str, action=ValidateMapFileArgument)
    p.add_argument("structure",
                   help="PDB-file containing structure.", type=str, action=ValidateStructureFileArgument)
    p.add_argument('selection', type=str,
                   help="Chain, residue id, and optionally insertion code for "
                        "residue in structure, e.g. A,105, or A,105:A.")

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
    p.add_argument("-rb", "--randomize-b", action="store_true", dest="randomize_b",
                   help="Randomize B-factors of generated conformers")
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
    p.add_argument("-par", "--phenix-aniso", action="store_true", dest="phenix_aniso",
                   help="Use phenix to perform anisotropic refinement of individual sites."
                        "This option creates an OMIT map and uses it as a default.")

    # Sampling options
    p.add_argument("--backbone", action=ToggleActionFlag, dest="sample_backbone", default=True,
                   help="Sample backbone using inverse kinematics")
    p.add_argument('-bbs', "--backbone-step", default=0.1, dest="sample_backbone_step",
                   metavar="<float>", type=float,
                   help="Stepsize for the amplitude of backbone sampling (Å)")
    p.add_argument('-bba', "--backbone-amplitude", default=0.3, dest="sample_backbone_amplitude",
                   metavar="<float>", type=float,
                   help="Maximum backbone amplitude (Å)")
    p.add_argument('-bbv', "--backbone-sigma", default=0.125, dest="sample_backbone_sigma",
                   metavar="<float>", type=float,
                   help="Backbone random-sampling displacement (Å)")
    p.add_argument("--sample-angle", action=ToggleActionFlag, dest="sample_angle", default=True,
                   help="Sample CA-CB-CG angle for aromatic F/H/W/Y residues")
    p.add_argument('-sas', "--sample-angle-step", default=3.75, dest="sample_angle_step",
                   metavar="<float>", type=float,
                   help="CA-CB-CG bond angle sampling step in degrees")
    p.add_argument('-sar', "--sample-angle-range", default=7.5, dest="sample_angle_range",
                   metavar="<float>", type=float,
                   help="CA-CB-CG bond angle sampling range in degrees [-x,x]")
    p.add_argument("--sample-rotamers", action=ToggleActionFlag, dest="sample_rotamers", default=True,
                   help="Sample sidechain rotamers")
    p.add_argument("-b", "--dofs-per-iteration", default=2,
                   metavar="<int>", type=int,
                   help="Number of internal degrees that are sampled/built per iteration")
    p.add_argument("-s", "--dihedral-stepsize", default=10,
                   metavar="<float>", type=float,
                   help="Stepsize for dihedral angle sampling in degrees")
    p.add_argument("-rn", "--rotamer-neighborhood", default=60,
                   metavar="<float>", type=float,
                   help="Chi dihedral-angle sampling range around each rotamer in degrees [-x,x]")
    p.add_argument("--remove-conformers-below-cutoff", action="store_true",
                   dest="remove_conformers_below_cutoff",
                   help=("Remove conformers during sampling that have atoms "
                         "with no density support, i.e. atoms are positioned "
                         "at density values below <density-cutoff>"))
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
    p.add_argument('-rmsd', "--rmsd-cutoff", default=0.01,
                   metavar="<float>", type=float,
                   help="RMSD cutoff for removal of identical conformers")
    p.add_argument("--threshold-selection", dest="bic_threshold", action=ToggleActionFlag, default=True,
                   help="Use BIC to select the most parsimonious MIQP threshold")

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
    p.add_argument("--write_intermediate_conformers", action="store_true",
                   help="Write intermediate structures to file (useful with debugging)")

    return p


def main():
    p = build_argparser()
    args = p.parse_args(args=None)
    try:
        os.makedirs(args.directory)
    except OSError:
        pass
    time0 = time.time()

    # Apply the arguments to options
    options = QFitOptions()
    options.apply_command_args(args)

    # Setup logger
    setup_logging(options=options)
    log_run_info(options, logger)

    #Skip over if everything is completed
    #try:
    if os.path.isfile(args.directory + '/multiconformer_residue.pdb'):
        print('This residue has completed')
        exit()
    else:
        print('Beginning qfit_residue')
     
    # Setup logger
    logging_fname = os.path.join(args.directory, 'qfit_residue.log')
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(filename=logging_fname, level=level)
    logger.info(' '.join(sys.argv))
    logger.info(time.strftime("%c %Z"))
    if args.verbose:
        console_out = logging.StreamHandler(stream=sys.stdout)
        console_out.setLevel(level)
        logging.getLogger('').addHandler(console_out)

    # Extract residue and prepare it
    structure = Structure.fromfile(args.structure).reorder()
    if not args.hydro:
        structure = structure.extract('e', 'H', '!=')
    chainid, resi = args.selection.split(',')
    if ':' in resi:
        resi, icode = resi.split(':')
        residue_id = (int(resi), icode)
    elif '_' in resi:
        resi, icode = resi.split('_')
        residue_id = (int(resi), icode)
    else:
        residue_id = int(resi)
        icode = ''

    # Extract the residue:
    structure_resi = structure.extract(f'resi {resi} and chain {chainid}')
    if icode:
        structure_resi = structure_resi.extract('icode', icode)

    chain = structure_resi[chainid]
    conformer = chain.conformers[0]
    residue = conformer[residue_id]
    rtype = residue_type(residue)
    if rtype != 'rotamer-residue':
        logger.info("Residue has no known rotamers. Stopping qfit_residue.")
        sys.exit()
    # Check which altlocs are present in the residue. If none, take the
    # A-conformer as default.
    altlocs = sorted(list(set(residue.altloc)))
    if len(altlocs) > 1:
        try:
            altlocs.remove('')
        except ValueError:
            pass

        # If more than 1 conformer were included, we want to select the
        # most complete conformer. If more than one conformers are complete,
        # Select the one with the highest occupancy:
        longest_conf = 0
        best_q = -1
        for i, altloc in enumerate(altlocs):
            conformer = structure_resi.extract('altloc', ('',altloc))
            if len(conformer.name) > longest_conf:
                idx = i
                longest_conf = len(conformer.name)
            elif len(conformer.name) == longest_conf:
                if conformer.q[0] > best_q:
                    idx = i
                    best_q = conformer.q[0]
        # Delete all the unwanted conformers:
        for altloc in altlocs:
            if altloc != altlocs[idx]:
                sel_str = f"resi {resi} and chain {chainid} and altloc {altloc}"
                sel_str = f"not ({sel_str})"
                structure = structure.extract(sel_str)
    residue_name = residue.resn[0]
    logger.info(f"Residue: {residue_name} {chainid}_{resi}{icode}")

    xmap = XMap.fromfile(args.map, resolution=args.resolution, label=args.label)
    xmap = xmap.canonical_unit_cell()
    if args.scale:
        # Prepare X-ray map
        scaler = MapScaler(xmap, scattering=options.scattering)
        if args.omit:
            footprint = structure_resi
        else:
            footprint = structure
        radius = 1.5
        reso = None
        if xmap.resolution.high is not None:
            reso = xmap.resolution.high
        elif options.resolution is not None:
            reso = options.resolution
        if reso is not None:
            radius = 0.5 + reso / 3.0
        scaler.scale(footprint, radius=args.scale_rmask*radius)
    xmap = xmap.extract(residue.coor, padding=args.padding)
    ext = '.ccp4'
    if not np.allclose(xmap.origin, 0):
        ext = '.mrc'
    scaled_fname = os.path.join(args.directory, f'scaled{ext}')
    xmap.tofile(scaled_fname)
    qfit = QFitRotamericResidue(residue, structure, xmap, options)
    qfit.run()
    # qfit.write_maps()
    conformers = qfit.get_conformers()
    nconformers = len(conformers)
    altloc = ''
    for n, conformer in enumerate(conformers, start=0):
        if nconformers > 1:
            altloc = ascii_uppercase[n]
        #skip = False
        #for conf in conformers[:n]:
        #    print("Checking RMSD")
        #    if conformer.rmsd(conf) < 0.2:
        #        skip = True
        #        print("Skipping")
        #        break
        #if skip:
        #    continue
        conformer.altloc = ''
        fname = os.path.join(options.directory, f'conformer_{n}.pdb')
        conformer.tofile(fname)
        conformer.altloc = altloc
        try:
            multiconformer = multiconformer.combine(conformer)
        except Exception:
            multiconformer = Structure.fromstructurelike(conformer.copy())
    fname = os.path.join(options.directory, f'multiconformer_{chainid}_{resi}.pdb')
    if icode:
        fname = os.path.join(options.directory, f'multiconformer_{chainid}_{resi}_{icode}.pdb')
    multiconformer.tofile(fname)

    passed = time.time() - time0
    logger.info(f"Time passed: {passed}s")
