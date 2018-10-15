"""Automatically build a multiconformer residue"""

import argparse
import logging
import os
import sys
import time
from string import ascii_uppercase
logger = logging.getLogger(__name__)

import numpy as np

from . import MapScaler, Structure, XMap, QFitRotamericResidue, QFitRotamericResidueOptions
from .structure import residue_type


def parse_args():

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("map", type=str,
            help="Density map in CCP4 or MRC format, or an MTZ file "
                 "containing reflections and phases. For MTZ files "
                 "use the --label options to specify columns to read.")
    p.add_argument("structure", type=str,
            help="PDB-file containing structure.")
    p.add_argument('selection', type=str,
            help="Chain, residue id, and optionally insertion code for residue in structure, e.g. A,105, or A,105:A.")

    # Map input options
    p.add_argument("-l", "--label", default="FWT,PHWT", metavar="<F,PHI>",
            help="MTZ column labels to build density.")
    p.add_argument('-r', "--resolution", type=float, default=None, metavar="<float>",
            help="Map resolution in angstrom. Only use when providing CCP4 map files.")
    p.add_argument("-m", "--resolution_min", type=float, default=None, metavar="<float>",
            help="Lower resolution bound in angstrom. Only use when providing CCP4 map files.")
    p.add_argument("-z", "--scattering", choices=["xray", "electron"], default="xray",
            help="Scattering type.")
    p.add_argument('-o', '--omit', action="store_true",
            help="Map file is an OMIT map. This affects the scaling procedure of the map.")

    # Map prep options
    p.add_argument("-ns", "--no-scale", action="store_false", dest="scale",
            help="Do not scale density.")
    p.add_argument("-dc", "--density-cutoff", type=float, default=0.3, metavar="<float>",
            help="Densities values below cutoff are set to <density_cutoff_value")
    p.add_argument("-dv", "--density-cutoff-value", type=float, default=-1, metavar="<float>",
            help="Density values below <density-cutoff> are set to this value.")

    # Sampling options
    p.add_argument('-bb', "--backbone", dest="sample_backbone", action="store_true",
            help="Sample backbone using inverse kinematics.")
    p.add_argument('-sa', "--sample-angle", dest="sample_angle", action="store_true",
            help="Sample N-CA-CB angle.")
    p.add_argument("-b", "--dofs-per-iteration", type=int, default=2, metavar="<int>",
            help="Number of internal degrees that are sampled/build per iteration.")
    p.add_argument("-s", "--dofs-stepsize", type=float, default=6, metavar="<float>",
            help="Stepsize for dihedral angle sampling in degree.")
    p.add_argument("-rn", "--rotamer-neighborhood", type=float,
            default=40, metavar="<float>",
            help="Neighborhood of rotamer to sample in degree.")
    p.add_argument("--no-remove-conformers-below-cutoff", action="store_false",
                   dest="remove_conformers_below_cutoff",
            help=("Remove conformers during sampling that have atoms that have "
                  "no density support for, i.e. atoms are positioned at density "
                  "values below cutoff value."))
    p.add_argument("-bs", "--bulk_solvent_level", default=0.3, type=float, metavar="<float>",
            help="Bulk solvent level in absolute values.")
    p.add_argument("-c", "--cardinality", type=int, default=5, metavar="<int>",
            help="Cardinality constraint used during MIQP.")
    p.add_argument("-t", "--threshold", type=float, default=0.3, metavar="<float>",
            help="Treshold constraint used during MIQP.")

    # Output options
    p.add_argument("-d", "--directory", type=os.path.abspath, default='.', metavar="<dir>",
            help="Directory to store results.")
    p.add_argument("--debug", action="store_true",
            help="Write intermediate structures to file for debugging.")
    p.add_argument("-v", "--verbose", action="store_true",
            help="Be verbose.")
    args = p.parse_args()

    return args


def main():

    args = parse_args()
    try:
        os.makedirs(args.directory)
    except OSError:
        pass
    time0 = time.time()

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
    structure = structure.extract('e', 'H', '!=')
    chainid, resi = args.selection.split(',')
    if ':' in resi:
        resi, icode = resi.split(':')
        residue_id = (int(resi), icode)
    else:
        residue_id = int(resi)
        icode = ''

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
        altloc = altlocs[0]
    else:
        altloc = 'A'
    structure = structure.extract('altloc', ('', altloc))

    residue_name = residue.resn[0]
    logger.info(f"Residue: {residue_name} {chainid}_{resi}{icode}")

    options = QFitRotamericResidueOptions()
    options.apply_command_args(args)

    xmap = XMap.fromfile(args.map, resolution=args.resolution, label=args.label)
    xmap = xmap.canonical_unit_cell()
    if args.scale:
        # Prepare X-ray map
        scaler = MapScaler(xmap, scattering=options.scattering)
        if args.omit:
            footprint = structure_resi
        else:
            sel_str = f"resi {resi} and chain {chainid}"
            if icode:
                sel_str += f" and icode {icode}"
            sel_str = f"not ({sel_str})"
            footprint = structure.extract(sel_str)
            footprint = footprint.extract('record', 'ATOM')
        scaler.scale(footprint, radius=1)
        #scaler.cutoff(options.density_cutoff, options.density_cutoff_value)
    xmap = xmap.extract(residue.coor, padding=5)
    ext = '.ccp4'
    if not np.allclose(xmap.origin, 0):
        ext = '.mrc'
    scaled_fname = os.path.join(args.directory, f'scaled{ext}')
    xmap.tofile(scaled_fname)

    qfit = QFitRotamericResidue(residue, structure, xmap, options)
    qfit.run()
    qfit.write_maps()
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
