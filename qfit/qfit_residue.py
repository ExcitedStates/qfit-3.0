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


def parse_args():

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("xmap", type=str,
            help="X-ray density map in CCP4 format.")
    p.add_argument("resolution", type=float,
            help="Map resolution in angstrom.")
    p.add_argument("structure", type=str,
            help="PDB-file containing structure.")
    p.add_argument('--selection', default=None, type=str,
            help="Chain, residue id, and optionally insertion code for residue in structure, e.g. A,105, or A,105:A.")
    p.add_argument("-ns", "--no-scale", action="store_true",
            help="Do not scale density.")
    p.add_argument("-b", "--dofs-per-iteration", type=int, default=1, metavar="<int>",
            help="Number of internal degrees that are sampled/build per iteration.")
    p.add_argument("-s", "--dofs-stepsize", type=float, default=5, metavar="<float>",
            help="Stepsize for dihedral angle sampling in degree.")
    p.add_argument("-m", "--resolution_min", type=float, default=None, metavar="<float>",
            help="Lower resolution bound in angstrom.")
    p.add_argument("-z", "--scattering", choices=["xray", "electron"], default="xray",
            help="Scattering type.")
    p.add_argument("-r", "--rotamer-neighborhood", type=float, default=40, metavar="<float>",
            help="Neighborhood of rotamer to sample in degree.")
    p.add_argument("-c", "--cardinality", type=int, default=5, metavar="<int>",
            help="Cardinality constraint used during MIQP.")
    p.add_argument("-t", "--threshold", type=float, default=None, metavar="<float>",
            help="Treshold constraint used during MIQP.")
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
    structure = Structure.fromfile(args.structure)
    chainid, resi = args.selection.split(',')
    selection_string = f'resi {resi} and chain {chainid}'
    if ':' in resi:
        resi, icode = resi.split(':')
        residue_id = (int(resi), icode)
    else:
        residue_id = int(resi)

    chain = structure[chainid]
    conformer = chain.conformers[0]
    residue = conformer[residue_id]
    residue.altloc = ''

    logger.info(f"Residue: {residue.resn[0]}")
    # Prepare X-ray map
    xmap = XMap.fromfile(args.xmap).canonical_unit_cell()

    options = QFitRotamericResidueOptions()
    options.apply_command_args(args)

    if not args.no_scale:
        scaler = MapScaler(
            xmap, mask_radius=1, scattering=options.scattering, subtract=True, cutoff=0)
        footprint = structure.extract('resi', residue_id, '!=')
        scaler(footprint)

    qfit = QFitRotamericResidue(residue, xmap, options)
    qfit()
    conformers = qfit.get_conformers()
    for n, conformer in enumerate(conformers):
        fname = os.path.join(options.directory, f'conformer_{n}.pdb')
        conformer.tofile(fname)

    passed = time.time() - time0
    print(f"Time passed: {passed}s")
