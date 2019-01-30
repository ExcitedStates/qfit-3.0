'''
Excited States software: qFit 3.0

Contributors: Saulo H. P. de Oliveira, Gydo van Zundert, and Henry van den Bedem.
Contact: vdbedem@stanford.edu

Copyright (C) 2009-2019 Stanford University
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

This entire text, including the above copyright notice and this permission notice
shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
'''

"""Hierarchically build a multiconformer ligand."""

import argparse
import logging
import os.path
import sys
import time
from itertools import izip
from string import ascii_uppercase
logger = logging.getLogger(__name__)

import numpy as np

from .builders import HierarchicalBuilder
from .helpers import mkdir_p
from .scaler import MapScaler
from .structure import Ligand, Structure
from .validator import Validator
from .volume import Volume


def parse_args():

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("xmap", type=str,
            help="X-ray density map in CCP4 format.")
    p.add_argument("resolution", type=float,
            help="Map resolution in angstrom.")
    p.add_argument("ligand", type=str,
            help="Ligand structure in PDB format. Can also be a whole structure if selection is added with --select option.")
    p.add_argument("-r", "--receptor", type=str, default=None,
            metavar="<file>",
            help="PDB file containing receptor for clash detection.")
    p.add_argument('--selection', default=None, type=str, metavar="<chain,resi>",
            help="Chain and residue id for ligand in main PDB file, e.g. A,105.")
    p.add_argument("-ns", "--no-scale", action="store_true",
            help="Do not scale density.")
    p.add_argument("-dc", "--density-cutoff", type=float, default=0.0, metavar="<float>",
            help="Density value cutoff in sigma of X-ray map. Values below this threshold are set to 0 after scaling to absolute density.")
    p.add_argument("-nb", "--no-build", action="store_true",
            help="Do not build ligand.")
    p.add_argument("-nl", "--no-local", action="store_true",
            help="Do not perform a local search.")
    p.add_argument("-b", "--build-stepsize", type=int, default=1, metavar="<int>",
            help="Number of internal degrees that are sampled/build per iteration.")
    p.add_argument("-s", "--stepsize", type=float, default=1, metavar="<float>",
            help="Stepsize for dihedral angle sampling in degree.")
    p.add_argument("-c", "--cardinality", type=int, default=5, metavar="<int>",
            help="Cardinality constraint used during MIQP.")
    p.add_argument("-t", "--threshold", type=float, default=None, metavar="<float>",
            help="Treshold constraint used during MIQP.")
    p.add_argument("-it", "--intermediate-threshold", type=float, default=0.01, metavar="<float>",
            help="Threshold constraint during intermediate MIQP.")
    p.add_argument("-ic", "--intermediate-cardinality", type=int, default=5, metavar="<int>",
            help="Cardinality constraint used during intermediate MIQP.")
    p.add_argument("-d", "--directory", type=os.path.abspath, default='.', metavar="<dir>",
            help="Directory to store results.")
    #p.add_argument("-p", "--processors", type=int,
    #        default=None, metavar="<int>",
    #        help="Number of threads to use. Currently this only changes the CPLEX/MIQP behaviour.")
    p.add_argument("--debug", action="store_true",
                   help="Write intermediate structures to file for debugging.")
    p.add_argument("-v", "--verbose", action="store_true",
            help="Be verbose.")
    args = p.parse_args()

    # If threshold and cutoff are not defined, use "optimal" values
    if args.threshold is None:
        if args.resolution < 2.00:
            args.threshold = 0.2
        else:
            args.threshold = 0.3

    return args


def main():

    args = parse_args()
    mkdir_p(args.directory)
    time0 = time.time()
    logging_fname = os.path.join(args.directory, 'qfit_ligand.log')
    logging.basicConfig(filename=logging_fname, level=logging.INFO)
    logger.info(' '.join(sys.argv))
    logger.info(time.strftime("%c %Z"))
    if args.verbose:
        console_out = logging.StreamHandler(stream=sys.stdout)
        console_out.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console_out)

    xmap = Volume.fromfile(args.xmap).fill_unit_cell()
    if args.selection is None:
        ligand = Ligand.fromfile(args.ligand)
        if args.receptor is not None:
            receptor = Structure.fromfile(args.receptor).select('e', 'H', '!=')
        else:
            receptor = None
    else:
        # Extract ligand and rest of structure
        structure = Structure.fromfile(args.ligand)
        logger.info("Extracting receptor and ligand from input structure.")
        types = (str, int)
        chain, resi = [t(x) for t, x in izip(types, args.selection.split(','))]
        # Select all ligand conformers
        ligand_selection = structure.select('resi', resi, return_ind=True)
        ligand_selection &= structure.select('chain', chain, return_ind=True)
        ligand = Ligand(structure.data[ligand_selection], structure.coor[ligand_selection])
        if ligand.natoms == 0:
            raise RuntimeError("No atoms were selected for the ligand. Check the selection input.")
        # Check if current ligand already has an alternate conformation. Discard all but one of them.
        altlocs = np.unique(ligand.altloc).tolist()
        naltlocs = len(altlocs)
        if naltlocs > 1 or altlocs[0] != "":
            if "" in altlocs:
                altlocs.remove('""')
                naltlocs -= 1
                logger.info("Ligand contains {naltlocs} alternate conformers.".format(naltlocs=naltlocs))
            altloc_to_use = altlocs[0]
            logger.info("Taking main chain and {altloc} conformer atoms of ligand.".format(altloc=altloc_to_use))
            ligand = ligand.select('altloc', ['', altloc_to_use])
        logger.info("Ligand atoms selected: {natoms}".format(natoms=ligand.natoms))

        receptor_selection = np.logical_not(ligand_selection)
        receptor = Structure(structure.data[receptor_selection],
                             structure.coor[receptor_selection]).select('e', 'H', '!=')
        logger.info("Receptor atoms selected: {natoms}".format(natoms=receptor.natoms))
    # Reset occupancies of ligand
    ligand.altloc.fill('')
    ligand.q.fill(1)

    if not args.no_scale:
        scaler = MapScaler(xmap, mask_radius=1, cutoff=args.density_cutoff)
        scaler(receptor.select('record', 'ATOM'))

    builder = HierarchicalBuilder(
            ligand, xmap, args.resolution, receptor=receptor,
            build=(not args.no_build), build_stepsize=args.build_stepsize,
            stepsize=args.stepsize, local_search=(not args.no_local),
            cardinality=args.intermediate_cardinality,
            threshold=args.intermediate_threshold,
            directory=args.directory, debug=args.debug
    )
    builder()
    fnames = builder.write_results(base='conformer', cutoff=0)

    conformers = builder.get_conformers()
    nconformers = len(conformers)
    if nconformers == 0:
        raise RuntimeError("No conformers were generated or selected. Check whether initial configuration of ligand is severely clashing.")

    validator = Validator(xmap, args.resolution)
    # Order conformers based on rscc
    for fname, conformer in izip(fnames, conformers):
        conformer.rscc = validator.rscc(conformer, rmask=1.5)
        conformer.fname = fname
    conformers_sorted = sorted(conformers, key=lambda conformer: conformer.rscc, reverse=True)
    logger.info("Number of conformers before RSCC filtering: {:d}".format(len(conformers)))
    logger.info("RSCC values:")
    for conformer in conformers_sorted:
        logger.info("{fname}: {rscc:.3f}".format(fname=conformer.fname, rscc=conformer.rscc))
    # Remove conformers with significantly lower rscc
    best_rscc = conformers_sorted[0].rscc
    rscc_cutoff = 0.9 * best_rscc
    conformers = [conformer for conformer in conformers_sorted if conformer.rscc >= rscc_cutoff]
    logger.info("Number of conformers after RSCC filtering: {:d}".format(len(conformers)))

    ## Remove geometrically similar ligands
    #noH = np.logical_not(conformers[0].select('e', 'H', return_ind=True))
    #coor_set = [conformers[0].coor]
    #filtered_conformers = [conformers[0]]
    #for conformer in conformers[1:]:
    #    max_dist = min([np.abs(
    #        np.linalg.norm(conformer.coor[noH] - coor[noH], axis=1).max()
    #        ) for coor in coor_set]
    #    )
    #    if max_dist < 1.5:
    #        continue
    #    coor_set.append(conformer.coor)
    #    filtered_conformers.append(conformer)
    #logger.info("Removing redundant conformers.".format(len(conformers)))
    #conformers = filtered_conformers
    #logger.info("Number of conformers: {:d}".format(len(conformers)))

    iteration = 1
    while True:
        logger.info("Consistency iteration: {}".format(iteration))
        # Use builder class to perform MIQP
        builder._coor_set = [conformer.coor for conformer in conformers]
        builder._convert()
        builder._MIQP(threshold=args.threshold, maxfits=args.cardinality)

        # Check if adding a conformer increasing the cross-correlation
        # sufficiently through the Fisher z transform
        filtered_conformers = []
        for occ, conformer in izip(builder._occupancies, conformers):
            if occ > 0.0001:
                conformer.data['q'].fill(occ)
                filtered_conformers.append(conformer)
        conformers = filtered_conformers
        logger.info("Number of conformers after MIQP: {}".format(len(conformers)))
        conformers[0].zscore = float('inf')
        multiconformer = conformers[0]
        multiconformer.data['altloc'].fill('A')
        nconformers = 1
        filtered_conformers = [conformers[0]]
        for conformer in conformers[1:]:
            conformer.data['altloc'].fill(ascii_uppercase[nconformers])
            new_multiconformer = multiconformer.combine(conformer)
            diff = validator.fisher_z_difference(
                multiconformer, new_multiconformer, rmask=1.5, simple=True
            )
            if diff < 0.1:
                continue
            multiconformer = new_multiconformer
            conformer.zscore = diff
            filtered_conformers.append(conformer)
            nconformers += 1
        logger.info("Number of conformers after Fisher zscore filtering: {}".format(len(filtered_conformers)))
        if len(filtered_conformers) == len(conformers):
            conformers = filtered_conformers
            break
        conformers = filtered_conformers
        iteration += 1
    if nconformers == 1:
        logger.info("No alternate conformer found.")
        multiconformer.data['altloc'].fill('')
    else:
        logger.info("Number of alternate conformers found: {}".format(len(conformers)))
        logger.info("Fisher z scores:")
        for conformer in conformers[1:]:
            logger.info("{altloc}: {score:.2f}".format(altloc=conformer.altloc[0], score=conformer.zscore))

    fname = os.path.join(args.directory, 'multiconformer.pdb')
    multiconformer.tofile(fname)

    m, s = divmod(time.time() - time0, 60)
    logger.info('Time passed: {m:.0f}m {s:.0f}s'.format(m=m, s=s))
    logger.info(time.strftime("%c %Z"))
