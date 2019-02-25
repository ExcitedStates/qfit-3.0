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
from string import ascii_uppercase
logger = logging.getLogger(__name__)

import numpy as np

#from .builders import HierarchicalBuilder
#from .helpers import mkdir_p
#from .validator import Validator
from . import MapScaler, Structure, XMap, _Ligand

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
    p.add_argument("-rb", "--randomize-b", action="store_true", dest="randomize_b",
        help="Randomize B-factors of generated conformers.")
    p.add_argument('-o', '--omit', action="store_true",
            help="Map file is an OMIT map. This affects the scaling procedure of the map.")


    # Map prep options
    p.add_argument("-ns", "--no-scale", action="store_false", dest="scale",
            help="Do not scale density.")
    p.add_argument("-dc", "--density-cutoff", type=float, default=0.3, metavar="<float>",
            help="Densities values below cutoff are set to <density_cutoff_value")
    p.add_argument("-dv", "--density-cutoff-value", type=float, default=-1, metavar="<float>",
            help="Density values below <density-cutoff> are set to this value.")
    p.add_argument("-par", "--phenix-aniso", action="store_true", dest="phenix_aniso",
            help="Use phenix to perform anisotropic refinement of individual sites."
                 "This option creates an OMIT map and uses it as a default.")

    # Sampling options
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
    p.add_argument("-t", "--threshold", type=float, default=0.2, metavar="<float>",
            help="Treshold constraint used during MIQP.")
    p.add_argument("-it", "--intermediate-threshold", type=float, default=0.01, metavar="<float>",
            help="Threshold constraint during intermediate MIQP.")
    p.add_argument("-ic", "--intermediate-cardinality", type=int, default=5, metavar="<int>",
            help="Cardinality constraint used during intermediate MIQP.")


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
    logging_fname = os.path.join(args.directory, 'qfit_ligand.log')
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




    # Extract ligand and rest of structure
    structure = Structure.fromfile(args.structure)
    logger.info("Extracting receptor and ligand from input structure.")

    chainid, resi = args.selection.split(',')
    if ':' in resi:
        resi, icode = resi.split(':')
        residue_id = (int(resi), icode)
    else:
        residue_id = int(resi)
        icode = ''

    # Extract the ligand:
    structure_ligand = structure.extract(f'resi {resi} and chain {chainid}')
    if icode:
        structure_ligand = structure_ligand.extract('icode', icode)
    ligand = _Ligand(structure_ligand.data,
                     structure_ligand._selection)
    if ligand.natoms == 0:
        raise RuntimeError("No atoms were selected for the ligand. Check the "
                           " selection input.")

    # Check which altlocs are present in the ligand. If none, take the
    # A-conformer as default.
    altlocs = sorted(list(set(ligand.altloc)))
    if len(altlocs) > 1:
        try:
            altlocs.remove('')
        except ValueError:
            pass
        for altloc in altlocs[1:]:
            sel_str = f"resi {resi} and chain {chainid} and altloc {altloc}"
            sel_str = f"not ({sel_str})"
#            structure = structure.extract(sel_str)
            ligand = ligand.extract(sel_str)
    ligand.altloc.fill('')
    ligand.q.fill(1)

    sel_str = f"resi {resi} and chain {chainid}"
    sel_str = f"not ({sel_str})"
    receptor = structure.extract(sel_str)
    logger.info("Receptor atoms selected: {natoms}".format(natoms=receptor.natoms))


    xmap = XMap.fromfile(args.map, resolution=args.resolution, label=args.label)
    xmap = xmap.canonical_unit_cell()
    if args.scale:
        # Prepare X-ray map
        scaler = MapScaler(xmap, scattering=args.scattering)
        if args.omit:
            footprint = structure_ligand
        else:
            sel_str = f"resi {resi} and chain {chainid}"
            if icode:
                sel_str += f" and icode {icode}"
            sel_str = f"not ({sel_str})"
            footprint = structure.extract(sel_str)
            footprint = footprint.extract('record', 'ATOM')
        scaler.scale(footprint, radius=1)
        #scaler.cutoff(options.density_cutoff, options.density_cutoff_value)
    xmap = xmap.extract(ligand.coor, padding=5)
    ext = '.ccp4'
    if not np.allclose(xmap.origin, 0):
        ext = '.mrc'
    scaled_fname = os.path.join(args.directory, f'scaled{ext}')
    xmap.tofile(scaled_fname)

    print("Under development!")


    '''
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
    multiconformer.tofile(fname)'''

    m, s = divmod(time.time() - time0, 60)
    logger.info('Time passed: {m:.0f}m {s:.0f}s'.format(m=m, s=s))
    logger.info(time.strftime("%c %Z"))
