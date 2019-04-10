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
import os
import sys
import time
from string import ascii_uppercase
logger = logging.getLogger(__name__)

import numpy as np

from . import MapScaler, Structure, XMap, Covalent_Ligand
from . import QFitCovalentLigand, QFitCovalentLigandOptions

os.environ["OMP_NUM_THREADS"] = "1"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("map", type=str,
            help="Density map in CCP4 or MRC format, or an MTZ file "
                 "containing reflections and phases. For MTZ files "
                 "use the --label options to specify columns to read.")
    p.add_argument("structure", type=str,
            help="PDB-file containing structure.")
    p.add_argument('-cif', "--cif_file", type=str, default=None,
            help="CIF file describing the ligand")
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
    p.add_argument("-nosub", "--no-subtract", action="store_false", dest="subtract",
            help="Do not subtract Fcalc of the neighboring residues when running qFit.")
    p.add_argument("-pad", "--padding", type=float, default=8.0, metavar="<float>",
            help="Padding size for map creation.")

    # Sampling options
    p.add_argument('-bb', "--no-backbone", dest="sample_backbone", action="store_false",
            help="Do not sample backbone using inverse kinematics.")
    p.add_argument('-bbs', "--backbone-step", dest="sample_backbone_step",
            type=float, default=0.1, metavar="<float>",
            help="Backbone sampling step (default = 0.1)")
    p.add_argument('-bba', "--backbone-amplitude", dest="sample_backbone_amplitude",
            type=float, default=0.3, metavar="<float>",
           help="Backbone sampling amplitude (default = 0.3)")
    p.add_argument('-sa', "--no-sample-angle", dest="sample_angle", action="store_false",
            help="Do not sample N-CA-CB angle.")
    p.add_argument('-sas', "--sample-angle-step", dest="sample_angle_step",
            type=float, default=3.75, metavar="<float>",
            help="Bond angle sampling step (default = 3.75)")
    p.add_argument('-sar', "--sample-angle-range", dest="sample_angle_range",
            type=float, default=7.5, metavar="<float>",
            help="Bond angle sampling range (default = 7.5)."
                 "Sampling is carried out in the [-x,x] range, where x is"
                 " determined by this sampling parameter.")
    p.add_argument("-b", "--dofs-per-iteration", type=int, default=2, metavar="<int>",
            help="Number of internal degrees that are sampled/build per iteration.")
    p.add_argument("-s", "--dofs-stepsize", type=float, default=10, metavar="<float>",
            help="Stepsize for dihedral angle sampling in degree.")
    p.add_argument("-rn", "--rotamer-neighborhood", type=float,
                   default=80, metavar="<float>",
                   help="Neighborhood of rotamer to sample in degree.")
    p.add_argument("-nl", "--no-ligand",
                   dest="sample_ligand", action="store_false",
                   help="Disable ligand sampling.")
    p.add_argument("--remove-conformers-below-cutoff", action="store_true",
                   dest="remove_conformers_below_cutoff",
            help=("Remove conformers during sampling that have atoms that have "
                  "no density support for, i.e. atoms are positioned at density "
                  "values below cutoff value."))
    p.add_argument('-cf', "--clash_scaling_factor", type=float, default=0.75, metavar="<float>",
            help="Set clash scaling factor. Default = 0.75")
    p.add_argument('-ec', "--external_clash", dest="external_clash", action="store_true",
            help="Enable external clash detection during sampling.")
    p.add_argument("-bs", "--bulk_solvent_level", default=0.3, type=float, metavar="<float>",
            help="Bulk solvent level in absolute values.")
    p.add_argument("-c", "--cardinality", type=int, default=5, metavar="<int>",
            help="Cardinality constraint used during MIQP.")
    p.add_argument("-t", "--threshold", type=float, default=0.2, metavar="<float>",
            help="Treshold constraint used during MIQP.")
    p.add_argument("-hy", "--hydro", dest="hydro", action="store_true",
            help="Include hydrogens during calculations.")
    p.add_argument("-M", "--miosqp", dest="cplex", action="store_false",
            help="Use MIOSQP instead of CPLEX for the QP/MIQP calculations.")
    p.add_argument("-T", "--no-threshold-selection", dest="bic_threshold", action="store_false",
            help="Do not use BIC to select the most parsimonious MIQP threshold")

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
    logging_fname = os.path.join(args.directory, 'qfit_covalent_ligand.log')
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

    # Load structure and prepare it
    structure = Structure.fromfile(args.structure).reorder()
    if not args.hydro:
        structure = structure.extract('e', 'H', '!=')

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
    if args.cif_file:
        covalent_ligand = Covalent_Ligand(structure_ligand.data,
                                          structure_ligand._selection,
                                          link_data=structure_ligand.link_data,
                                          cif_file=args.cif_file)
    else:
        covalent_ligand = Covalent_Ligand(structure_ligand.data,
                                          structure_ligand._selection,
                                          link_data=structure_ligand.link_data)
    if covalent_ligand.natoms == 0:
        raise RuntimeError("No atoms were selected for the ligand. Check the "
                           "selection input.")

    # Select all ligand conformers:
    # Check which altlocs are present in the ligand. If none, take the
    # A-conformer as default.
    altlocs = sorted(list(set(covalent_ligand.altloc)))
    if len(altlocs) > 1:
        try:
            altlocs.remove('')
        except ValueError:
            pass
        for altloc in altlocs[1:]:
            sel_str = f"resi {resi} and chain {chainid} and altloc {altloc}"
            sel_str = f"not ({sel_str})"
            covalent_ligand = covalent_ligand.extract(sel_str)
    covalent_ligand.altloc.fill('')
    covalent_ligand.q.fill(1)

    sel_str = f"resi {resi} and chain {chainid}"
    sel_str = f"not ({sel_str})"
    receptor = structure.extract(sel_str)
    logger.info("Receptor atoms selected: {natoms}".format(natoms=receptor.natoms))

    options = QFitCovalentLigandOptions()
    options.apply_command_args(args)

    # Load and process the electron density map:
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
        radius = 1.5
        reso = None
        if xmap.resolution.high is not None:
            reso = xmap.resolution.high
        elif options.resolution is not None:
            reso = options.resolution
        if reso is not None:
            radius = 0.5 + reso / 3.0
        scaler.scale(footprint, radius=radius)
    xmap = xmap.extract(covalent_ligand.coor, padding=5)
    ext = '.ccp4'
    if not np.allclose(xmap.origin, 0):
        ext = '.mrc'
    scaled_fname = os.path.join(args.directory, f'scaled{ext}')
    xmap.tofile(scaled_fname)

    qfit = QFitCovalentLigand(covalent_ligand, receptor, xmap, options)
    qfit.run()
    conformers = qfit.get_conformers_covalent()
    nconformers = len(conformers)
    altloc = ''
    for n, conformer in enumerate(conformers, start=0):
        if nconformers > 1:
            altloc = ascii_uppercase[n]
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
