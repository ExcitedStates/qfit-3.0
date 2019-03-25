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
    p.add_argument("-par", "--phenix-aniso", action="store_true", dest="phenix_aniso",
            help="Use phenix to perform anisotropic refinement of individual sites."
                 "This option creates an OMIT map and uses it as a default.")

    # Sampling options
    p.add_argument('-bb', "--backbone", dest="sample_backbone", action="store_true",
            help="Sample backbone using inverse kinematics.")
    p.add_argument('-bbs', "--backbone-step", dest="sample_backbone_step",
            type=float, default=0.1, metavar="<float>",
            help="Sample N-CA-CB angle.")
    p.add_argument('-bba', "--backbone-amplitude", dest="sample_backbone_amplitude",
            type=float, default=0.3, metavar="<float>",
           help="Sample N-CA-CB angle.")
    p.add_argument('-sa', "--sample-angle", dest="sample_angle", action="store_true",
            help="Sample N-CA-CB angle.")
    p.add_argument('-sas', "--sample-angle-step", dest="sample_angle_step",
            type=float, default=3.75, metavar="<float>",
            help="Sample N-CA-CB angle.")
    p.add_argument('-sar', "--sample-angle-range", dest="sample_angle_range",
            type=float, default=7.5, metavar="<float>",
            help="Sample N-CA-CB angle.")
    p.add_argument("-hy", "--hydro", dest="hydro", action="store_true",
            help="Include hydrogens during calculations.")

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
#            structure = structure.extract(sel_str)
            covalent_ligand = covalent_ligand.extract(sel_str)
    covalent_ligand.altloc.fill('')
    covalent_ligand.q.fill(1)

    sel_str = f"resi {resi} and chain {chainid}"
    sel_str = f"not ({sel_str})"
    receptor = structure.extract(sel_str)

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
        scaler.scale(footprint, radius=1)
    xmap = xmap.extract(covalent_ligand.coor, padding=5)
    ext = '.ccp4'
    if not np.allclose(xmap.origin, 0):
        ext = '.mrc'
    scaled_fname = os.path.join(args.directory, f'scaled{ext}')
    xmap.tofile(scaled_fname)

    qfit = QFitCovalentLigand(covalent_ligand, receptor, xmap, options)
    qfit.run()
