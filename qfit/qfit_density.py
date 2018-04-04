from __future__ import division

import argparse
import os

from .volume import Volume
from .structure import Structure
from .transformer import Transformer

def parse_args():

    p = argparse.ArgumentParser("Transform a structure into a density.")

    p.add_argument("pdb_file", type=str,
            help="PDB file containing structure.")
    p.add_argument('xmap', type=str,
            help="CCP4 density with P1 symmetry.")
    p.add_argument('resolution', type=float,
            help="Resolution of map in angstrom.")
    p.add_argument("-s", "--simple", action="store_true",
            help="Produce simple density.")
    p.add_argument("-o", "--output", type=str, default=None,
            help="Name of output density.")

    args = p.parse_args()

    if args.output is None:
        args.output = os.path.splitext(args.pdb_file)[0] + '.ccp4'
    return args


def main():
    args = parse_args()

    structure = Structure.fromfile(args.pdb_file)
    xmap = Volume.fromfile(args.xmap).fill_unit_cell()
    resolution = args.resolution

    out = Volume.zeros_like(xmap)
    smax = 0.5 / resolution
    transformer = Transformer(structure, out, simple=args.simple)
    transformer.mask(1)
    out.tofile('mask.ccp4')
    transformer.reset()
    transformer.density()
    out.tofile(args.output)
