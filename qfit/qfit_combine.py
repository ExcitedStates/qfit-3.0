"""Combine output structures optionally together with receptor."""

from argparse import ArgumentParser
import string
from itertools import izip

import numpy as np

from .structure import Structure


def parse_args():

    p = ArgumentParser(description=__doc__)
    p.add_argument("ligands", nargs="+", type=str,
            help="Ligand structures to be combined in multiconformer model.")
    p.add_argument("-r", "--receptor", type=str,
            help="Receptor.")
    p.add_argument("-o", "--output", type=str, default='multiconformer.pdb',
            help="Name of output file.")
    p.add_argument("--remove", action="store_true",
            help="First remove present ligand, based on input ligands chain and residue id.")

    args = p.parse_args()
    return args


def main():

    args = parse_args()

    
    if len(args.ligands) == 1:
        multiconf = Structure.fromfile(args.ligands[0])
    else:
        for altloc, fname in izip(string.ascii_uppercase, args.ligands):
            l = Structure.fromfile(fname)
            l.altloc[:] = altloc
            try:
                multiconf = multiconf.combine(l)
            except:
                multiconf = l

    if args.receptor is not None:
        receptor = Structure.fromfile(args.receptor)
        if args.remove:
            chain = multiconf.data['chain'][0]
            resi = multiconf.data['resi'][0]
            selection = receptor.select('resi', resi, return_ind=True)
            selection &= receptor.select('chain', chain, return_ind=True)
            selection = np.logical_not(selection)
            receptor = Structure(receptor.data[selection], receptor.coor[selection])
        multiconf = receptor.combine(multiconf)

    multiconf.tofile(args.output)
