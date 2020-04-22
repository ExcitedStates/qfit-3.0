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
