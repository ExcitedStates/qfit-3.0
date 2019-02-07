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

import argparse
import os

from . import XMap
from . import Structure
from .transformer import FFTTransformer, Transformer

def parse_args():

    p = argparse.ArgumentParser("Transform a structure into a density.")

    p.add_argument("pdb_file", type=str,
            help="PDB file containing structure.")
    p.add_argument('xmap', type=str,
            help="")
    p.add_argument('-r', '--resolution', type=float,
            help="Resolution of map in angstrom.")
    p.add_argument('-nf', '--no-fft', action='store_true',
            help="No FFT density map creation.")
    p.add_argument("-o", "--output", type=str, default=None,
            help="Name of output density.")

    args = p.parse_args()

    if args.output is None:
        args.output = os.path.splitext(args.pdb_file)[0] + '.ccp4'
    return args


def main():
    args = parse_args()

    structure = Structure.fromfile(args.pdb_file)
    xmap = XMap.fromfile(args.xmap)
    resolution = args.resolution
    out = XMap.zeros_like(xmap)

    if hasattr(xmap, 'hkl') and not args.no_fft:
        transformer = FFTTransformer(structure, out)
    else:
        if args.resolution is not None:
            smax = 0.5 / resolution
            simple = False
        else:
            smax = None
            simple = True
        transformer = Transformer(structure, out, smax=smax, simple=simple)
    transformer.density()
    out.tofile(args.output)
