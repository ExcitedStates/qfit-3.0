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

"""Automatically build a multiconformer residue"""
import numpy as np
import argparse
import logging
import os
import sys
import time
from string import ascii_uppercase
from . import Structure
from .structure import residue_type


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("structure", type=str,
                   help="PDB-file containing structure.")

    # Output options
    p.add_argument("-d", "--directory", type=os.path.abspath, default='.',
                   metavar="<dir>", help="Directory to store results.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Be verbose.")
    p.add_argument('-occ', "--occ_cutoff", type=float, default=0.01, metavar="<float>",
            help="Remove conformers with occupancies below occ_cutoff. Default = 0.01")
    args = p.parse_args()

    return args


def main():
    args = parse_args()
    try:
        os.makedirs(args.directory)
        output_file = os.path.join(args.directory,
                                 args.structure[:-4]+"_norm.pdb")
    except OSError:
        output_file = args.structure[:-4]+"_norm.pdb"

    structure = Structure.fromfile(args.structure).reorder()
    to_remove = []

    for rg in structure.extract('record',"ATOM").residue_groups:
        total_occupancy = 0.0
        altlocs = list(set(rg.altloc))
        # If the alternate conformer has collapsed atoms:
        if '' in altlocs and len(altlocs) > 1:
            mask = ( rg.altloc == '' )
            rg.q[mask] = 1.0
            altlocs.remove('')

        # If only a single conformer
        if len(altlocs) == 1:
            rg.q = 1.0
        else:
            for altloc in altlocs:
                mask = (rg.altloc == altloc)
                if rg.q[mask][0] < args.occ_cutoff:
                    to_remove.append([rg.resi[mask][0], rg.chain[mask][0],
                                      rg.altloc[mask][0]])
                else:
                    total_occupancy += rg.q[mask][0]
            if total_occupancy > 0.01:
                rg.q = np.divide(rg.q,total_occupancy)
                rg.q = np.clip(rg.q, 0,1)
    # Remove conformers with 0 occupancy
    for conformer in to_remove:
        res, chain, altloc = conformer
        sel_str = f"resi {res} and chain {chain} and altloc {altloc}"
        sel_str = f"not ({sel_str})"
        structure = structure.extract(sel_str)

    structure.tofile(output_file)
