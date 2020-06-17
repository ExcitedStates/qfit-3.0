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

import numpy as np
import argparse
import logging
import copy
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
    p.add_argument('-occ', "--occ_cutoff", type=float, default=0.10, metavar="<float>",
                   help="Remove conformers with occupancies below occ_cutoff. (default: 0.10)")
    p.add_argument('-rmsd', "--rmsd_cutoff", type=float, default=0.01, metavar="<float>")
    args = p.parse_args()

    return args


def main():
    args = parse_args()
    try:
        os.makedirs(args.directory)
        output_file = os.path.join(args.directory,
                                   args.structure[:-4] + "_norm.pdb")
    except OSError:
        output_file = args.structure[:-4] + "_norm.pdb"

    structure = Structure.fromfile(args.structure)
    # Remove conformers whose occupancy fall below cutoff
    link_data = structure.link_data
    mask = structure.q < args.occ_cutoff
    removed = np.sum(mask)
    data = {}
    for attr in structure.data:
        data[attr] = getattr(structure, attr).copy()[~mask]
    structure = Structure(data).reorder()
    structure.link_data = link_data

    for chain_id in np.unique(structure.chain):
        chain = structure.extract("chain", chain_id, "==")
        for res_id in np.unique(chain.resi):
            residue = chain.extract("resi", res_id, "==")
            # Normalize occupancies and fix altlocs:
            altlocs = np.unique(residue.altloc)
            altlocs = altlocs[altlocs != ""]
            if len(altlocs):
                conf = residue.extract("altloc", altlocs)
                natoms = len(residue.extract("altloc", altlocs[-1]).name)
                factor = natoms / np.sum(conf.q)
                residue._q[conf._selection] *= factor
            else:
                residue._q[residue._selection] = 1.0
                residue._altloc[residue._selection] = ""

    structure.tofile(output_file)
    print(removed)
