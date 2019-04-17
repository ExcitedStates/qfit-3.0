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
    p.add_argument("restraint", type=str,
                   help="EFF file containing the restraints.")
    p.add_argument('selection1', type=str,
                    help="Chain, residue id, and optionally insertion code for residue in structure, e.g. A,105, or A,105:A.")
    p.add_argument('selection2', type=str,
                    help="Chain, residue id, and optionally insertion code for residue in structure, e.g. A,105, or A,105:A.")


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

    chainid1, resi1 = args.selection1.split(',')
    chainid2, resi2 = args.selection2.split(',')

    structure_res1 = structure.extract(f'resi {resi1} and chain {chainid1}')
    structure_res2 = structure.extract(f'resi {resi2} and chain {chainid2}')

    altlocs1 = list(set(structure_res1.altloc))
    altlocs2 = list(set(structure_res2.altloc))
    altlocs1.sort()
    altlocs2.sort()

    with open(args.restraint,"r") as restraint_file:
        for line in restraint_file.readlines():
            if "{" in line or "}" in line:
                print(line.strip())
            else:
                fields = line.split()
                if "atom_selection_1" in fields:
                    chainid3 = fields[3]
                    resi3 = fields[6]
                    name3 = fields[9]
                elif "atom_selection_2" in fields:
                    chainid4 = fields[3]
                    resi4 = fields[6]
                    name4 = fields[9]
                elif "distance_ideal" in fields:
                    dist = fields[2]
                elif "sigma" in fields:
                    sigma = fields[2]

                    # Decide if this is the case we care about:
                    if (resi3 == resi1 and chainid3 == chainid1 and resi4 == resi2 and chainid4 == chainid2):
                        first = True
                        for alt1, alt2 in zip(altlocs1, altlocs2):
                                if not first:
                                     print("   }\n   bond {")
                                print(f"     atom_selection_1 = chain {chainid3} and resseq {resi3} and name {name3} and altloc {alt1}")
                                print(f"     atom_selection_2 = chain {chainid4} and resseq {resi4} and name {name4} and altloc {alt2}")
                                print(f"     distance_ideal = {dist:.1}")
                                print(f"     sigma = {sigma:.1}")
                                first = False
                    else:
                        print(f"     atom_selection_1 = chain {chainid3} and resseq {resi3} and name {name3}")
                        print(f"     atom_selection_2 = chain {chainid4} and resseq {resi4} and name {name4}")
                        print(f"     distance_ideal = {dist:.1}")
                        print(f"     sigma = {sigma:.1}")
