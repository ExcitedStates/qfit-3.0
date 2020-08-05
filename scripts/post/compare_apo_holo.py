#!/usr/bin/env python

import numpy as np
import argparse
import logging
import os
import sys
import time
from string import ascii_uppercase
from qfit.structure import Structure


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("structure", type=str,
                   help="PDB-file containing structure.")
    p.add_argument('chain', type=str,
            help="Chain in structure, e.g. A.")
    p.add_argument("structure2", type=str,
                    help="PDB-file containing structure2.")
    p.add_argument("res_list",type=str,
                    help="Path to the file containing the list of residues to compare")
    
    # Output options
    p.add_argument("-d", "--directory", type=os.path.abspath, default='.',
                   metavar="<dir>", help="Directory to store results.")
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

    structure = Structure.fromfile(args.structure).reorder()
    structure2 = Structure.fromfile(args.structure2).reorder()
    with open(args.res_list) as f:
        for line in f.readlines():
            print(line)
            res,aa = line.strip().split()
            structure_resi = structure.extract(f'resi {res} and chain {args.chain}')
            structure_resi2 = structure2.extract(f'resi {res} and chain {args.chain}')

            altlocs = len(list(set(structure_resi.altloc)))
            altlocs2 = len(list(set(structure_resi2.altloc)))
            print(args.structure[-8:-4],args.structure2[-8:-4],altlocs,altlocs2,structure_resi.resn[0],structure_resi2.resn[0],structure_resi.chain[0],structure_resi.resi[0],structure_resi2.chain[0],structure_resi2.resi[0])


if __name__ == '__main__':
    main()
