"""Delete duplicate atom entries"""
import numpy as np
import argparse
import logging
import os
import sys
import time
from string import ascii_uppercase
from . import Structure
from .structure import residue_type
from .structure.residue import _RotamerResidue
from .structure.rotamers import ROTAMERS

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("structure", type=str,
                   help="PDB-file containing structure.")

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
    mask = structure.active
    # Identify duplicated atoms:
    for i in range(len(structure.name)):
        if structure.resn[i] in ROTAMERS:
            continue
        for j in range(i, len(structure.name)):
            if structure.resn[j] in ROTAMERS:
                continue
            if (structure.resi[i]==structure.resi[j] and
                structure.resn[i]==structure.resn[j] and
                structure.altloc[i]==structure.altloc[j] and
                structure.icode[i]==structure.icode[j] and
                structure.chain[i]==structure.chain[j] and
                structure.name[i]==structure.name[j] and i!=j and
                mask[i]==True and mask[j]==True):
                mask[j]=False

    # Remove duplicated atoms
    data = {}
    for attr in structure.data:
        data[attr] = structure.data[attr][mask]
    new_structure = Structure(data)
    # Print the new structure to file
    new_structure.tofile(args.structure+".fixed")
