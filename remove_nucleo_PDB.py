import numpy as np
import argparse
import logging
import os
import sys
import time
from string import ascii_uppercase
from . import Structure
from .structure import residue_type
from .structure.rotamers import ROTAMERS

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("structure", type=str,
                   help="PDB-file containing structure.")
    p.add_argument("pdb", type=str, help="PDB name")
    args = p.parse_args()
    return args


def main():
    args = parse_args()
    try:
        structure = Structure.fromfile(args.structure)
    except:
        return
    NT = np.array(['DA','DT','DC', 'DG', 'DU'])
    resname_unique = np.unique(structure.resn)
    resname = np.in1d(resname_unique, NT)
    with open('no_nucleotide_file.txt', 'a') as no_nucleotide_file, open('nucleotide_file.txt', 'a') as nucleotide_file:
      if True in resname:
       print('contains Nucleotide')
       nucleotide_file.write(args.pdb + "\n")
      else:
       print('no nucleotide')
       no_nucleotide_file.write(args.pdb + "\n")
