
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
    resname = np.in1d(resname_unique,NT)
    with open('/wynton/group/fraser/swankowicz/PDB_2A_nonnucleotide_191120.txt', 'a') as pdb_file, open('/wynton/group/fraser/swankowicz/PDB_with_nucleotide_191120.txt', 'a') as pdb2:
      if True in resname:
       print('contains Nucleotide')
       pdb2.write(args.pdb + "\n")
      else:
       print('no nucleotide')
       pdb_file.write(args.pdb + "\n")

