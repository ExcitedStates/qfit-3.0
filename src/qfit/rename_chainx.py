"""Renaming Chain X based on corresponding apo/holo"""
import numpy as np
import argparse
import logging
import copy
import os
import sys
from string import ascii_uppercase
from . import Structure
from .structure import residue_type


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("holo_str", type=str,
                   help="PDB-file containing structure.")
    p.add_argument("apo_str", type=str,
                   help="PDB-file containing structure.")
    p.add_argument("holo_name", type=str,
                   help='holo pdb name')
    p.add_argument("apo_name", type=str,
                   help='holo pdb name')
    p.add_argument("directory", type=str,
                  help="text file output location")
    # Output options
    args = p.parse_args()
    return args

def main():
    args = parse_args()
    output_holo_file = os.path.join(args.holo_str[:-4]+"_renamed.pdb")
    output_apo_file = os.path.join(args.apo_str[:-4]+"_renamed.pdb")

    holo = Structure.fromfile(args.holo_str)#.reorder()
    apo = Structure.fromfile(args.apo_str)#.reorder()

    if 'X' not in holo.chain and 'X' not in apo.chain:
        print('no chain X')
        exit()

    elif 'X' in holo.chain and 'X' in apo.chain:
       print('Chain X in both structures')
       exit()

    elif 'X' in holo.chain:

       if len(np.unique(apo.chain)) == 1:
          print('only one chain')
          holo.chain = np.unique(apo.chain)
          holo.tofile(output_holo_file)
          with open(args.directory + args.holo_name + 'renamed.txt', 'w') as file:
               file.write('Yes')
       else:
          tmp_holo = holo.extract("chain", 'X', '==')
          tmp_holo = tmp_holo.extract('record', 'ATOM')
          tmp_holo = tmp_holo.extract('e', 'H', '!=')
          tmp_apo = apo.extract('record', 'ATOM')
          tmp_apo = tmp_apo.extract('e', 'H', '!=')
          for chain in np.unique(apo.chain):
              tmp_apo = tmp_apo.extract("chain", chain, '==')
              tot_dist = 0
              for coor in tmp_holo.coor:
                tot_dist += np.linalg.norm(tmp_apo.coor - coor, axis=1)

    elif 'X' in apo.chain:
        if len(np.unique(holo.chain)) == 1:
           apo.chain = np.unique(holo.chain)
           apo.tofile(output_apo_file)
           with open(args.directory + args.apo_name + 'renamed.txt', 'w') as file:
               file.write('Yes')
        else:
          print('more than one chain')
          tmp_apo = apo.extract("chain", 'X', '==')
          tmp_apo = tmp_apo.extract('record', 'ATOM')
          tmp_apo = tmp_apo.extract('e', 'H', '!=')
          tmp_holo = holo.extract('record', 'ATOM')
          tmp_holo = tmp_holo.extract('e', 'H', '!=')
          for chain in np.unique(holo.chain):
              tmp_holo = tmp_holo.extract("chain", chain, '==')
              tot_dist = 0
              for coor in tmp_holo.coor:
                #dist = np.linalg.norm(tmp_apo.coor - coor, axis=1)
                tot_dist += np.linalg.norm(tmp_apo.coor - coor, axis=1)