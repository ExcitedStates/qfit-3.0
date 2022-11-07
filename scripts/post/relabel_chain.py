#!/usr/bin/env python
import numpy as np
import argparse
import os
from string import ascii_uppercase
from qfit.structure import Structure

"""Renaming Chains in holo based on corresponding apo"""

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
    args = p.parse_args()
    return args

def main():
    args = parse_args()
    output_holo_file = os.path.join(args.holo_str[:-4]+"_renamed.pdb")
    holo = Structure.fromfile(args.holo_str)
    apo = Structure.fromfile(args.apo_str)
    apo = apo.extract('record', 'ATOM')
    output_holo = holo.extract("resi", 0, '==')
    for chain_h in np.unique(holo.chain):
        holo_copy = holo.copy()
        tmp_h = holo.extract("chain", chain_h, '==')
        tmp_h_atom = tmp_h.extract('record', 'ATOM')
        dist = None
        for chain_a in np.unique(apo.chain):
            tmp_a = apo.extract("chain", chain_a, '==')
            tot_dist = 0
            for coor in tmp_h_atom.coor:
               tot_dist += np.linalg.norm(tmp_a.coor - coor, axis=1)
               tmp_dist = np.median(tot_dist)
            if dist == None:
                 dist = tmp_dist
                 rename_chain = chain_a
            else:
                 if dist > tmp_dist:
                tot_dist += np.linalg.norm(tmp_a.coor - coor, axis=1)
                tmp_dist = np.median(tot_dist)
            if dist is None:
                dist = tmp_dist
                rename_chain = chain_a
            else:
                if dist > tmp_dist:
                    dist = tmp_dist
                    rename_chain = chain_a
        output = holo_copy.extract("chain", chain_h, '==')
        output.chain = rename_chain
        output_holo = output_holo.combine(output)
        del tmp_h
    output_holo.reorder().tofile(output_holo_file)

if __name__ == '__main__':
    main()
                                                                                                            
