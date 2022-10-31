#!/usr/bin/env python
'''
The purpose of this script is to calculate the distance between each alpha carbon of two sequence matched PDB structures. This script is also dependent on
the two structures having the same numbered residues and chain ids. 

INPUT: 2 PDB structures, names of PDB structures
OUTPUT: CSV file with distance between the alpha carbon atom of every residue between the two input structures

example:
alpha_carbon_rmsd.py pdb1.pdb pdb2.pdb pdb1_name pdb2_name 


'''

import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from qfit.structure import Structure


def parse_args():
    p = ArgumentParser(description=__doc__)
    p.add_argument("structure1", type=str,
                   help="PDB-file containing structure.")
    p.add_argument("structure2", type=str,
                   help="PDB-file containing structure.") 
    p.add_argument("pdb_name1", type=str, help="name of first PDB.")
    p.add_argument("pdb_name2", type=str, help="name of second PDB.")
    args = p.parse_args()
    return args

def rmsd_alpha_carbon(pdb1, pdb2, pdb1_name, pdb2_name):
   for chain in np.unique(pdb1.chain):
    if chain not in pdb2.chain:
       continue
    pdb1 = pdb1.extract('chain', chain, '==')
    pdb2 = pdb2.extract('chain', chain, '==')
    rmsd = []
    for i in np.unique(pdb1.resi):
        CA1 = pdb1.extract(f'chain {chain} and resi {i}')
        CA2 = pdb2.extract(f'chain {chain} and resi {i}')
        CA1 = CA1.extract('name','CA', '==').coor.mean(axis=0)
        CA2 = CA2.extract('name','CA', '==').coor.mean(axis=0)
        rmsd.append(tuple((chain, i, np.linalg.norm(CA1 - CA1, axis=0))))
   df_rmsf = pd.DataFrame(rmsd, columns=['Chain', 'Residue', 'RMSD'])
   df_rmsf.to_csv(pdb1_name + '_' + pdb2_name + '_rmsd.csv', index=False)

def main():
    args = parse_args()
    structure1 = Structure.fromfile(args.structure1)
    structure2 = Structure.fromfile(args.structure2)

    structure1 = structure1.extract('record', 'ATOM') #we only want to look at protein atoms

    structure2 = structure2.extract('record', 'ATOM')


    rmsd_alpha_carbon(structure1, structure2, args.pdb_name1, args.pdb_name2)

if __name__ == '__main__':
    main()
