#!/usr/bin/env python

from argparse import ArgumentParser

import numpy as np
import pandas as pd
from qfit.structure import Structure


def parse_args():
    p = ArgumentParser(description=__doc__)
    p.add_argument("holo_structure", type=str,
                   help="PDB-file containing structure.")
    p.add_argument("apo_structure", type=str,
                   help="PDB-file containing structure.") #this should be a superposed PDB files
    p.add_argument("holo_pdb_name", type=str, help="name of Holo PDB.")
    p.add_argument("apo_pdb_name", type=str, help="name of Holo PDB.")
    p.add_argument("-dist", type=float, default='5.0',
                   metavar="<float>", help="angstrom distance from ligand")
    p.add_argument("-lig", type=str, help="ligand name")
    args = p.parse_args()
    return args


def rmsd_alpha_carbon(holo, apo, holo_name, apo_name):
    for chain in np.unique(holo.chain):
        if chain not in apo.chain:
            print('not in apo')
            continue
    holo = holo.extract('chain', chain, '==')
    apo = apo.extract('chain', chain, '==')
    rmsd = []
    for i in np.unique(holo.resi):
        CA_holo = holo.extract(f'chain {chain} and resi {i}')
        CA_apo = apo.extract(f'chain {chain} and resi {i}')
        CA_holo = CA_holo.extract('name','CA', '==').coor.mean(axis=0)
        CA_apo = CA_apo.extract('name','CA', '==').coor.mean(axis=0)
        rmsd.append(tuple((chain, i, np.linalg.norm(CA_holo - CA_apo, axis=0))))
    df_rmsf = pd.DataFrame(rmsd, columns=['Chain', 'Residue', 'RMSD'])
    df_rmsf.to_csv(holo_name + '_' + apo_name + '_rmsd.csv')


def main():
    args = parse_args()
    holo_structure = Structure.fromfile(args.holo_structure)
    apo_structure = Structure.fromfile(args.apo_structure)

    holo_receptor = holo_structure.extract('record', 'ATOM')
    holo_receptor = holo_receptor.extract('e', 'H', '!=')

    apo_receptor = apo_structure.extract('record', 'ATOM')
    apo_receptor = apo_receptor.extract('e', 'H', '!=')

    rmsd_alpha_carbon(holo_receptor, apo_receptor, args.holo_pdb_name, args.apo_pdb_name)


if __name__ == '__main__':
    main()
