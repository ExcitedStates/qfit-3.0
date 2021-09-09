#!/usr/bin/env python

'''
This script will take in a PDB structure and output multiple csv files describing the closest protein residue and distance away from water molecules
'''

from argparse import ArgumentParser

import numpy as np
from qfit.structure import Structure


def parse_args():
    p = ArgumentParser(description=__doc__)
    p.add_argument("structure", type=str,
                   help="PDB-file containing structure.")
    p.add_argument("--dist", type=float, default='3.0',
                   help="Distance between water residue and close residues")
    p.add_argument("--pdb", help="Name of the input PDB.")
    args = p.parse_args()
    return args


def residue_closest(structure, dist, pdb_name):
    neighbors = {}
    pdb = structure.extract('record', 'ATOM')
    water = structure.extract('resn', 'HOH', '==')
    for chain in np.unique(water.chain):
        tmp_water = water.extract('chain', chain, '==')
        for i in tmp_water.resi:
            if len(tmp_water.extract('resi',i, '==').altloc) == 1:
                dist_pdb = np.linalg.norm(pdb.coor - tmp_water.extract('resi',i, '==').coor, axis=1)
            else:
                tmp_water2 = tmp_water.extract('resi',i, '==')
                for alt in tmp_water2.altloc:
                    value = value + alt
                    dist_pdb = np.linalg.norm(pdb.coor - tmp_water2.extract('altloc',alt, '==').coor, axis=1)
            water = str(i) + "," + chain + "," + str(tmp_water.extract('resi',i, '==').q)+ "," + str(tmp_water.extract('resi',i, '==').b) #info we want from the water molecule
            protein = str(pdb.resi[dist_pdb == np.amin(dist_pdb)]) + "," + str(pdb.chain[dist_pdb == np.amin(dist_pdb)]) + str(np.amin(dist_pdb)) #info we want from the protein
            neighbors[water] = protein #add both the water and protein values to the dictionary
    with open(pdb_name + '_waterclosestresidue.txt', 'w') as file:
        for key,value in neighbors.items():
            file.write(value + ',' + key + "\n")


def residues_close(structure, dist, pdb_name):
    neighbors = {}
    pdb = structure.extract('resn','HOH' , '!=')
    water = structure.extract('resn', 'HOH', '==')
    for chain in np.unique(water.chain):
        tmp_water = water.extract('chain', chain, '==')
        for i in tmp_water.resi:
            value = str(i) + ',' +chain
            if len(tmp_water.extract('resi',i, '==').altloc) == 1:
                dist_pdb = np.linalg.norm(pdb.coor - tmp_water.extract('resi', i , '==').coor, axis=1)
                for near_residue, near_chain in zip(pdb.resi[dist_pdb < dist], pdb.chain[dist_pdb < dist]):
                    key = str(near_residue)+" "+near_chain +" "+str(dist_pdb[near_residue])
                    if key not in neighbors:
                        neighbors[key] = value
    with open(pdb_name + '_' + str(dist) + '_watercloseresidue.txt', 'w') as file:
        for key,value in neighbors.items():
            residue_id,chain,dist = key.split()
            file.write(value + ',' + chain + ',' + residue_id + ',' + dist + "\n")


def residues_close_partial(structure, dist, pdb_name):
    neighbors = {}
    pdb = structure.extract('resn', 'HOH', '!=')
    water = structure.extract('resn', 'HOH', '==')
    for chain in np.unique(water.chain):
        tmp_water = water.extract('chain', chain, '==')
        for i in tmp_water.resi:
            if tmp_water.extract('resi',i, '==').q != 1:
                value = str(i) + ',' +chain
                dist_pdb = np.linalg.norm(pdb.coor - tmp_water.extract('resi',i, '==').coor, axis=1)
                for near_residue, near_chain in zip(pdb.resi[dist_pdb < dist], pdb.chain[dist_pdb < dist]):
                    key = str(near_residue)+" "+near_chain
                    if key not in neighbors:
                        neighbors[key]=value
    with open(pdb_name + '_' + str(dist) + '_partialwatercloseresidue.txt', 'w') as file:
        for key,value in neighbors.items():
            residue_id,chain = key.split()
            file.write(value + ',' + chain + ',' + residue_id + "\n")


def main():
    args = parse_args()
    # Load structure and prepare it
    structure = Structure.fromfile(args.structure).reorder() #put H20 on the bottom
    if not args.pdb is None:
        pdb_name = args.pdb
    else:
        pdb_name = ''
    residue_closest(structure, args.dist, pdb_name)
    # partial_occ_water(structure, args.pdb)


if __name__ == '__main__':
    main()
