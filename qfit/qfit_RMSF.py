#Edited by Stephanie Wankowicz
#began: 2019-05-01
'''
Excited States software: qFit 3.0

Contributors: Saulo H. P. de Oliveira, Gydo van Zundert, Henry van den Bedem, Stephanie Wankowicz
Contact: vdbedem@stanford.edu
'''

import pkg_resources  # part of setuptools
from .qfit import QFitRotamericResidue, QFitRotamericResidueOptions
from .qfit import QFitSegment, QFitSegmentOptions
from .qfit import print_run_info
from .qfit_protein import QFitProteinOptions, QFitProtein
import os
import sys
import time
import copy
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from math import ceil
from . import MapScaler, Structure, XMap
from .structure.base_structure import _BaseStructure


os.environ["OMP_NUM_THREADS"] = "1"

def parse_args():
    p = ArgumentParser(description=__doc__)
    p.add_argument("structure", type=str,
                   help="PDB-file containing structure.")

    # Output options
    p.add_argument("-d", "--directory", type=os.path.abspath, default='.',
                   metavar="<dir>", help="Directory to store results.")
    p.add_argument("--pdb", help="Name of the input PDB.")

    args = p.parse_args()
    return args

class RMSF_options(QFitRotamericResidueOptions, QFitSegmentOptions):
    def __init__(self):
        super().__init__()
        self.nproc = 1
        self.verbose = True
        self.omit = False
        self.pdb = None

class RMSF():
    def __init__(self, structure, options):
        self.structure = structure #PDB with HOH at the bottom
        self.options = options #user input
    def run(self):
        if not self.options.pdb == None:
            self.pdb = self.options.pdb + '_'
        else:
            self.pdb = ''
        self.average_coor_heavy_atom()


    def average_coor_heavy_atom(self):
        rmsf = pd.DataFrame()
        select = self.structure.extract('record', 'ATOM', '==')
        n=0
        for chain in np.unique(select.chain):
            select2 = select.extract('chain', chain, '==')
            residues = set(list(select2.resi))
            residue_ids = []
            for i in residues:
                tmp_i = str(i)
                if ':' in tmp_i:
                    resi = int(tmp_i.split(':')[1][1:])
                else:
                    resi = tmp_i
                residue_ids.append(resi)
            for id in residue_ids:
                res_tmp = select2.extract('resi', int(id), '==') #this is seperating each residues
                resn_name = (np.array2string(np.unique(res_tmp.resi)), np.array2string(np.unique(res_tmp.resn)),np.array2string(np.unique(res_tmp.chain)))
                if len(np.unique(res_tmp.altloc))>1:
                    RMSF_list = []
                    num_alt = len(np.unique(res_tmp.altloc))
                    #now iterating over each atom and getting center for each atom
                    for atom in np.unique(res_tmp.name):
                        RMSF_atom_list = []
                        tmp_atom = res_tmp.extract('name', atom, '==')
                        atom_center = tmp_atom.coor.mean(axis=0)
                        for i in np.unique(tmp_atom.altloc):
                            atom_alt=tmp_atom.extract('altloc', i, '==')
                            RMSF_atom=np.linalg.norm(atom_alt.coor-atom_center, axis=1)
                            RMSF_atom_list.append(RMSF_atom)
                        RMSF=(sum(RMSF_atom_list)/len(RMSF_atom_list))
                        RMSF_list.append(RMSF)
                        rmsf.loc[n,'resseq']=resn_name[0]
                        rmsf.loc[n,'AA']=resn_name[1]
                        rmsf.loc[n,'Chain']=resn_name[2]
                        rmsf.loc[n,'rmsf']=(sum(RMSF_list)/ len(RMSF_list))
                else:
                    rmsf.loc[n,'resseq']=resn_name[0]
                    rmsf.loc[n,'AA']=resn_name[1]
                    rmsf.loc[n,'Chain']=resn_name[2]
                    rmsf.loc[n,'rmsf']=0
                n+=1
        rmsf['resseq'] = rmsf['resseq'].str.replace('[', '')
        rmsf['resseq'] = rmsf['resseq'].str.replace(']', '')
        rmsf['AA'] = rmsf['AA'].str.replace('[', '')
        rmsf['AA'] = rmsf['AA'].str.replace(']', '')
        rmsf['AA'] = rmsf['AA'].str.replace('\'', '')
        rmsf['Chain'] = rmsf['Chain'].str.replace('[', '')
        rmsf['Chain'] = rmsf['Chain'].str.replace(']', '')
        rmsf['Chain'] = rmsf['Chain'].str.replace('\'', '')
        rmsf['PDB_name'] = self.options.pdb
        rmsf.to_csv(self.pdb+'qfit_RMSF.csv')


def main():
    args = parse_args()
    structure = Structure.fromfile(args.structure).reorder() #put H20 on the bottom
    R_options = RMSF_options()
    R_options.apply_command_args(args)
    time0 = time.time()
    rmsf = RMSF(structure, R_options)
    fin_rmsf = rmsf.run()
