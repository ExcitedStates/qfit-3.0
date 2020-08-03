#Edited by Stephanie Wankowicz
#began: 2019-05-01
'''
Excited States software: qFit 3.0

Contributors: Saulo H. P. de Oliveira, Gydo van Zundert, Henry van den Bedem, Stephanie Wankowicz
Contact: vdbedem@stanford.edu
'''

import pkg_resources  # part of setuptools
from qfit.qfit import QFitRotamericResidue, QFitRotamericResidueOptions
from qfit.qfit_protein import QFitProteinOptions, QFitProtein
import os
import sys
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from math import ceil
from qfit.qfit import MapScaler, Structure, XMap
from qfit.structure.base_structure import _BaseStructure


def build_argparser():
    p = ArgumentParser(description=__doc__)
    p.add_argument("structure", type=str,
                   help="PDB-file containing structure.")

    # Output options
    p.add_argument("-d", "--directory", type=os.path.abspath, default='.',
                   metavar="<dir>", help="Directory to store results.")
    p.add_argument("--pdb", help="Name of the input PDB.")

    return p

class RMSF_options(QFitRotamericResidueOptions):
    def __init__(self):
        super().__init__()
        self.pdb = None

class RMSF():
    def __init__(self, options):
        self.options = options #user input
        self.structure = self.options.structure #PDB with HOH at the bottom

    def run(self):
        if not self.options.pdb == None:
            self.pdb = self.options.pdb + '_'
        else:
            self.pdb = ''
        self.average_coor_heavy_atom()


    def average_coor_heavy_atom(self):
        rmsf = pd.DataFrame()
        print(self.structure)
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
                    #iterating over each atom and getting center for each atom
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
        rmsf.to_csv(self.pdb + 'qfit_RMSF.csv')


def main():
    p = build_argparser()
    args = p.parse_args()

    options = RMSF_options()
    options.apply_command_args(args)

    rmsf = RMSF(options)
    rmsf_final = rmsf.run()
