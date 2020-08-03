#Edited by Stephanie Wankowicz
#began: 2019-05-01
'''
Excited States software: qFit 3.0
Contributors: Saulo H. P. de Oliveira, Gydo van Zundert, Henry van den Bedem, Stephanie Wankowicz
Contact: vdbedem@stanford.edu
How to run:
b_factor $pdb.mtz $pdb.pdb --pdb $pdb
'''

import pkg_resources  # part of setuptools
from qfit.qfit import QFitRotamericResidue, QFitRotamericResidueOptions
from qfit.qfit_protein import QFitProteinOptions, QFitProtein
import os.path
import os
import sys
import copy
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from math import ceil
from qfit.structure import Structure
#from .structure.base_structure import _BaseStructure


os.environ["OMP_NUM_THREADS"] = "1"

def build_argparser():
    p = ArgumentParser(description=__doc__)
    p.add_argument("structure", type=str,
                   help="PDB-file containing structure.")
   # Output options
    p.add_argument("--pdb", help="Name of the input PDB.")

    return p

class Bfactor_options(QFitRotamericResidueOptions):
    def __init__(self):
        super().__init__()
        self.pdb = None

class B_factor():
    def __init__(self, structure, options):
        self.structure = structure 
        self.options = options 
    
    def run(self):
        if not self.options.pdb == None:
            self.pdb = self.options.pdb+'_'
        else:
            self.pdb = ''
        self.get_bfactors()

    def get_bfactors(self):
        B_factor = pd.DataFrame()
        atom_name = []
        chain_ser = []
        residue_name = []
        b_factor = []
        residue_num = []
        model_number = []
        select = self.structure.extract('record', 'ATOM', '==')
        n = 0
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
        
        n=1
        for id in residue_ids:
            res_tmp = select2.extract('resi', int(id), '==') #this is seperating each residues
            #is this going to give us the alternative coordinate for everything?
            resn_name = (np.array2string(np.unique(res_tmp.resi)), np.array2string(np.unique(res_tmp.resn)),np.array2string(np.unique(res_tmp.chain)))
            b_factor = res_tmp.b
            B_factor.loc[n,'resseq'] = resn_name[0]
            B_factor.loc[n,'AA'] = resn_name[1]
            B_factor.loc[n,'Chain'] = resn_name[2]
            B_factor.loc[n,'Max_Bfactor'] = np.amax(b_factor)
            B_factor.loc[n, 'Average_Bfactor'] = np.average(b_factor)
            n+=1
        B_factor.to_csv(self.pdb + '_B_factors.csv', index=False)

def main():
    print(sys.path)
    p = build_argparser()
    args = p.parse_args()

    structure = Structure.fromfile(args.structure).reorder()
    print(structure)
    B_options = Bfactor_options()
    B_options.apply_command_args(args)

    b_factor = B_factor(structure, B_options)
    b_fin = b_factor.run()

