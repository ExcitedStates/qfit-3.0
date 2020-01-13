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
from .qfit import QFitRotamericResidue, QFitRotamericResidueOptions
from .qfit import QFitSegment, QFitSegmentOptions
from .qfit import print_run_info
from .qfit_protein import QFitProteinOptions, QFitProtein
import multiprocessing as mp
import os.path
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
<<<<<<< HEAD

=======
>>>>>>> 3069973e5d2fe3ae019b565f0c625624e36df374
   # Output options
    p.add_argument("--pdb", help="Name of the input PDB.")

    args = p.parse_args()
    return args

class Bfactor_options(QFitRotamericResidueOptions, QFitSegmentOptions):
    def __init__(self):
        super().__init__()
        self.nproc = 1
        self.verbose = True
        self.omit = False
        self.pdb = None

class B_factor():
    def __init__(self, structure, options):
        self.structure = structure #PDB with HOH at the bottom
        self.options = options #user input
    
    def run(self):
        if not self.options.pdb==None:
            self.pdb=self.options.pdb
        else:
            self.pdb=''
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
<<<<<<< HEAD
=======
            #is this going to give us the alternative coordinate for everything?
>>>>>>> 3069973e5d2fe3ae019b565f0c625624e36df374
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
    args = parse_args()
    # Load structure and prepare it
    structure = Structure.fromfile(args.structure).reorder() #put H20 on the bottom
<<<<<<< HEAD
    structure = structure.extract('e', 'H', '!=')
=======
>>>>>>> 3069973e5d2fe3ae019b565f0c625624e36df374
    B_options = Bfactor_options()
    B_options.apply_command_args(args)
    time0 = time.time()
    b_factor = B_factor(structure, B_options)
<<<<<<< HEAD
    b_output = b_factor.run()
=======
    b_fin = b_factor.run()
>>>>>>> 3069973e5d2fe3ae019b565f0c625624e36df374

