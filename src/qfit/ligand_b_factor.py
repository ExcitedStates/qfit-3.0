import pkg_resources  # part of setuptools
from .qfit import QFitRotamericResidue, QFitRotamericResidueOptions
from .qfit import QFitSegment, QFitSegmentOptions
from .qfit import print_run_info
from .qfit_protein import QFitProteinOptions, QFitProtein
import multiprocessing as mp
import os.path
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from .structure.base_structure import _BaseStructure


def parse_args():
    p = ArgumentParser(description=__doc__)
    p.add_argument("structure", type=str,
                   help="PDB-file containing structure.")
    p.add_argument("ligand", type=str, help="ligand of interest")

   # Output options
    p.add_argument("--pdb", help="Name of the input PDB.")

    args = p.parse_args()
    return args

class Bfactor_options(QFitRotamericResidueOptions, QFitSegmentOptions):
    def __init__(self):
        super().__init__()
        self.nproc = 1
        self.ligand = None
        self.omit = False
        self.pdb = None

class B_factor():
    def __init__(self, structure, options):
        self.structure = structure 
        self.options = options 
        
    def run(self):
        if not self.options.pdb==None:
            self.pdb=self.options.pdb
        else:
            self.pdb=''
        self.get_bfactors()

    def get_bfactors(self):
        B_factor = pd.DataFrame()
        select = self.structure.extract('resn', self.options.ligand, '==')
        b_factor = select.b
        B_factor.loc[1,'PDB'] = self.pdb
        B_factor.loc[1,'Ligand_Name'] = self.options.ligand
        B_factor.loc[1,'Max_Bfactor'] = np.amax(b_factor)
        B_factor.loc[1, 'Average_Bfactor'] = np.average(b_factor)
        B_factor.to_csv(self.pdb + 'ligand_B_factors.csv', index=False)

def main():
    args = parse_args()
    # Load structure and prepare it
    structure = Structure.fromfile(args.structure).reorder() #put H20 on the bottom
    structure = structure.extract('e', 'H', '!=')
    B_options = Bfactor_options()
    B_options.apply_command_args(args)
    b_factor = B_factor(structure, B_options)
    b_output = b_factor.run()
    b_fin = b_factor.run()
