#!/usr/bin/env python

import sys
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from qfit.qfit import QFitRotamericResidueOptions
from qfit.structure import Structure


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
        self.ca = None


class Bfactor():
    def __init__(self, structure, options):
        self.structure = structure
        self.options = options

    def run(self):
        if not self.options.pdb is None:
            self.pdb = self.options.pdb+'_'
        else:
            self.pdb = ''
        self.get_bfactors()

    def get_bfactors(self):
        B_factor = pd.DataFrame()
        b_factor = []
        select = self.structure.extract('record', 'ATOM', '==')
        select = select.extract('e', 'H', '!=')
        if not self.options.ca is None:
            select = select.extract('name', 'CA', '==')
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

        n = 1
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
            n += 1
        B_factor.to_csv(self.pdb + '_B_factors.csv', index=False)


def main():
    print(sys.path)
    p = build_argparser()
    args = p.parse_args()

    structure = Structure.fromfile(args.structure).reorder()
    print(structure)
    B_options = Bfactor_options()
    B_options.apply_command_args(args)

    b_factor = Bfactor(structure, B_options)
    b_factor.run()


if __name__ == '__main__':
    main()
