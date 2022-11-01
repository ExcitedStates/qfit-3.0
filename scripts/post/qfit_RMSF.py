#!/usr/bin/env python

from qfit.qfit import QFitOptions
from qfit.qfit_protein import QFitProtein
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from qfit.structure import Structure


def build_argparser():
    p = ArgumentParser(description=__doc__)
    p.add_argument("structure", type=str,
                   help="PDB-file containing structure.")

    # Output options
    p.add_argument("-d", "--directory", type=os.path.abspath, default='.',
                   metavar="<dir>", help="Directory to store results.")
    p.add_argument("--pdb", help="Name of the input PDB.")

    return p


class RMSF_options(QFitOptions):
    def __init__(self):
        super().__init__()
        self.pdb = None


class RMSF():
    def __init__(self, options):
        self.options = options #user input
        self.structure = self.options.structure

    def run(self):
        if not self.options.pdb is None:
            self.pdb = self.options.pdb + '_'
        else:
            self.pdb = ''
        self.average_coor_heavy_atom()


    def average_coor_heavy_atom(self):
        structure = Structure.fromfile(self.structure)
        select = structure.extract('record', 'ATOM', '==')
        rmsf_data = []

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
            for resid in residue_ids:
                res_tmp = select2.extract('resi', int(resid), '==')  # this separates each residue
                resn_name = (np.unique(res_tmp.resi)[0], np.unique(res_tmp.resn)[0], np.unique(res_tmp.chain)[0])
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
                        RMSF_list.append((sum(RMSF_atom_list)/len(RMSF_atom_list))[0])
                    rmsf_data.append(tuple((resn_name[0],resn_name[1],resn_name[2],(sum(RMSF_list)/len(RMSF_list)))))
                else:
                    rmsf_data.append(tuple((resn_name[0],resn_name[1],resn_name[2],0)))

        rmsf = pd.DataFrame(rmsf_data, columns=['resseq', 'AA', 'Chain', 'RMSF'])
        rmsf['PDB_name'] = self.options.pdb
        rmsf.to_csv(self.options.dir + self.pdb + 'qfit_RMSF.csv')


def main():
    p = build_argparser()
    args = p.parse_args()

    options = RMSF_options()
    options.apply_command_args(args)

    rmsf = RMSF(options)
    rmsf.run()


if __name__ == '__main__':
    main()
