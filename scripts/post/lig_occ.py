import numpy as np
import pandas as pd
from argparse import ArgumentParser
from qfit.structure import Structure
from qfit.structure.base_structure import _BaseStructure

'''
To extract ligand occupancy from a PDB.
'''
def parse_args():
    p = ArgumentParser(description=__doc__)
    p.add_argument("structure", type=str,
                   help="PDB-file containing structure.")
    p.add_argument("-l", "--ligand")
    p.add_argument("--pdb", help="Name of the input PDB.")

    args = p.parse_args()
    return args

def get_occ(structure, ligand, pdb):
        lig_str = structure.extract('resn', ligand, '==')
        lig = []
        for i in np.unique(lig_str.chain):
            lig.append(tuple((pdb, ligand, i, np.amin(np.unique(lig_str.extract('chain', i, '==').q)), np.unique(lig_str.extract('chain', i, '==').q), len(set(lig_str.extract('chain', i, '==').altloc)))))
        occ = pd.DataFrame(lig, columns =['PDB', 'ligand_name', 'chain', 'min_occ', 'tot_occ', 'num_altloc'])
        occ.to_csv(pdb + '_ligand_occupancy.csv', index=False)

def main():
    args = parse_args()
    # Load structure and prepare it
    structure = Structure.fromfile(args.structure).reorder()
    structure = structure.extract('e', 'H', '!=')
    get_occ(structure, args.ligand, args.pdb)

if __name__ == '__main__':
    main()

