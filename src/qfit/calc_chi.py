import numpy as np
import argparse
import pandas as pd

from .samplers import ChiRotator, CBAngleRotator, BondRotator
from .samplers import CovalentBondRotator, GlobalRotator
from .samplers import RotationSets, Translator
from .structure import Structure
from .structure.residue import residue_type
from .structure.rotamers import ROTAMERS

'''
This script will take in a PDB file and output a csv with the chi angle of each residue and altloc. 
'''

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("structure", help="PDB-file containing structure.", type=str)
    return p


class rotamers:
    def __init__(self, residue):
        self.residue = residue
        print(self.residue.resi)

    def run(self):
        # get_all rotamers
        rotamers = []
        rotamers.append(
            [self.residue.get_chi(i) for i in range(1, self.residue.nchi + 1)]
        )
        # get number of chi angles
        nchi = self.residue.nchi

        # Prepare data for CSV
        data = []
        for r in rotamers:
            for n in range(0, nchi):
                self.residue.set_chi(n + 1, r[n])
                data.append({
                    'chain': np.unique(self.residue.chain),
                    'residue': np.unique(self.residue.id),
                    'residue_name': np.unique(self.residue.resn[0]),
                    'altloc': np.unique(self.residue.altloc),
                    'rotamer_value': r[n],
                    'nchi': n
                })
        return data



def get_rotamers(structure):
    data = []
    for chain in np.unique(structure.chain):
        chain_structure = structure.extract("chain", chain, "==")
        for residue_id in np.unique(chain_structure.resi):
            residue_structure = chain_structure.extract("resi", residue_id, "==")
            for altloc in np.unique(residue_structure.altloc):
                altloc_structure = residue_structure.extract("altloc", altloc)
                residue = list(altloc_structure.single_conformer_residues)[0]
                rot = rotamers(residue)
                data.extend(rot.run())


    df = pd.DataFrame(data)
    df.to_csv('rotamers_output.csv', index=False)

def main():
    p = build_argparser()
    args = p.parse_args(args=None)
    structure = Structure.fromfile(args.structure).reorder()
    get_rotamers(structure)


if __name__ == "__main__":
    main()
