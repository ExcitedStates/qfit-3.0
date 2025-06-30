import itertools
import logging
import os
from sys import argv
import copy
import subprocess
import argparse

import numpy as np
import pandas as pd

from qfit.samplers import ChiRotator, CBAngleRotator, BondRotator
from qfit.samplers import CovalentBondRotator, GlobalRotator
from qfit.samplers import RotationSets, Translator
from qfit.structure import Structure
from qfit.structure.residue import residue_type
from qfit.structure.rotamers import ROTAMERS

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("structure", help="PDB-file containing structure.", type=str)
    p.add_argument("pdb_id", help="PDB ID for output.", type=str)
    return p


class rotamers:
    def __init__(self, residue):
        self.residue = residue

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
                    'chain': np.unique(self.residue.chain)[0],
                    'residue': np.unique(self.residue.resi)[0],
                    'residue_name': np.unique(self.residue.resn)[0],
                    'altloc': np.unique(self.residue.altloc)[0],
                    'rotamer_value': r[n],
                    'nchi': n
                })
        return data



def get_rotamers(structure,pdb_id):
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
    df.to_csv(f'{pdb_id}_rotamers_output.csv', index=False)

def main():
    p = build_argparser()
    args = p.parse_args(args=None)
    structure = Structure.fromfile(args.structure).reorder()
    get_rotamers(structure, args.pdb_id)


if __name__ == "__main__":
    main()
