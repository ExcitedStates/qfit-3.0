#!/usr/bin/env python

'''Water overlap script.

This will take in two PDBs, one containing water molecules, one containing only protein or protein/hetatoms.
It will then determine how many clashes occur between the two and adjust accordingly.
'''

from argparse import ArgumentParser

import numpy as np
import pandas as pd
from qfit.clash import ClashDetector
from qfit.structure import Structure


def parse_args():
    p = ArgumentParser(description=__doc__)
    p.add_argument("no_water_structure", type=str,
                   help="PDB-file containing structure.")
    p.add_argument("water_structure", type=str,
                   help="PDB-file containing structure.")
    p.add_argument("map", type=str, help="ccp4 map of the structure")
    p.add_argument("--pdb", help="Name of the input PDB.")
    p.add_argument("--water", help="Name of the water structure.")
    p.add_argument("-r","--resolution", help="High resolution of structure")
    args = p.parse_args()
    return args


class Water_Options():
    def __init__(self):
        super().__init__()
        self.resolution = None
        self.distance = None
        self.pdb = None
        self.water = None

    def apply_command_args(self, args):
        for key, value in vars(args).items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class Water_Overlap:
    def __init__(self, structure_pro, structure_wa, model_map, options):
        self.structure_water = structure_wa
        self.structure_pro = structure_pro
        self.model_map = model_map
        self.options = options

    def run(self):
        self.set_up()
        self.compare_density()

    def set_up(self):
        if not self.options.pdb is None:
            self.pdb = self.options.pdb
        else:
            self.pdb = ''
        if not self.options.water is None:
            self.water = self.options.water
        else:
            self.water = ''

        #subset structure into protein and water
        self.water_str = self.structure_water.extract('resn', 'HOH', '==')
        self.pro_str = self.structure_pro.extract('record', 'ATOM')

    def compare_density(self):
        clash = []
        for c in (set(self.pro_str.chain)):
            #print(c)
            for r in set(self.pro_str.extract('chain', c, '==').resi):
                #print(r)
                self._cd = ClashDetector(self.water_str, self.pro_str.extract(f"resi {r} and chain {c}"), scaling_factor=0.75)
                if self._cd():
                    clash.append(tuple((c, r, np.unique(self.pro_str.extract(f"resi {r} and chain {c}").resn))))
        clash_summary = pd.DataFrame(clash, columns =['Chain', 'Resi', 'Resn'])
        clash_summary.to_csv(f'clash_summary_{self.pdb}_{self.water}.csv')


def main():
    args = parse_args()

    options = Water_Options()
    options.apply_command_args(args)
    structure_pro = Structure.fromfile(args.no_water_structure).reorder()
    structure_pro = structure_pro.extract('e', 'H', '!=')
    structure_wa = Structure.fromfile(args.water_structure).reorder()

    sub_structure = Water_Overlap(structure_pro, structure_wa, args.map, options)
    sub_structure.run()
