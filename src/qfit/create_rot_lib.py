import itertools
import logging
import os
from sys import argv
import copy
import subprocess
import numpy as np
import argparse

from .samplers import ChiRotator, CBAngleRotator, BondRotator
from .samplers import CovalentBondRotator, GlobalRotator
from .samplers import RotationSets, Translator
from .structure import Structure, _Segment, calc_rmsd
from .structure.residue import residue_type
from .structure.rotamers import ROTAMERS


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("structure", help="PDB-file containing structure.", type=str)
    p.add_argument("selection", help="residue chain and id", type=str)
    return p


class rotamers:
    def __init__(self, residue):
        self.residue = residue

    def run(self):
        # get_all rotamers
        rotamers = self.residue.rotamers
        rotamers.append(
            [self.residue.get_chi(i) for i in range(1, self.residue.nchi + 1)]
        )
        # get number of chi angles
        nchi = self.residue.nchi
        for r in rotamers:
            for n in range(0, nchi):
                self.residue.set_chi(n + 1, r[n])
            self.residue.tofile(f"{self.residue.resn[0]}_{r}.pdb")


def get_rotamers(structure, selection):
    chainid, residue_id = selection.split(",")
    structure_resi = structure.extract(f"resi {residue_id} and chain {chainid}")
    chain = structure_resi[chainid]
    conformer = chain.conformers[0]
    residue = conformer[int(residue_id)]
    rot = rotamers(residue)
    rot.run()


def main():
    p = build_argparser()
    args = p.parse_args(args=None)
    structure = Structure.fromfile(args.structure).reorder()
    get_rotamers(structure, args.selection)


if __name__ == "__main__":
    main()
