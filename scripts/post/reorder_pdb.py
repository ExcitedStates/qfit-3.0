#!/usr/bin/env python

"""Ordering PDB so that the ATOMs are listed first, and then HETATOMs"""

import argparse
import os

import numpy as np
from qfit.structure import Structure


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stru", type=str, help="PDB-file containing structure.")
    p.add_argument("name", type=str, help="pdb name")
    # Output options
    args = p.parse_args()

    return args


def main():
    args = parse_args()
    output_file = os.path.join(args.stru[:-4] + "_chain_renamed.pdb")

    stru = Structure.fromfile(args.stru).reorder()
    output = stru.extract("resi", 0, "==")  # to populate
    stru_atom = stru.extract("record", "ATOM")
    stru_hetatm = stru.extract("record", "HETATM")
    for chain in np.unique(stru_atom.chain):
        tmp = stru_atom.extract("chain", chain, "==")
        tmp_het = stru_hetatm.extract("chain", chain, "==")
        output = output.combine(tmp)
        output = output.combine(tmp_het)
    output.tofile(output_file)


if __name__ == "__main__":
    main()
