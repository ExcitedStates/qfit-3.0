#!/usr/bin/env python

"""Calculate RMSF by residue and altloc.

Iterate through altlocs of each residue,
reporting mean RMSD from all other altlocs in the residue.
"""

import numpy as np
import argparse
import logging
import os
import sys
import time
from string import ascii_uppercase
from qfit import Structure


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("structure", type=str,
                   help="PDB-file containing structure.")

    # Output options
    p.add_argument("-d", "--directory", type=os.path.abspath, default='.',
                   metavar="<dir>", help="Directory to store results.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Be verbose.")
    args = p.parse_args()

    return args


def main():
    args = parse_args()
    try:
        os.makedirs(args.directory)
    except OSError:
        pass

    structure = Structure.fromfile(args.structure).reorder()


    for residue in structure.extract('record',"ATOM").extract(
      'resn', "HOH","!=").residue_groups:
        altlocs = sorted(list(set(residue.altloc)))
        resi = residue.resi[0]
        chainid = residue.chain[0]
        tot_rmsd = 0.0
        numlocs = 0
        if len(altlocs) > 1:
            try:
                altlocs.remove('')
            except ValueError:
                pass
            for altloc1 in altlocs:
                conf1 = residue.extract('altloc', altloc1)
                for altloc2 in altlocs:
                    if altloc1 != altloc2:
                        conf2 = residue.extract('altloc', altloc2)
                        rmsd = conf1.rmsd(conf2)
                        tot_rmsd += rmsd
                        numlocs += 1
                if numlocs > 0:
                    print(resi, chainid, round(tot_rmsd / numlocs, 2), len(altlocs))
        else:
            print(resi, chainid, 0.0, len(altlocs))


if __name__ == '__main__':
    main()
