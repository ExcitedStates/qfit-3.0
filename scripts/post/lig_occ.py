#!/usr/bin/env python
"""
This script will take in a PDB and ligand code and return the occupancy and b-factors of each ligand conformer. 
INPUT: PDB file, name of PDB, name of ligand
OUTPUT: Text file {pdb_name}_ligand_occupancy.csv with ligand occupancy information

example:
lig_occ.py pdb.pdb --pdb {pdb name} -l {ligand name}
"""

from argparse import ArgumentParser

import numpy as np
import pandas as pd
from qfit.structure import Structure


def parse_args():
    p = ArgumentParser(description=__doc__)
    p.add_argument("structure", type=str, help="PDB-file containing structure.")
    p.add_argument("-l", "--ligand")
    p.add_argument("--pdb", help="Name of the input PDB.")
    p.add_argument("--qFit", help="qFit included in output file name")

    args = p.parse_args()
    return args


def get_occ(structure, ligand, pdb, qfit):
    lig_str = structure.extract("resn", ligand, "==")
    lig = []
    for i in np.unique(lig_str.chain):
        lig.append(
            tuple(
                (
                    pdb,
                    ligand,
                    i,
                    np.amin(np.unique(lig_str.extract("chain", i, "==").q)),
                    np.average(lig_str.extract("chain", i, "==").b),
                    len(set(lig_str.extract("chain", i, "==").altloc)),
                )
            )
        )
    occ = pd.DataFrame(
        lig,
        columns=["PDB", "ligand_name", "chain", "min_occ", "average_b", "num_altloc"],
    )
    occ.to_csv(pdb + qfit + "_ligand_occupancy.csv", index=False)


def main():
    args = parse_args()
    if not args.qFit == None:
        qfit = _qFit
    else:
        qfit = ''
    # Load structure and prepare it
    structure = Structure.fromfile(args.structure).reorder()
    structure = structure.extract("e", "H", "!=")
    get_occ(structure, args.ligand, args.pdb, qfit)


if __name__ == "__main__":
    main()
