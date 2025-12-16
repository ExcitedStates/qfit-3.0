#!/usr/bin/env python
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from qfit.structure import Structure
import itertools


def build_argparser():
    p = ArgumentParser(description="Calculate average B-factor per residue.")
    p.add_argument("structure", type=str, help="PDB-file containing structure.")
    p.add_argument(
        "--ca",
        dest="ca",
        action="store_true",
        help="Return B factor for alpha carbons only",
    )
    p.add_argument(
        "--sidechain",
        dest="sc",
        action="store_true",
        help="Return B factor for sidechain atoms only",
    )
    p.add_argument("--pdb", help="Name of the input PDB.")
    return p


def get_bfactor(structure, pdb, ca, sidechain):
    if ca and sidechain:
        print("Both alpha carbon and sidechain selected. Please choose one")
        return

    bfactors = []

    select = structure.extract("record", "ATOM", "==")
    select = select.extract("e", "H", "!=")

    if ca:
        select = select.extract("name", "CA", "==")
    if sidechain:
        # Excludes N, CA, C, O (standard backbone)
        select = select.extract("name", (["N", "CA", "C", "O"]), "!=")

    resi_icode = np.array(list(zip(select.resi, select.icode)))
    unique_chain_resi_icode = []

    for c in np.unique(select.chain):
        chain_select = select.extract("chain", c, "==")
        # Find unique resi/icode pairs only within the current chain
        chain_resi_icode = np.array(list(zip(chain_select.resi, chain_select.icode)))
        unique_chain_resi_icode.extend([
            (c, resi, icode)
            for resi, icode in np.unique(chain_resi_icode, axis=0)
        ])

    # Iterate through unique (chain, resi, icode) combinations
    for c, r, i in unique_chain_resi_icode:
        # Construct the selection string using resi AND icode
        # If icode is an empty string (''), the selection still works correctly.
        selection_string = f"resi {r} and chain {c}"
        if i.strip(): # Only add icode if it's not empty/default
            selection_string += f" and icode {i.strip()}"

        # Extract the atoms belonging to the unique residue identifier
        residue_atoms = select.extract(selection_string)
        if len(residue_atoms.resn) == 0:
           continue

        b_factor = np.average(residue_atoms.b)

        resn_str = residue_atoms.resn[0].strip()
        chain_str = c.strip()
        icode_str = i.strip()

        # Append result tuple
        bfactors.append(
            (
                pdb,
                str(r), # resi (sequence number)
                resn_str, # resn (residue name)
                chain_str, # chain
                icode_str, # icode (newly added)
                b_factor,
                len(np.unique(residue_atoms.altloc)), # num_altlocs
            )
        )

    B_factor = pd.DataFrame(
        bfactors,
        columns=["PDB", "resi", "resn", "chain", "icode", "b_factor", "num_altlocs"], # Updated columns
    )
    med = B_factor["b_factor"].median()  # median b-factor

    return B_factor, med

def main():
    p = build_argparser()
    args = p.parse_args()
    structure = Structure.fromfile(args.structure).reorder()
    B_factor, median = get_bfactor(structure, args.pdb, args.ca, args.sc)
    B_factor.to_csv(args.pdb + "_B_factors.csv", index=False)
    print(median)  # for analysis scripts to give to OP script
    return median


if __name__ == "__main__":
    main()
