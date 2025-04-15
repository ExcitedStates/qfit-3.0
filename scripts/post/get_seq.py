#!/usr/bin/env python
"""
This script extracts chain-wise sequences from a PDB and outputs them in FASTA format.

INPUT:
- PDB file (processed by qfit.structure.Structure)
- --pdb: name of the input PDB (used for sequence IDs)

OUTPUT:
- FASTA file: <pdb>_chains.fasta with one entry per chain

Example:
    get_chain_seqs.py pdb.pdb --pdb pdb_name
"""

import numpy as np
from argparse import ArgumentParser
from qfit.structure import Structure

# 3-letter to 1-letter AA code dictionary
d = {
    "CYS": "C", "ASP": "D", "SER": "S", "GLN": "Q", "LYS": "K",
    "ILE": "I", "PRO": "P", "THR": "T", "PHE": "F", "ASN": "N",
    "GLY": "G", "HIS": "H", "LEU": "L", "ARG": "R", "TRP": "W",
    "ALA": "A", "VAL": "V", "GLU": "E", "TYR": "Y", "MET": "M"
}

def build_argparser():
    p = ArgumentParser(description=__doc__)
    p.add_argument("structure", type=str, help="PDB file containing structure.")
    p.add_argument("--pdb", required=True, help="Name of the input PDB (used for naming output).")
    return p

def get_chain_sequences(structure_file, pdb_name):
    struct = Structure.fromfile(structure_file).reorder()
    select = struct.extract("record", "ATOM", "==")

    chain_seqs = {}
    for c in np.unique(select.chain):
        seq = []
        for r in np.unique(select.extract("chain", c, "==").resi):
            resn = select.extract(f"chain {c} and resi {r}").resn[0]
            if resn in d:
                seq.append(d[resn])
        if seq:
            chain_seqs[f"{pdb_name}_{c}"] = "".join(seq)

    # Write FASTA
    fasta_file = f"{pdb_name}_chains.fasta"
    with open(fasta_file, "w") as out:
        for chain_id, seq in chain_seqs.items():
            out.write(f">{chain_id}\n{seq}\n")
    print(f"FASTA written to {fasta_file}")
    return chain_seqs

def main():
    p = build_argparser()
    args = p.parse_args()
    get_chain_sequences(args.structure, args.pdb)

if __name__ == "__main__":
    main()
