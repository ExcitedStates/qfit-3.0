#!/usr/bin/env python

import argparse
import os

from qfit.structure import Structure
from qfit.structure.rotamers import ROTAMERS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process multiconformer file for refinement."
    )
    parser.add_argument("structure", type=str, help="Path to the structure file")
    return parser.parse_args()


def create_refine_restraints(multiconformer):
    """
    Create occupancy constraint groups for refinement. Residues in the same
    segments that share an altloc are constrained together.
    """
    fname = "qFit_occupancy.params"
    with open(fname, "w+") as f:
        f.write("refinement {\n")
        f.write("  refine {\n")
        f.write("    occupancies {\n")

        segment = []  # list of (resi, chain, set_of_altlocs)

        def flush(seg):
            if not seg:
                return
            all_altlocs = set().union(*(alts for _, _, alts in seg))
            for a in sorted(all_altlocs):
                members = [(resi, ch) for resi, ch, alts in seg if a in alts]
                if not members:
                    continue
                f.write("      constrained_group {\n")
                parts = [f"(chain {ch} and resseq {resi})" for resi, ch in members]
                resi_selection = " or ".join(parts)
                f.write(f"        selection = altid {a} and ({resi_selection})\n")
                f.write("             }\n")

        prev_chain = None
        for chain in multiconformer:
            for residue in chain:
                if residue.resn[0] not in ROTAMERS:
                    continue
                ch = residue.chain[0]
                if ch != prev_chain:
                    flush(segment)
                    segment = []
                    prev_chain = ch
                if len(residue.extract("name", "CA", "==").q) == 1:
                    flush(segment)
                    segment = []
                else:
                    alts = {alt[0] for alt in set(residue.altloc) if alt}
                    segment.append((residue.resi[0], ch, alts))
        flush(segment)

        f.write("   }\n")
        f.write(" }\n")
        f.write("}\n")



def main():
    args = parse_args()
    structure = Structure.fromfile(args.structure).reorder()
    create_refine_restraints(structure)


if __name__ == "__main__":
    main()
