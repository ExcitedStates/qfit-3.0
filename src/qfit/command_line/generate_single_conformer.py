"""Convert a multi-conformer model to single-conformer."""
# XXX this seems like a duplicate of remove_altconfs

import argparse
import os

from qfit import Structure


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("structure", type=str, help="PDB-file containing structure.")

    # Output options
    p.add_argument(
        "-d",
        "--directory",
        type=os.path.abspath,
        default=".",
        metavar="<dir>",
        help="Directory to store results.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Be verbose.")
    args = p.parse_args()

    return args


def main():
    args = parse_args()
    os.makedirs(args.directory, exist_ok=True)

    structure = Structure.fromfile(args.structure).reorder()
    altloc = "A"
    structure = structure.extract("altloc", ("", altloc))
    structure.tofile(args.structure[0:-4] + "_single.pdb")
