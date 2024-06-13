"""Delete duplicate atom entries"""
import argparse
import os

import numpy as np

from qfit import Structure
from qfit.structure.rotamers import ROTAMERS


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


def find_unique_atoms(structure):
    """Find unique atoms in a structure.

    Atoms that belong to amino acids are assumed to be unique.

    Args:
        structure (qfit.Structure): structure to scan for unique atoms.

    Returns:
        np.ndarray[bool]: a (structure.natoms,) mask, with unique atoms
            marked True.
    """

    # First, assume all atoms are identical
    identical_ij = np.ones((structure.natoms, structure.natoms), dtype=bool)
    # The main diagonal does not contain 'duplicated' atoms
    identical_ij &= np.invert(np.identity(structure.natoms, dtype=bool))

    # Identify duplicated atoms by comparing selected properties.
    for attr in ("resi", "resn", "altloc", "icode", "chain", "name"):
        attrvec = getattr(structure, attr)
        ident_prop = attrvec[:, np.newaxis] == attrvec[np.newaxis, :]
        identical_ij &= ident_prop

    # We only care about the upper triangle
    #   (duplication will only mark atoms with higher index for removal)
    identical_ij = np.triu(identical_ij)

    # Logical-or rows together
    identical_i = np.any(identical_ij, axis=0)

    # Name atoms which are not unique
    if np.sum(identical_i) > 0:
        indices = tuple(*np.nonzero(identical_i))
        print(
            f"Atoms {indices} had earlier, identical atoms.\nThey are being removed."
        )

    # We are not concerned if amino acids have duplicate atoms
    is_amino_acid = np.frompyfunc(ROTAMERS.__contains__, 1, 1)
    amino_acids_i = is_amino_acid(structure.resn).astype(bool)
    identical_i &= np.invert(amino_acids_i)

    # Unique atoms are not identical
    return np.invert(identical_i)


def main():
    args = parse_args()
    os.makedirs(args.directory, exist_ok=True)
    structure = Structure.fromfile(args.structure).reorder()
    mask = find_unique_atoms(structure)
    new_structure = structure.get_selected_structure(mask)
    new_structure.tofile(args.structure + "_fixed.pdb")
