"""Strip alternate conformers using CCTBX"""
# XXX note that phenix.pdbtools remove_alt_confs=True will do the same thing

import argparse
import logging
import os
from string import ascii_uppercase
import sys
import time

from scitbx.array_family import flex
from iotbx.file_reader import any_file

from qfit import Structure
from qfit.structure import residue_type


def parse_args(argv=sys.argv):
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
    return p.parse_args(argv[1:])


def main(argv=sys.argv):
    args = parse_args(argv)
    os.makedirs(args.directory, exist_ok=True)
    pdb_file = any_file(args.structure)
    hierarchy = pdb_file.file_object.hierarchy
    hierarchy.remove_alt_confs(always_keep_one_conformer=True)
    pdb_atoms = hierarchy.atoms()
    occ = flex.double(len(pdb_atoms), 1.0)
    pdb_atoms.set_occ(occ)
    basename = os.path.basename(args.structure)
    pdb_out = os.path.join(args.directory, f"{basename[:-4]}.single.pdb")
    print(pdb_out)
    hierarchy.write_pdb_file(pdb_out,
        crystal_symmetry=pdb_file.file_object.crystal_symmetry())
    return 0
