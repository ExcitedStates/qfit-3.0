"""Automatically build a multiconformer residue"""

import argparse
import logging
import copy
import os
from string import ascii_uppercase
import sys
import time

import numpy as np

from qfit import Structure
from qfit.structure import residue_type


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
    p.add_argument(
        "-occ",
        "--occ_cutoff",
        type=float,
        default=0.01,
        metavar="<float>",
        help="Remove conformers with occupancies below occ_cutoff. Default = 0.01",
    )
    p.add_argument(
        "-rmsd", "--rmsd_cutoff", type=float, default=0.01, metavar="<float>"
    )
    args = p.parse_args()

    return args


def main():
    args = parse_args()
    os.makedirs(args.directory, exist_ok=True)
    output_file = os.path.join(args.directory, args.structure[:-4] + "_norm.pdb")

    structure = Structure.fromfile(args.structure).reorder()
    to_remove = []

    # Iterate over every residue...
    for chain in structure:
        for residue in chain:
            should_collapse = False
            if residue_type(residue) != ("aa-residue" and "rotamer-residue"):
                continue
            altlocs = list(set(residue.altloc))
            # Deal with the simplest case first: only a single conformer
            if len(altlocs) == 1:
                residue._q[residue._selection] = 1.0
                continue

            # Should we collapse the backbone for the current residue?
            if not "" in altlocs:
                for i, altloc1 in enumerate(altlocs):
                    conf1 = residue.extract("altloc", altloc1)
                    conf1 = conf1.extract("name", ("N", "CA", "C", "O"))
                    for altloc2 in altlocs[i + 1 :]:
                        conf2 = residue.extract("altloc", altloc2)
                        conf2 = conf2.extract("name", ("N", "CA", "C", "O"))
                        # If the conformer has occupancy greater than the cutoff
                        # and if it is not identical to all
                        if (
                            np.mean(np.linalg.norm(conf2.coor - conf1.coor, axis=1))
                            > 0.05
                        ) and (np.min(conf2.q) > args.occ_cutoff):
                            should_collapse = False

                # Add the atoms of the collapsed backbone to the to_remove list
                # and fix altloc and occupancy of the backbone
                if should_collapse:
                    print("collapse!")
                    conf1._q[conf1._selection] = 1.0
                    conf1._altloc[conf1._selection] = ""
                    for altloc2 in altlocs[1:]:
                        conf2 = residue.extract("altloc", altloc2)
                        conf2 = conf2.extract("name", ("N", "CA", "C", "O"))
                        [to_remove.append(x) for x in conf2._selection]
                    print(to_remove)
                    conf1.tofile(str(residue.chain[0]) + str(residue.resi[0]) + ".pdb")

            # If the backbone is collapsed, we can remove identical side chain conformers
            # or side chain conformers that fall below the occupancy cutoff:
            if residue.resn[0] != "GLY" and (should_collapse or ("" in altlocs)):
                for i, altloc1 in enumerate(altlocs):
                    if altloc1 == "":
                        continue
                    conf1 = residue.extract("altloc", altloc1)
                    conf1 = conf1.extract("name", ("N", "CA", "C", "O"), "!=")
                    if np.min(conf1.q) < args.occ_cutoff:
                        [to_remove.append(x) for x in conf1._selection]
                        continue
                    for altloc2 in altlocs[i + 1 :]:
                        conf2 = residue.extract("altloc", altloc2)
                        conf2 = conf2.extract("name", ("N", "CA", "C", "O"), "!=")
                        if conf1.rmsd(conf2) < args.rmsd_cutoff:
                            [to_remove.append(x) for x in conf2._selection]
                """ try:
                    structure._altloc[conf._selection] = ''
                except:
                    pass """
            # Now, to the case where the backbone is not collapsed
            else:
                # Here, we only want to remove if ALL conformers are identical or below
                # occupancy cutoff
                is_identical = True
                for i, altloc1 in enumerate(altlocs):
                    if not is_identical:
                        break
                    conf1 = residue.extract("altloc", altloc1)
                    conf1.tofile(
                        str(residue.chain[0]) + str(residue.resi[0]) + "_conf1.pdb"
                    )
                    for altloc2 in altlocs[i + 1 :]:
                        conf2 = residue.extract("altloc", altloc2)
                        conf2.tofile(
                            str(residue.chain[0]) + str(residue.resi[0]) + "_conf2.pdb"
                        )
                        # If the conformer has occupancy greater than the cutoff
                        # and if it is not identical to all
                        if (
                            (np.min(conf2.q) > args.occ_cutoff)
                            and (np.min(conf1.q) > args.occ_cutoff)
                            and (conf1.rmsd(conf2) > args.rmsd_cutoff)
                        ):
                            is_identical = False
                            break
                # If all conformers converged (either because of RMSD or occupancy)
                # keep one occupancy > args.occ_cutoff
                found_unique_conf = False
                if is_identical:
                    for altloc1 in altlocs:
                        conf1 = residue.extract("altloc", altloc1)
                        if np.min(conf1.q) > args.occ_cutoff and found_unique_conf:
                            [to_remove.append(x) for x in conf1._selection]
                            found_unique_conf = True

                # If the occupancy of the conformer fell below the cutoff...
                for altloc in altlocs:
                    conf = residue.extract("altloc", altloc)
                    if np.min(conf.q) < args.occ_cutoff:
                        [to_remove.append(x) for x in conf._selection]

    # Remove conformers in to_remove list:
    mask = structure.active
    mask[to_remove] = False
    data = {}
    for attr in structure.data:
        data[attr] = getattr(structure, attr).copy()[mask]
    structure = Structure(data).reorder()
    # for chain in structure:
    #   for residue in chain:
    #       print (residue.resi[0])

    # Normalize occupancies and fix altlocs:
    for chain in structure:
        for residue in chain:
            altlocs = list(set(residue.altloc))
            try:
                altlocs.remove("")
            except ValueError:
                pass
            naltlocs = len(altlocs)
            if naltlocs < 2:
                residue._q[residue._selection] = 1.0
                residue._altloc[residue._selection] = ""
            else:
                conf = residue.extract("altloc", altlocs)
                natoms = len(residue.extract("altloc", altlocs[-1]).name)
                factor = natoms / np.sum(conf.q)
                residue._q[conf._selection] *= factor
    structure.tofile(output_file)
    print(len(to_remove))
