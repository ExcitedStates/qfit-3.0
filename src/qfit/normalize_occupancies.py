'''
Excited States software: qFit 3.0

Contributors: Saulo H. P. de Oliveira, Gydo van Zundert, and Henry van den Bedem.
Contact: vdbedem@stanford.edu

Copyright (C) 2009-2019 Stanford University
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

This entire text, including the above copyright notice and this permission notice
shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
'''

import numpy as np
import argparse
import logging
import copy
import os
import sys
import time
from string import ascii_uppercase
from . import Structure
from .structure import residue_type


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("structure", type=str,
                   help="PDB-file containing structure.")

    # Output options
    p.add_argument("-d", "--directory", type=os.path.abspath, default='.',
                   metavar="<dir>", help="Directory to store results.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Be verbose.")
    p.add_argument('-occ', "--occ_cutoff", type=float, default=0.10, metavar="<float>",
                   help="Remove conformers with occupancies below occ_cutoff. (default: 0.10)")
    args = p.parse_args()

    return args


def redistribute_occupancies_by_atom(residue, cutoff):
    # Create a map of atomname → occupancies
    atom_occs = dict()
    for (name, altloc, atomidx, q) in zip(residue.name, residue.altloc, residue._selection, residue.q):
        if name not in atom_occs:
            atom_occs[name] = list()
        atom_occs[name].append((altloc, atomidx, q))

    # For each atomname:
    for name, alt_id_q_list in atom_occs.items():
        # If any of the qs are less than cutoff
        if any(q < cutoff for (alt, atomidx, q) in alt_id_q_list):
            # Which confs are either side of the cutoff?
            confs_low = [alt for (alt, atomidx, q) in alt_id_q_list
                             if q < cutoff]
            confs_high = [alt for (alt, atomidx, q) in alt_id_q_list
                              if alt not in confs_low]

            # Describe occupancy redistribution intentions
            print(f"{residue.parent.id}/{residue.resn[-1]}{''.join(map(str, residue.id))}/{name}")
            print(f"  {[(alt, q) for (alt, atomidx, q) in alt_id_q_list if alt in confs_low]} "
                  f"→ {[(alt, q) for (alt, atomidx, q) in alt_id_q_list if alt in confs_high]}")

            # Redistribute occupancy
            if len(confs_high) == 1:
                for (alt, atomidx, q) in alt_id_q_list:
                    if alt in confs_high:
                        residue._q[atomidx] = 1.0
                        residue._altloc = ""
            else:
                sum_q_high = sum(q for (alt, atomidx, q) in alt_id_q_list if alt in confs_high)
                for (alt_high, atomidx_high, q_high) in alt_id_q_list:
                    if alt_high in confs_high:
                        for (alt_low, atomidx_low, q_low) in alt_id_q_list:
                            if alt_low in confs_low:
                                residue._q[atomidx_high] += q_low * q_high / sum_q_high

            # Describe occupancy redistribution results
            print(f"  ==> {[(alt, residue._q[atomidx]) for (alt, atomidx, q) in alt_id_q_list if alt in confs_high]}")


def redistribute_occupancies_by_residue(residue, cutoff):
    altconfs = dict((agroup.id[1], agroup) for agroup in residue.atom_groups
                                           if agroup.id[1] != "")

    # Which confs are either side of the cutoff?
    confs_low = [alt for (alt, altconf) in altconfs.items()
                     if altconf.q[-1] < cutoff]
    confs_high = [alt for alt in altconfs.keys()
                      if alt not in confs_low]

    # Describe occupancy redistribution intentions
    print(f"{residue.parent.id}/{residue.resn[-1]}{''.join(map(str, residue.id))}")
    print(f"  {[(alt, altconfs[alt].q[-1]) for alt in confs_low]} "
          f"→ {[(alt, altconfs[alt].q[-1]) for alt in confs_high]}")

    # Redistribute occupancy
    if len(confs_high) == 1:
        altconfs[confs_high[0]].q = 1.0
        altconfs[confs_high[0]].altloc = ""
    else:
        sum_q_high = sum(altconfs[target].q[-1] for target in confs_high)
        for target in confs_high:
            q_high = altconfs[target].q[-1]
            for source in confs_low:
                q_low = altconfs[source].q[-1]
                altconfs[target].q += q_low * q_high / sum_q_high

    # Describe occupancy redistribution results
    print(f"  ==> {[(alt, altconfs[alt].q[-1]) for alt in confs_high]}")


def main():
    args = parse_args()
    try:
        os.makedirs(args.directory)
        output_file = os.path.join(args.directory,
                                   args.structure[:-4] + "_norm.pdb")
    except OSError:
        output_file = args.structure[:-4] + "_norm.pdb"

    structure = Structure.fromfile(args.structure)

    # Capture LINK records
    link_data = structure.link_data

    # Which atoms fall below cutoff?
    mask = structure.q < args.occ_cutoff
    n_removed = np.sum(mask)

    # Loop through structure, redistributing occupancy from altconfs below cutoff to above cutoff
    for chain in structure:
        for residue in chain:
            if np.any(residue.q < args.occ_cutoff):
                # How many occupancy-values can we find in each altconf?
                occs_per_alt = [np.unique(agroup.q).size for agroup in residue.atom_groups
                                                         if agroup.id[1] != ""]
                if occs_per_alt.count(1) == len(occs_per_alt):
                    redistribute_occupancies_by_residue(residue, args.occ_cutoff)
                else:
                    redistribute_occupancies_by_atom(residue, args.occ_cutoff)

    # Create structure without low occupancy confs
    data = {}
    for attr in structure.data:
        data[attr] = getattr(structure, attr).copy()[~mask]
    structure = Structure(data).reorder()

    # Reattach LINK records
    structure.link_data = link_data

    # Save structure
    structure.tofile(output_file)

    print(f"normalize_occupancies: {n_removed} atoms had occ < {args.occ_cutoff} and were removed.")
    print(n_removed)  # for post_refine_phenix.sh
