import numpy as np
import argparse
import logging
import os
from collections import namedtuple
from . import Structure
from .structure.rotamers import ROTAMERS
from string import ascii_uppercase


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
        default=0.10,
        metavar="<float>",
        help="Remove conformers with occupancies below occ_cutoff. (default: 0.10)",
    )
    args = p.parse_args()

    return args

def redistribute_array(arr):
    total_sum = np.sum(arr)

    # Using numpy's round function
    scaled_arr = [np.round(x / total_sum, 2) for x in arr]

    new_sum = np.sum(scaled_arr)
    if new_sum < 1:
        diff = 1 - new_sum
        max_index = np.argmax(scaled_arr)
        scaled_arr[max_index] += diff

    return scaled_arr

def remove_redistribute_conformer(residue, remove, keep):
    """Redistributes occupancy from altconfs below cutoff to above cutoff.

    This function iterates over residue.atom_groups (i.e. altconfs).
    It should only be used when each of these atom_groups have uniform occupancy.
    Otherwise, use `redistribute_occupancies_by_atom`.

    Atoms below cutoff should be culled after this function is run.

    Args:
        residue: residue to perform occupancy
            redistribution on
        remove: altlocs that fall below cutoff value
        keep: altlocs that are above cutoff value
    """
    total_occ_redist = np.sum(np.unique(residue.extract("altloc", remove).q))
    residue_out = residue.extract("altloc", keep)

    if len(set(residue_out.altloc)) == 1:  # if only one altloc left
        residue_out.q = 1.0
        residue_out.altloc = ""  # adjusting altloc label

    else:
        naltlocs = len(
            np.unique(residue_out.extract("q", 1.0, "!=").altloc)
        )  # number of altlocs left
        occ_redist = round(total_occ_redist / naltlocs, 2)
        add_occ_redist = 0

        if (
            (occ_redist * naltlocs)
            + np.sum(np.unique(residue_out.extract("q", 1.0, "!=").q))
        ) != 1.0:  # make sure that rounding and distributing is still adding to 1
            additional_occ_redist = round(
                1.0
                - (
                    (occ_redist * naltlocs)
                    + np.sum(np.unique(residue_out.extract("q", 1.0, "!=").q))
                ),
                2,
            )
        if abs(additional_occ_redist) > 0.01:
            print(
                "Additional occupancy redistribution is too large. Check input occupancies"
            )
            return
        add_occ = False
        for n, alt in enumerate(
            np.unique(residue_out.altloc)
        ):  # redistribute occupancies
            if np.all(
                residue_out.extract("altloc", alt, "==").q < 1.0
            ):  # ignoring backbone atoms with full occ
                if add_occ == False:
                    residue_out.extract("altloc", alt, "==").q = (
                        residue_out.extract("altloc", alt, "==").q
                        + occ_redist
                        + additional_occ_redist
                    )
                    add_occ = True
                else:
                    residue_out.extract("altloc", alt, "==").q = (
                        residue_out.extract("altloc", alt, "==").q + occ_redist
                    )
                residue_out.altloc = ascii_uppercase[n]

    return residue_out


def redistribute_occupancies_by_atom(residue, cutoff):
    """Redistributes occupancy from atoms below cutoff to above cutoff.

    This function iterates over atoms, grouping atoms with the same name.

    Atoms below cutoff should be culled after this function is run.

    Args:
        residue (qfit.structure._ResidueGroup): residue to perform occupancy
            redistribution on by iterating over atoms
        cutoff (float): occupancy threshold
    """
    # Create AltAtom struct
    AltAtom = namedtuple("AltAtom", ["altloc", "atomidx", "q"])

    # Create a map of atomname → occupancies
    atom_occs = dict()
    for name, altloc, atomidx, q in zip(
        residue.name, residue.altloc, residue._selection, residue.q
    ):
        if name not in atom_occs:
            atom_occs[name] = list()
        atom_occs[name].append(AltAtom(altloc, atomidx, q))

    # For each atomname:
    for name, altatom_list in atom_occs.items():
        # If any of the qs are less than cutoff
        if any(atom.q < cutoff for atom in altatom_list):
            # Which confs are either side of the cutoff?
            confs_low = [atom for atom in altatom_list if atom.q < cutoff]
            confs_high = [atom for atom in altatom_list if atom.q >= cutoff]

            # Describe occupancy redistribution intentions
            print(
                f"{residue.parent.id}/{residue.resn[-1]}{''.join(map(str, residue.id))}/{name}"
            )
            print(
                f"  {[(atom.altloc, round(atom.q, 2)) for atom in confs_low]} "
                f"→ {[(atom.altloc, round(atom.q, 2)) for atom in confs_high]}"
            )

            # Redistribute occupancy
            if len(confs_high) == 1:
                residue._q[confs_high[0].atomidx] = 1.0
                residue._altloc[confs_high[0].atomidx] = ""
            else:
                sum_q_high = sum(atom.q for atom in confs_high)
                for atom_high in confs_high:
                    q_high = atom_high.q
                    for atom_low in confs_low:
                        q_low = atom_low.q
                        residue._q[atom_high.atomidx] += q_low * q_high / sum_q_high

            # Describe occupancy redistribution results
            print(
                f"  ==> {[(atom.altloc, round(residue._q[atom.atomidx], 2)) for atom in confs_high]}"
            )


def redistribute_occupancies_by_residue(residue, cutoff):
    """Redistributes occupancy from altconfs below cutoff to above cutoff.

    This function iterates over residue.atom_groups (i.e. altconfs).
    It should only be used when each of these atom_groups have uniform occupancy.
    Otherwise, use `redistribute_occupancies_by_atom`.

    Atoms below cutoff should be culled after this function is run.

    Args:
        residue (qfit.structure._ResidueGroup): residue to perform occupancy
            redistribution on
        cutoff (float): occupancy threshold
    """
    altconfs = dict(
        (agroup.id[1], agroup) for agroup in residue.atom_groups if agroup.id[1] != ""
    )

    # Which confs are either side of the cutoff?
    confs_low = [alt for (alt, altconf) in altconfs.items() if altconf.q[-1] < cutoff]
    confs_high = [alt for alt in altconfs.keys() if alt not in confs_low]

    # Describe occupancy redistribution intentions
    print(f"{residue.parent.id}/{residue.resn[-1]}{''.join(map(str, residue.id))}")
    print(
        f"  {[(alt, round(altconfs[alt].q[-1], 2)) for alt in confs_low]} "
        f"→ {[(alt, round(altconfs[alt].q[-1], 2)) for alt in confs_high]}"
    )

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
    print(f"  ==> {[(alt, round(altconfs[alt].q[-1], 2)) for alt in confs_high]}")


def main():
    args = parse_args()
    try:
        os.makedirs(args.directory)
        output_file = os.path.join(args.directory, args.structure[:-4] + "_norm.pdb")
    except OSError:
        output_file = args.structure[:-4] + "_norm.pdb"

    structure = Structure.fromfile(args.structure)

    # seperate het versus atom (het allowed to have <1 occ)
    water = structure.extract("resn", "HOH", "==")
    structure = structure.extract("resn", "HOH", "!=")

    # Capture LINK records
    link_data = structure.link_data

    # Which atoms fall below cutoff?
    mask = structure.q < args.occ_cutoff
    n_removed = np.sum(mask)

    # Get list of all non-hetatom residue
    n_removed = 0  # keep track of the residues we are removing
    # Loop through structure, redistributing occupancy from altconfs below cutoff to above cutoff
    for chain in structure:
        for residue in chain:
            if np.any(residue.q < args.occ_cutoff):
                # How many occupancy-values can we find in each altconf?
                occs_per_alt = [
                    np.unique(agroup.q).size
                    for agroup in residue.atom_groups
                    if agroup.id[1] != ""
                ]
                redistribute_occupancies_by_residue(residue, args.occ_cutoff)
                n_removed += 1

    # Create structure without low occupancy confs (culling)
    data = {}
    for attr in structure.data:
        data[attr] = getattr(structure, attr).copy()[~mask]
    structure = Structure(data).reorder()

    # add het atoms back in
    structure = structure.combine(water)
    # Reattach LINK records
    structure.link_data = link_data

    # output structure
    structure.tofile(output_file)

    print(
        f"normalize_occupancies: {n_removed} atoms had occ < {args.occ_cutoff} and were removed."
    )
    print(n_removed)  # for post_refine_phenix.sh
