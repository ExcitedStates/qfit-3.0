import argparse
from collections import namedtuple
import os
from string import ascii_uppercase
import numpy as np
import traceback

from qfit.structure.math import calc_rmsd
from qfit.structure.rotamers import ROTAMERS
from qfit.structure.residue import RotamerResidue
from qfit import Structure


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("structure", type=str, help="PDB-file containing structure.")
    p.add_argument(
        "--run_rmsd",
        action="store_true",
        help="Option to run RMSD-based collapse of conformers."
    )
    p.add_argument(
        "--run_rotamer",
        action="store_true", 
        help="Option to run rotamer-based collapse of conformers."
    )
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
    p.add_argument(
        "--rmsd",
        type=float,
        default=0.20,
        metavar="<float>",
        help="RMSD threshold (Å) to collapse same-residue altlocs. (default: 0.20 Å)",
    )
    p.add_argument(
        "--angle_tol",
        type=float,
        default=15.0,
        metavar="<float>",
        help="Angular tolerance (degrees) for rotamer comparison. (default: 15.0°)",
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
        additional_occ_redist = 0
        naltlocs = len(
            np.unique(residue_out.extract("q", 1.0, "!=").altloc)
        )  # number of altlocs left
        occ_redist = round(total_occ_redist / naltlocs, 2)

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
        residue (qfit.structure.ResidueGroup): residue to perform occupancy
            redistribution on by iterating over atoms
        cutoff (float): occupancy threshold
    """
    # Create AltAtom struct
    AltAtom = namedtuple("AltAtom", ["altloc", "atomidx", "q"])

    # Create a map of atomname → occupancies
    atom_occs = dict()
    for name, altloc, atomidx, q in zip(
        residue.name, residue.altloc, residue.selection, residue.q
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
            # FIXME no private member access!
            if len(confs_high) == 1:
                residue._q[confs_high[0].atomidx] = 1.0  # pylint: disable=protected-access
                residue._altloc[confs_high[0].atomidx] = ""  # pylint: disable=protected-access
            else:
                sum_q_high = sum(atom.q for atom in confs_high)
                for atom_high in confs_high:
                    q_high = atom_high.q
                    for atom_low in confs_low:
                        q_low = atom_low.q
                        residue._q[atom_high.atomidx] += q_low * q_high / sum_q_high  # pylint: disable=protected-access

            # Describe occupancy redistribution results
            results = [(atom.altloc, round(residue._q[atom.atomidx], 2)) for atom in confs_high]  # pylint: disable=protected-access
            print(f"  ==> {results}")


def redistribute_occupancies_by_residue(residue, cutoff):
    """Redistributes occupancy from altconfs below cutoff to above cutoff.

    This function iterates over residue.atom_groups (i.e. altconfs).
    It should only be used when each of these atom_groups have uniform occupancy.
    Otherwise, use `redistribute_occupancies_by_atom`.

    Atoms below cutoff should be culled after this function is run.

    Args:
        residue (qfit.structure.ResidueGroup): residue to perform occupancy
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
    #print(f"{residue.parent.id}/{residue.resn[-1]}{''.join(map(str, residue.id))}")
    if confs_low:
      print(
        f"  {[(alt, round(altconfs[alt].q[-1], 2)) for alt in confs_low]} "
        f"→ {[(alt, round(altconfs[alt].q[-1], 2)) for alt in confs_high]}"
      )

    if len(altconfs) == 0:
       return residue

    # Redistribute occupancy
    if len(confs_high) == 1:
        altconfs[confs_high[0]].q = 1.0
        altconfs[confs_high[0]].altloc = ""
    else:
        q_high = [altconfs[target].q[-1] for target in confs_high]
        redistributed_occupancies = redistribute_array(q_high)

        # Reassign occupancies to conformers
        for n, target in enumerate(confs_high):
            altconfs[target].q = redistributed_occupancies[n]

    return residue

def collapse_conformers_by_rmsd(residue, rmsd_cutoff):
    """
    Collapse alternate conformers within a residue if their Cartesian RMSD is
    below the provided cutoff. Uses the first conformer as reference and
    collapses others onto it, redistributing occupancies like
    redistribute_occupancies_by_residue.

    Hydrogens are excluded from RMSD calculations.

    Args:
        residue (qfit.structure.ResidueGroup): Residue to process
        rmsd_cutoff (float): RMSD threshold in Å
    """
    # Gather alt conformers (keep original altconfs, we'll filter H during RMSD calc)
    altconfs = dict(
        (agroup.id[1], agroup) 
        for agroup in residue.atom_groups 
        if agroup.id[1] != ""
    )
    
    # Filter out altconfs that are only hydrogen or a single atom
    altconfs = {
        alt: agroup for alt, agroup in altconfs.items()
        if agroup.extract("e", "H", "!=").natoms > 1
    }
    if len(altconfs) <= 1:
        return residue

    # Compute RMSD of each alt to the first one as reference (excluding hydrogens)
    altconf_keys = list(altconfs.keys())
    
    ref_alt = altconf_keys[0]
    # Filter out hydrogens for RMSD calculation
    ref_heavy = altconfs[ref_alt].extract("e", "H", "!=")
    ref_coords = ref_heavy.coor

    to_merge = []
    for alt in altconf_keys[1:]:
        # Filter out hydrogens for RMSD calculation
        alt_heavy = altconfs[alt].extract("e", "H", "!=")
        coords = alt_heavy.coor
        # ensure same shape by intersecting atom names
        # (qfit ensures consistent atom ordering within residue altconfs)
        rmsd = calc_rmsd(coords, ref_coords)
        if rmsd < rmsd_cutoff:
            to_merge.append(alt)

    # If nothing to merge, exit
    if not to_merge:
        return residue

    # Redistribute occupancies: sum merged occupancies into the representative
    q_total = altconfs[ref_alt].q[-1] + sum(altconfs[alt].q[-1] for alt in to_merge)
    altconfs[ref_alt].q = q_total
    if len(altconfs) - len(to_merge) == 1:  # if only one altloc left
        altconfs[ref_alt].q = 1.0
        altconfs[ref_alt].altloc = ""  # adjusting altloc label
    else:
        altconfs[ref_alt].altloc = ref_alt

    # Normalize occupancies across the remaining conformers (like redistribute_occupancies_by_residue)
    remaining = [alt for alt in residue.altloc if alt != ""]
    if len(remaining) == 0:
        # single conformer left
        residue.atom_groups[0].q = 1.0
        residue.atom_groups[0].altloc = ""
    else:
        q_vals = [agroup.q[-1] for agroup in residue.atom_groups if agroup.id[1] != ""]
        q_norm = np.array(q_vals) / np.sum(q_vals)
        for agroup, q in zip(
            [ag for ag in residue.atom_groups if ag.id[1] != ""], q_norm
        ):
            agroup.q = q

    return residue


def circ_diff_deg(a, b):
    """Calculate smallest absolute angular difference in degrees."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def collapse_conformers_by_rotamer(residue, angle_tol=15.0):
    """
    Collapse alternate conformers within a residue if they have the same rotamer state.
    Two conformers are considered to have the same rotamer if all their chi angles
    match within the specified angular tolerance.

    This revised version also ensures all residue occupancies are normalized to 1.0,
    regardless of whether any conformers were collapsed.

    Args:
        residue (qfit.structure.ResidueGroup): Residue to process
        angle_tol (float): Angular tolerance in degrees for chi angle comparison (default: 15.0)

    Returns:
        residue: Modified residue with collapsed and normalized conformers
    """
    # Get residue name and check if it has rotamer information
    resn = residue.resn[0]
    res_id = f"{residue.resn[-1]}{''.join(map(str, residue.id))}"

    if resn not in ROTAMERS or ROTAMERS[resn].get("nchi", 0) == 0:
        if resn in ROTAMERS:
          q_vals = [agroup.q[-1] for agroup in residue.atom_groups if agroup.q[-1] < 1.0]
          q_sum = np.sum(q_vals)
          if q_sum != 1.0: 
             q_norm = np.array(q_vals) / q_sum

             norm_index = 0
             for agroup in residue.atom_groups:
                # Only apply normalization to partial conformers
                if agroup.q[-1] < 1.0 and norm_index < len(q_norm):
                    agroup.q = q_norm[norm_index]
                    norm_index += 1

          elif len(residue.atom_groups) == 1 and residue.atom_groups[0].q[-1] != 1.0:
             # Special case: only one conformer left, but its Q isn't 1.0 yet (e.g., if collapse left 1 but didn't set to 1.0)
             residue.atom_groups[0].q = 1.0
             residue.atom_groups[0].altloc = "" # Ensure it's marked as the sole conformer
        return residue

    # Gather alt conformers
    altconfs = dict(
        (agroup.id[1], agroup)
        for agroup in residue.atom_groups
        if agroup.id[1] != ""
    )

    if len(altconfs) <= 1:
        return residue

    chi_by_altloc = {}

    # Get parent chain, then parent structure
    if residue.parent is None:
        pass
    else:
        chain = residue.parent
        if chain.parent is not None:
            structure = chain.parent
            resi, icode = residue.id
            chain_id = chain.id

            for alt in altconfs.keys():
                if alt == '':
                    continue
                try:
                    # Selection logic to get combined altloc and backbone atoms
                    sel = structure.extract("chain", chain_id, "==").extract("resi", resi, "==")
                    if icode: sel = sel.extract("icode", icode, "==")
                    alt_sel = sel.extract("altloc", alt, "==")
                    bb_sel = sel.extract("altloc", "", "==")
                    combined = alt_sel.combine(bb_sel) if bb_sel.natoms > 0 else alt_sel

                    residues = list(combined.single_conformer_residues)
                    if not residues: continue
                    res = residues[0]

                except Exception:
                    traceback.print_exc()
                    continue

                nchi = getattr(res, "nchi", 0)
                if nchi < 1: continue

                # Get chi angles
                angles = []
                for i in range(1, nchi + 1):
                    try:
                        angle = res.get_chi(i)
                        if angle is not None: angles.append(float(angle))
                    except Exception as e:
                        # print(f"[DEBUG] {res_id} altloc '{alt}': Could not get chi {i}: {e}")
                        continue

                if angles:
                    chi_by_altloc[alt] = angles
                # else:
                #    print(f"[DEBUG] {res_id} altloc '{alt}': No valid chi angles found")


    
    to_merge = []
    altloc_list = list(chi_by_altloc.keys())
    ref_alt = altloc_list[0]
    ref_angles = chi_by_altloc[ref_alt]

    # Compare each additional conformer to baseline
    for alt in altloc_list[1:]:
        alt_angles = chi_by_altloc[alt]

        if len(alt_angles) != len(ref_angles): continue
        # Compare all chi angles
        diffs = [circ_diff_deg(ref_chi, alt_chi) for ref_chi, alt_chi in zip(ref_angles, alt_angles)]
        all_match = all(diff <= angle_tol for diff in diffs)

        if all_match:
            to_merge.append(alt)

    # If merge is needed, perform occupancy sum
    if to_merge:
        # Redistribute occupancies: sum merged occupancies into the representative
        q_total = altconfs[ref_alt].q[-1] + sum(altconfs[alt].q[-1] for alt in to_merge)
        altconfs[ref_alt].q = q_total
        if len(altconfs) - len(to_merge) == 1:
            altconfs[ref_alt].q = 1.0
            altconfs[ref_alt].altloc = ""
        else:
            altconfs[ref_alt].altloc = ref_alt

    q_vals = [agroup.q[-1] for agroup in residue.atom_groups if agroup.q[-1] < 1.0]

    # If there are Q-values to normalize (i.e., multiple partial conformers exist)
    if q_vals and len(q_vals) > 1:
        q_sum = np.sum(q_vals)
        if q_sum != 1.0:
            q_norm = np.array(q_vals) / q_sum

            # 2. Re-assign the normalized Q-values
            norm_index = 0
            for agroup in residue.atom_groups:
                # Only apply normalization to partial conformers
                if agroup.q[-1] < 1.0 and norm_index < len(q_norm):
                    agroup.q = q_norm[norm_index]
                    norm_index += 1

    elif len(residue.atom_groups) == 1 and residue.atom_groups[0].q[-1] != 1.0:
        # Special case: only one conformer left, but its Q isn't 1.0 yet (e.g., if collapse left 1 but didn't set to 1.0)
        residue.atom_groups[0].q = 1.0
        residue.atom_groups[0].altloc = "" # Ensure it's marked as the sole conformer


    return residue

def main():
    args = parse_args()
    os.makedirs(args.directory, exist_ok=True)
    output_file = os.path.join(args.directory, args.structure[:-4] + "_norm.pdb")

    structure = Structure.fromfile(args.structure)

    # seperate het versus atom (het allowed to have <1 occ)
    hetatms = structure.extract("record", "HETATM", "==")
    structure = structure.extract("record", "HETATM", "!=")

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
            if args.run_rmsd:
                collapse_conformers_by_rmsd(residue, args.rmsd)
            elif args.run_rotamer:
                collapse_conformers_by_rotamer(residue, args.angle_tol)
            elif np.any(residue.q < args.occ_cutoff):
                redistribute_occupancies_by_residue(residue, args.occ_cutoff)
                n_removed += 1

    # Create structure without low occupancy confs (culling)
    structure = structure.copy().get_selected_structure(~mask).reorder()

    # add het atoms back in
    structure = structure.combine(hetatms)
    # Reattach LINK records
    structure.link_data = link_data

    # output structure
    structure.tofile(output_file)

    print(
        f"normalize_occupancies: {n_removed} atoms had occ < {args.occ_cutoff} and were removed."
    )
    print(n_removed)  # for post_refine_phenix.sh

