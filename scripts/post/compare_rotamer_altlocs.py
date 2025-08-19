#!/usr/bin/env python
import argparse
import os
import re
import pandas as pd
import numpy as np
from qfit.structure import Structure

ANGLE_TOL = 15.0  # degrees

def four_char_id_from_path(path: str) -> str:
    """Best-effort 4-char PDB-like ID from filename/stem."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r'([0-9A-Za-z]{4})$', stem)
    return m.group(1).upper() if m else stem[:8]

def build_argparser():
    p = argparse.ArgumentParser(
        description="Compare alt locs and rotamer states between two PDB files."
    )
    p.add_argument("base_PDB", type=str, help="Base PDB file.")
    p.add_argument("comp_PDB", type=str, help="Comparison PDB file.")
    p.add_argument("--base_pdb_id", type=str, default=None,
                   help="PDB ID label for the base PDB file.")
    p.add_argument("--comp_pdb_id", type=str, default=None,
                   help="PDB ID label for the comparison PDB file.")
    p.add_argument("--directory", type=str, default="", 
                   help="Where to save RSCC info")
    return p

def circ_diff_deg(a: float, b: float) -> float:
    """Smallest absolute angular difference in degrees."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)

def backbone_main_altloc(residue_struct: Structure) -> Structure:
    """Backbone with main altloc ('') as a Structure selection."""
    bb = residue_struct.extract("name", "N", "==").combine(
        residue_struct.extract("name", "CA", "==")
    ).combine(
        residue_struct.extract("name", "C", "==")
    ).combine(
        residue_struct.extract("name", "O", "==")
    )
    return bb.extract("altloc", "", "==")

def residue_rotamers(struct_chain: Structure, resi) -> list[tuple[int, str, float]]:
    """
    Return list of (chi_index, altloc, angle_degrees) for a residue.
    Combines each altloc with main-backbone atoms to stabilize geometry.
    """
    out = []
    rsel = struct_chain.extract("resi", resi, "==")
    if not rsel.chains:  # Check if there are any chains
        return out
    bb = backbone_main_altloc(rsel)

    for alt in np.unique(rsel.altloc):
        # Keep explicit altlocs and also allow '' (some residues only have '')
        alt_sel = rsel.extract("altloc", alt, "==")
        if not alt_sel.chains:  # Check if there are any chains
            continue
        try:
            combined = alt_sel.combine(bb)
            residues = list(combined.single_conformer_residues)
            if not residues:
                continue
            res = residues[0]
        except Exception:
            continue

        if getattr(res, "nchi", 0) < 1:
            continue

        for i in range(1, res.nchi + 1):
            # Some residues (e.g., Ser/Thr) have <4 atoms for some chis; guard via library length if present
            try:
                angle = res.get_chi(i)
            except Exception:
                continue
            if angle is None:
                continue
            out.append((i, alt, float(angle)))
    return out

def classify_rotamer_sets(r1: list[tuple[int,str,float]], r2: list[tuple[int,str,float]]) -> str:
    """
    Heuristic:
      - no chis in either -> 'no_chi'
      - any shared chi index with <= TOL difference across any altloc pairing -> 'shared'
      - all matching chi indices differ by > TOL -> 'different'
      - if all present and all <= TOL -> 'same'
    """
    if not r1 and not r2:
        return "no_chi"
    if not r1 or not r2:
        return "different"

    # Organize by chi index
    from collections import defaultdict
    by_idx1 = defaultdict(list)
    by_idx2 = defaultdict(list)
    for idx, alt, ang in r1:
        by_idx1[idx].append(ang)
    for idx, alt, ang in r2:
        by_idx2[idx].append(ang)

    shared_any = False
    same_all = True
    compared_any = False

    for idx in sorted(set(by_idx1.keys()) & set(by_idx2.keys())):
        compared_any = True
        # Compare all-vs-all across altlocs for this chi index; accept best match
        best = min(circ_diff_deg(a, b) for a in by_idx1[idx] for b in by_idx2[idx])
        if best <= ANGLE_TOL:
            shared_any = True
        else:
            same_all = False

    if not compared_any:
        # No overlapping chi indices
        return "different"
    if same_all and shared_any and len(by_idx1) == len(by_idx2):
        return "same"
    if shared_any:
        return "shared"
    return "different"

def compare_rotamers(structure1: Structure, structure2: Structure):
    rows = []

    chains1 = list(np.unique(structure1.chain))
    chains2 = set(np.unique(structure2.chain))
    # Keep stable order from structure1; only compare common chains
    common_chains = [c for c in chains1 if c in chains2]

    for chain in common_chains:
        c1 = structure1.extract("chain", chain, "==")
        c2 = structure2.extract("chain", chain, "==")

        # Residue IDs assumed to match numerically; intersect to be safe
        res1 = set(np.unique(c1.resi))
        res2 = set(np.unique(c2.resi))
        for resi in sorted(res1 & res2, key=lambda x: (int(x) if str(x).isdigit() else x)):
            resn = c1.extract("resi", resi, "==").resn[0]
            rset1 = residue_rotamers(c1, resi)
            rset2 = residue_rotamers(c2, resi)
            classification = classify_rotamer_sets(rset1, rset2)
            rows.append({
                "chain": chain,
                "residue": resi,
                "residue_name": resn,
                "classification": classification,
                "nchis_base": len({i for i,_,_ in rset1}),
                "nchis_comp": len({i for i,_,_ in rset2}),
                "nrotamers_base": len(rset1),
                "nrotamers_comp": len(rset2),
            })
    return rows

def main():
    options = build_argparser().parse_args()
    base_id = options.base_pdb_id or four_char_id_from_path(options.base_PDB)
    comp_id = options.comp_pdb_id or four_char_id_from_path(options.comp_PDB)

    # Load and drop waters
    base = Structure.fromfile(options.base_PDB).extract("resn", "HOH", "!=")
    comp = Structure.fromfile(options.comp_PDB).extract("resn", "HOH", "!=")

    # --- Altloc counts per residue ---
    alt_rows = []
    for chain in sorted(set(base.chain) & set(comp.chain)):
        b_chain = base.extract("chain", chain, "==")
        c_chain = comp.extract("chain", chain, "==")
        for resi in sorted(set(np.unique(b_chain.resi)) & set(np.unique(c_chain.resi)),
                           key=lambda x: (int(x) if str(x).isdigit() else x)):
            b_res = b_chain.extract("resi", resi, "==")
            c_res = c_chain.extract("resi", resi, "==")
            resn = b_res.resn[0]
            alt_rows.append({
                "chain": chain,
                "residue": resi,
                "residue_name": resn,
                f"altloc_{base_id}": len(set(b_res.altloc)),
                f"altloc_{comp_id}": len(set(c_res.altloc)),
            })

    alt_df = pd.DataFrame(alt_rows)
    alt_df.to_csv(f"{options.directory}{base_id}_{comp_id}_altloc_differences.csv", index=False)

    # --- Rotamer comparisons ---
    rot_rows = compare_rotamers(base, comp)
    rot_df = pd.DataFrame(rot_rows)
    rot_df.to_csv(f"{options.directory}{base_id}_{comp_id}_rotamer_difference.csv", index=False)

if __name__ == "__main__":
    main()
