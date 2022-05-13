import numpy as np
import argparse
import logging
import os
from collections import namedtuple
from . import Structure
from .structure.rotamers import ROTAMERS


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
    total_occ_redist = np.sum(np.unique(residue.extract('altloc', remove).q))
    residue_out = residue.extract('altloc', keep)
    if len(set(residue_out.altloc)) == 1: #if only one altloc left, assign 1:
       residue_out.q = 1.0
    else:
       naltlocs = len(np.unique(residue_out.extract('q', 1.0, '!=').altloc)) #number of altlocs left
       occ_redist = round(total_occ_redist/naltlocs, 2)
       add_occ_redist = 0
       if ((occ_redist*naltlocs) + np.sum(np.unique(residue_out.extract('q', 1.0, '!=').q))) != 1.0:
          add_occ_redist = round(1.0 - ((occ_redist*naltlocs) + np.sum(np.unique(residue_out.extract('q', 1.0, '!=').q))), 2)
       occ_sum = [] #get sum of occupancies
       add_occ = False
       for alt in np.unique(residue_out.altloc):
           if np.all(residue_out.extract('altloc', alt, '==').q < 1.0): #ignoring backbone atoms with full occ
              if add_occ == False:
                 residue_out.extract('altloc', alt, '==').q = residue_out.extract('altloc', alt, '==').q + occ_redist + add_occ_redist
                 add_occ = True
              else:
                 residue_out.extract('altloc', alt, '==').q = residue_out.extract('altloc', alt, '==').q + occ_redist
              occ_sum.append((residue_out.extract('altloc', alt, '==').q[0]))

    return residue_out


def main():
    args = parse_args()
    try:
        os.makedirs(args.directory)
        output_file = os.path.join(args.directory,
                                   args.structure[:-4] + "_norm.pdb")
    except OSError:
        output_file = args.structure[:-4] + "_norm.pdb"

    structure = Structure.fromfile(args.structure)
    
    #seperate het versus atom (het allowed to have <1 occ)
    hetatm = structure.extract('record', 'HETATM', '==')
    structure = structure.extract('record', 'ATOM', '==')

    # Capture LINK records
    link_data = structure.link_data

    # Get list of all non-hetatom residue
    n_removed = 0  #keep track of the residues we are removing
    chains = set(structure.chain)
    for chain in chains:
        residues = set(structure.extract('chain', chain, '==').resi)
        for res in residues:
            residue = structure.extract(f'chain {chain} and resi {res}')
            if np.any(residue.q < args.occ_cutoff): #if any atom occupancy falls below the cutoff value
               altlocs_remove = set(residue.extract('q', args.occ_cutoff, '<=').altloc)
               #confirm all atoms have the same q in each conformer
               for alt in altlocs_remove:
                   all_same = np.all(residue.extract('altloc', alt, '==').q)
                   if all_same == False:
                      print(f'Not all atoms have the same occupancy in resi {res}, chain {chain}, altloc {alt}')
                      break
               n_removed += len(set(altlocs_remove))
               altlocs_keep = set(residue.extract('q', args.occ_cutoff, '>').altloc)
               residue_out = remove_redistribute_conformer(residue, altlocs_remove, altlocs_keep)

               try:
                  out_structure = out_structure.combine(residue_out)
               except UnboundLocalError:
                  out_structure = residue_out
            else:
              try:
                 out_structure = out_structure.combine(residue) #no altlocs removed
              except UnboundLocalError:
                 out_structure = residue
    
    #add het atoms back in
    structure = structure.combine(hetatm)
    # Reattach LINK records
    structure.link_data = link_data

    #output structure
    structure.tofile(output_file)

    print(f"normalize_occupancies: {n_removed} atoms had occ < {args.occ_cutoff} and were removed.")
    print(n_removed)  # for post_refine_phenix.sh
