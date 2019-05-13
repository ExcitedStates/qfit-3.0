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

"""Automatically build a multiconformer residue"""
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
    p.add_argument('-occ', "--occ_cutoff", type=float, default=0.01, metavar="<float>",
            help="Remove conformers with occupancies below occ_cutoff. Default = 0.01")
    p.add_argument('-rmsd',"--rmsd_cutoff", type=float, default=0.01, metavar="<float>")
    args = p.parse_args()

    return args


def main():
    args = parse_args()
    try:
        os.makedirs(args.directory)
        output_file = os.path.join(args.directory,
                                 args.structure[:-4]+"_norm.pdb")
    except OSError:
        output_file = args.structure[:-4]+"_norm.pdb"

    structure = Structure.fromfile(args.structure).reorder()
    to_remove = []

    # Iterate over every residue...
    for chain in structure:
        for residue in chain:
            altlocs = list(set(residue.altloc))
            # Deal with the simplest case first: only a single conformer
            if len(altlocs) == 1:
                residue._q[residue._selection] = 1.0
                continue       

            # Should we collapse the backbone for the current residue?
            altloc1 = altlocs[0]
            if not altloc1:
                altloc1 = altlocs[1]
            conf1 = residue.extract("altloc",altloc1)
            conf1 = conf1.extract("name",('N','CA','C','O'))
            should_collapse = True
            for altloc2 in altlocs[1:]:
                if altloc1 == altloc2:
                    continue
                conf2 = residue.extract("altloc",altloc2)  
                conf2 = conf2.extract("name",('N','CA','C','O'))
                if np.mean(np.linalg.norm(conf2.coor-conf1.coor,axis=1)) > 0.05 and np.min(conf2.q) > args.occ_cutoff :
                    should_collapse = False
        
            # Add the atoms of the collapsed backbone to the to_remove list
            # and fix altloc and occupancy of the backbone
            if should_collapse:
                conf1._q[conf1._selection] = 1.0
                for altloc2 in altlocs[1:]:
                    conf2 = residue.extract("altloc",altloc2)  
                    conf2 = conf2.extract("name",('N','CA','C','O'))
                    [to_remove.append(x) for x in conf2._selection]
                
            # If the backbone is collapsed, we can remove identical side chain conformers
            # or side chain conformers that fall below the occupancy cutoff:
            if '' in altlocs or should_collapse:
                try:
                    altlocs.remove('')
                except ValueError:
                    conf = copy.deepcopy(conf1)
                    pass
                for i,altloc1 in enumerate(altlocs):
                    conf1 = residue.extract("altloc",altloc1)
                    if np.min(conf1.q) < args.occ_cutoff:
                       [to_remove.append(x) for x in conf1._selection]
                       continue
                    for altloc2 in altlocs[i+1:]:
                       conf2 = residue.extract("altloc",altloc2)
                       if conf1.rmsd(conf2) < args.rmsd_cutoff:
                            [to_remove.append(x) for x in conf2._selection]
                try:
                    structure._altloc[conf._selection] = ''
                except:
                    pass
            # Now, to the case where the backbone is not collapsed
            else:
                # Here, we only want to remove if ALL conformers are identical or below 
                # occupancy cutoff
                is_identical = True
                for i,altloc1 in enumerate(altlocs):
                    conf1 = residue.extract("altloc", altloc1)
                    for altloc2 in altlocs[i+1:]:
                        conf2 = residue.extract("altloc", altloc2)
                        # If the conformer has occupancy greater than the cutoff
                        # and if it is not identical to all                         
                        if np.min(conf2.q) > args.occ_cutoff and conf1.rmsd(conf2) > args.rmsd_cutoff:
                                is_identical = False
                                break
                # If all conformers converged (either because of RMSD or occupancy) 
                if is_identical:
                    for altloc1 in altlocs[1:]:
                        conf1 = residue.extract("altloc",altloc1)
                        [to_remove.append(x) for x in conf1._selection]

                
    
    # Remove conformers in to_remove list:
    mask = structure.active
    mask[to_remove] = False
    data = {}
    for attr in structure.data:
            data[attr] = getattr(structure, attr).copy()[mask]
    structure = Structure(data)
    # Normalize occupancies and fix altlocs:
    for chain in structure:
        for residue in chain:
            altlocs=list(set(residue.altloc))
            try:
                altlocs.remove('')
            except ValueError:
                pass
            naltlocs = len(altlocs)
            if naltlocs < 2:
                residue._q[residue._selection] = 1.0
                residue._altloc[residue._selection] = ''
            else:
                conf = residue.extract('altloc',altlocs)
                natoms = len(residue.extract('altloc',altlocs[-1]).name)
                factor = natoms/np.sum(conf.q)
                residue._q[conf._selection] *= factor
    structure.tofile(output_file)
    print(len(to_remove))
