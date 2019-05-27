
import argparse
import logging
import os.path
import os
import sys
import time
from string import ascii_uppercase
logger = logging.getLogger(__name__)

import numpy as np

from .backbone import NullSpaceOptimizer, move_direction_adp
from .clash import ClashDetector
from .samplers import ChiRotator, CBAngleRotator, BondRotator
from .samplers import CovalentBondRotator, GlobalRotator
from .samplers import RotationSets, Translator
from .solvers import QPSolver, MIQPSolver, QPSolver2, MIQPSolver2
from .structure import Structure
from .structure.ligand import BondOrder
from .transformer import Transformer
from .validator import Validator
from .volume import XMap
from .scaler import MapScaler
from .relabel import RelabellerOptions, Relabeller
from .qfit import QFitRotamericResidueOptions
from .structure.rotamers import ROTAMERS

os.environ["OMP_NUM_THREADS"] = "1"

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("structure", type=str,
            help="PDB-file containing structure.")
    p.add_argument('-cif', "--cif_file", type=str, default=None,
            help="CIF file describing the ligand")
    p.add_argument('selection', type=str,
            help="Chain 1, Chain 2 determining the interface to be optimized e.g. A,B")
    args = p.parse_args()

    return args

""" for rotamer in resilist chainID1:
    for AA in AAlist chainID2:
        for rotamer in AA:
            compute self energy
            compute pairwise energies
 """

class QFit_ppiDesignResidue():

    def __init__(self, residue, structure, options):
        self.chain = residue.chain[0]
        self.resi = residue.resi[0]
        self.conformer = residue
        self._coor_set = [self.conformer.coor]
        self.structure = structure
        self.options = options
        self.incomplete = False

        # Check if residue is complete. If not, complete it:
        atoms = residue.name
        for atom in residue._rotamers['atoms']:
            if atom not in atoms:
                residue.complete_residue()
                residue._init_clash_detection()
                # self.incomplete = True
                # Modify the structure to include the new residue atoms:
                index = len(structure.record)
                mask = getattr(residue, 'atomid') >= index
                data = {}
                for attr in structure.data:
                    data[attr] = np.concatenate((getattr(structure, attr),
                                                 getattr(residue, attr)[mask]))
                structure = Structure(data)
                break

        # If including hydrogens:
        if options.hydro:
            for atom in residue._rotamers['hydrogens']:
                if atom not in atoms:
                    print(f"[WARNING] Missing atom {atom} of residue "
                          f"{residue.resi[0]},{residue.resn[0]}")
                    continue

        # super().__init__(residue, structure, xmap, options)
        self.residue = residue
        self.residue._init_clash_detection(self.options.clash_scaling_factor)
        # Get the segment that the residue belongs to
        chainid = self.residue.chain[0]
        self.segment = None
        for segment in self.structure.segments:
            if segment.chain[0] == chainid and self.residue in segment:
                index = segment.find(self.residue.id)
                if (len(segment[index].name) == len(self.residue.name)
                   and segment[index].altloc[-1] == self.residue.altloc[-1]):
                    self.segment = segment
                    break
        if self.segment is None:
            raise RuntimeError(f"Could not determine the protein segment of "
                               f"residue {self.chain}, {self.resi}.")

        # Set up the clashdetector, exclude the bonded interaction of the N and
        # C atom of the residue
        # self._setup_clash_detector()

    def run(self):
        if self.options.sample_backbone:
            self._sample_backbone()
        if self.options.sample_angle and self.residue.resn[0] != 'PRO' and self.residue.resn[0] != 'GLY':
            self._sample_angle()
        if self.residue.nchi >= 1 and self.options.sample_rotamers:
            self._sample_sidechain()

    def _sample_sidechain(self):
        opt = self.options
        start_chi_index = 1
        end_chi_index = self.residue.nchi + 1

        rotamers = self.residue.rotamers
        new_coor_set = []

        for coor in self._coor_set:
            self.residue.coor = coor
            for rotamer in rotamers:
                for chi_index in range (start_chi_index, end_chi_index):
                    self.residue.set_chi(chi_index, rotamer[chi_index - 1])
                    new_coor_set.append(self.residue.coor)
        self._coor_set = new_coor_set

    def get_conformers(self):
        conformers = []
        for coor in self._coor_set:
            conformer = self.conformer.copy()
            conformer = conformer.extract(f"resi {self.conformer.resi[0]} and chain {self.conformer.chain[0]}")
            conformer.coor = coor
            conformer.q = 1
            conformers.append(conformer)
        return conformers

    def tofile(self):
        conformers = self.get_conformers()
        for n, conformer in enumerate(conformers, start=1):
            fname = os.path.join(self.options.directory, f'conformer_{n}.pdb')
            conformer.tofile(fname)
        # Make a multiconformer residue
        nconformers = len(conformers)
        if nconformers < 1:
            msg = "No conformers could be generated. \
             Check for initial clashes."
            raise RuntimeError(msg)
        mc_residue = Structure.fromstructurelike(conformers[0])
        if nconformers == 1:
            mc_residue.altloc = ''
        else:
            mc_residue.altloc = 'A'
            for altloc, conformer in zip(ascii_uppercase[1:], conformers[1:]):
                conformer.altloc = altloc
                mc_residue = mc_residue.combine(conformer)

        mc_residue = mc_residue.reorder()
        fname = os.path.join(self.options.directory,
                             f"multiconformer_residue.pdb")
        mc_residue.tofile(fname)
 
def main():
    args = parse_args()

    #Extract chains from command line options
    chainID1, chainID2 = args.selection.split(',')
    
    #Load structure from file
    structure = Structure.fromfile(args.structure)

    sel_str = f"chain {chainID1}"
    prot1 = structure.extract(sel_str)

    sel_str = f"chain {chainID2}"
    prot2 = structure.extract(sel_str)

    options = QFitRotamericResidueOptions()

    # Get first residue
    resid = 181
    resi = structure.extract(f'resi {resid} and chain {chainID1}')
    chain1 = resi[chainID1]
    conformer1 = chain1.conformers[0]
    conf_resi = conformer1[int(resid)]
    res1 = QFit_ppiDesignResidue(conf_resi, structure, options)

    # Get second residue
    resid = 142
    resi = structure.extract(f'resi {resid} and chain {chainID2}')
    chain2 = resi[chainID2]
    conformer2 = chain2.conformers[0]
    conf_resi = conformer2[int(resid)]
    res2 = QFit_ppiDesignResidue(conf_resi, structure, options)
    
    
    res1._sample_sidechain()
    res2._sample_sidechain()
    #qfit.tofile()