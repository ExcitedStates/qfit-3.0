
import argparse
import logging
import os.path
import os
import sys
import time
import itertools
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
from .vdw_radii import vdwRadiiTable, EpsilonTable

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

class QFit_ppiResidueSampler():

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

    def _sample_backbone(self):
        # Check if residue has enough neighboring residues
        index = self.segment.find(self.residue.id)
        # active = self.residue.active
        nn = self.options.neighbor_residues_required
        if index < nn or index + nn > len(self.segment):
            return
        segment = self.segment[index - nn: index + nn + 1]
        atom_name = "CB"
        if self.residue.resn[0] == "GLY":
            atom_name = "O"
        atom = self.residue.extract('name', atom_name)
        directions = np.identity(3)

        for n, residue in enumerate(self.segment.residues[::-1]):
            for backbone_atom in ['N', 'CA', 'C', 'O']:
                if backbone_atom not in residue.name:
                    print(f"[WARNING] Missing backbone atom for residue "
                          f"{residue.resi[0]} of chain {residue.chain[0]}.\n"
                          f"Skipping backbone sampling for residue "
                          f"{self.residue.resi[0]} of chain {residue.chain[0]}.")
                    self._coor_set.append(self.segment[index].coor)
                    return

        optimizer = NullSpaceOptimizer(segment)

        start_coor = atom.coor[0]
        torsion_solutions = []
        amplitudes = np.arange(0.1, self.options.sample_backbone_amplitude + 0.01,
                                 self.options.sample_backbone_step)
        sigma = self.options.sample_backbone_sigma
        for amplitude, direction in itertools.product(amplitudes, directions):
            endpoint = start_coor + (amplitude + sigma * np.random.random()) * direction
            optimize_result = optimizer.optimize(atom_name, endpoint)
            torsion_solutions.append(optimize_result['x'])

            endpoint = start_coor - (amplitude + sigma * np.random.random()) * direction
            optimize_result = optimizer.optimize(atom_name, endpoint)
            torsion_solutions.append(optimize_result['x'])
        starting_coor = segment.coor

        for solution in torsion_solutions:
            optimizer.rotator(solution)
            self._coor_set.append(self.segment[index].coor)
            segment.coor = starting_coor
        # print(f"\nBackbone sampling generated {len(self._coor_set)} conformers.\n"        

    def get_conformers(self):
        conformers = []
        for coor in self._coor_set:
            conformer = self.conformer.copy()
            conformer = conformer.extract(f"resi {self.conformer.resi[0]} and chain {self.conformer.chain[0]}")
            conformer.coor = coor
            conformer.q = 1
            conformers.append(conformer)
        return conformers

    def tofile(self, fn):
        conformers = self.get_conformers()
        for n, conformer in enumerate(conformers, start=1):
            fname = os.path.join(self.options.directory, f'{fn}_{n}.pdb')
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
                             fn)
        mc_residue.tofile(fname)

class QFit_FF():
    def initMetric(self):
        # print("Calculating all possible Van der Waals interactions:")
        for i in range(len(self.nodes)):
            for j in range(i+1,len(self.nodes)):
                if self.nodes[i].resi[0]!=self.nodes[j].resi[0] or self.nodes[i].chain[0]!=self.nodes[j].chain[0]:
                    self.metric[i][j] = self.calc_energy(self.nodes[i],self.nodes[j])
                    self.metric[j][i] = self.metric[i][j]
            # update_progress(i/len(self.nodes))

    def vdw_energy(self,atom1, atom2, coor1, coor2):
        e = EpsilonTable[atom1][atom2]
        s = (vdwRadiiTable[atom1]+vdwRadiiTable[atom2]) / 1.122
        r = np.linalg.norm(coor1 - coor2)
        return 4 * e * (np.power(s/r, 12) - np.power(s/r, 6))
 
    def calc_energy(self,node1, node2):
        energy = 0.0
        if np.linalg.norm(node1.coor[0]-node2.coor[0]) < 16.0:
            for name1,ele1,coor1 in zip(node1.name,node1.e,node1.coor):
                for name2,ele2,coor2 in zip(node2.name,node2.e,node2.coor):
                    if name1 not in ["N","CA","C","O","H","HA"] or name2 not in ["N","CA","C","O","H","HA"] or np.abs(node1.resi[0] - node2.resi[0]) != 1:
                        energy += self.vdw_energy(ele1,ele2,coor1,coor2)
        return energy
 
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

    L_res = []
    for chainID, resid in zip([chainID1, chainID2], [485, 72]):
        resi = structure.extract(f'resi {resid} and chain {chainID}')
        chain = resi[chainID]
        conformer = chain.conformers[0]
        conf_resi = conformer[int(resid)]
        L_res.append (QFit_ppiResidueSampler(conf_resi, structure, options))

    for res in L_res:
        res.run()
        #res1.tofile("485.pdb")
        #res2.tofile("72.pdb")

    conf1 = res1.get_conformers()
    conf2 = res2.get_conformers()

    print(len(conf1))
    print(len(conf2))

    #int i = (n * r) + c – ((r * (r+1)) / 2)

    ee = np.empty((len(conf1),len(conf2)))
    #print(ee)

    E = QFit_FF()

    for n1, c1 in enumerate(conf1, start=1):
        for n2, c2 in enumerate(conf2, start=1):
            ee[n1-1,n2-1] = E.calc_energy (c1,c2)

    print(ee)