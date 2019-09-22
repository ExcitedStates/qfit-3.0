#Edited by Stephanie Wankowicz
#began: 2019-04-10
#last edited: 2019-09-17
'''
Excited States software: qFit 3.0
Contributors: Saulo H. P. de Oliveira, Gydo van Zundert, Henry van den Bedem, Stephanie Wankowicz
Contact: vdbedem@stanford.edu
'''

import pkg_resources  # part of setuptools
from .qfit import QFitRotamericResidue, QFitRotamericResidueOptions
from .qfit import QFitSegment, QFitSegmentOptions
from .qfit import print_run_info
from .qfit_protein import QFitProteinOptions, QFitProtein
import multiprocessing as mp
import os.path
import os
import sys
import time
import copy
import numpy as np
from argparse import ArgumentParser
from math import ceil
from . import MapScaler, Structure, XMap
from .structure.base_structure import _BaseStructure
#from .structure.ligand import
#from .structure,.residue import residue_type


os.environ["OMP_NUM_THREADS"] = "1"

def parse_args():
    p = ArgumentParser(description=__doc__)
    p.add_argument("apo_structure", type=str,
                   help="Apo Structure.")
    p.add_argument("holo_structure", type=str,
                   help="Holo structure post alignment")

    # Output options
    p.add_argument("-d", "--directory", type=os.path.abspath, default='.',
                   metavar="<dir>", help="Directory to store results.")
    p.add_argument("--debug", action="store_true",
                   help="Write intermediate structures to file for debugging.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Be verbose.")
    p.add_argument("--pdb_holo", help="Name of the input Holo PDB.")
    p.add_argument("--pdb_apo", help="Name of the input Apo PDB.")
    
 #new subset arguments
    p.add_argument("-dis", "--distance", type=float, default='5.0',
                  metavar="<float>", help="Distance from start site to identify ")
    p.add_argument("-sta", "--starting_site", type=float, default='1.0',
             metavar="<float>", help="Distance from start site to identify ")
    p.add_argument("-ls", "--ligand_start", help="Ligand in which you want to measure your distance from")
    args = p.parse_args()
    return args

class QFitMultiResOptions(QFitRotamericResidueOptions, QFitSegmentOptions):
    def __init__(self):
        super().__init__()
        self.nproc = 1
        self.verbose = True
        self.omit = False
        self.ligand_start = None
        self.distance = None
        self.pdb_holo = None
        self.pdb_apo = None


class QFitMultiResidue:
    def __init__(self, holo_structure, apo_structure, options):
        self.holo_structure = holo_structure #PDB with HOH at the bottom
        self.apo_structure = apo_structure #PDB with HOH at the bottom
        self.close_atoms_chain_holo = None
        self.close_atoms_chain_apo = None
        self.close_hetatoms_apo = None
        self.options = options #user input
    
     def run(self):
        if not self.options.pdb_holo==None:
            self.pdb_holo=self.options.pdb_holo + '_'
        else:
            self.pdb_holo=''
        if not self.options.pdb_apo==None:
            self.pdb_apo=self.options.pdb_apo + '_'
        else:
            self.pdb_apo=''
        print(self.options.ligand_start)
        lig_strucutre = self.select_lig()
        lig_overlap = self.select_close_ligands()
        if not lig_overlap == None:
            with open(self.pdb_apo + 'ligand_overlap.txt', 'w') as file:
                file.write(lig_overlap)
        substructure_apo, substructure_holo = self.select_close_residues()
        fname = (self.pdb_holo + "subset.pdb") #self.options.distance + pdbname
        substructure_holo.tofile(fname)
        fname = (self.pdb_apo + "subset.pdb") #self.options.distance + pdbname
        substructure_apo.tofile(fname)
        return substructure_apo, substructure_holo

    def select_lig(self):
        '''
        Select the residue IDs of the ligands you want to extract; get a central value of all atoms in that ligand
        '''
        #first we are going to check which resiudes are ligands
        #hetatms = self.structure.extract('record', 'HETATM', '==')
        lig_structure=self.holo_structure.extract('resn', self.options.ligand_start) #
        #calculate center distance structure.residue.calc_coordinates
        center_x=np.mean(lig_structure.coor[:,0])
        center_y=np.mean(lig_structure.coor[:,1])
        center_z=np.mean(lig_structure.coor[:,2])
        self.lig_center=[center_x,center_y,center_z]
        #print(self.lig_center)
        return lig_structure

   def select_close_ligands(self):
        self.hetatoms_apo = self.apo_structure.extract('record', 'HETATOM', '==')
        dist_apo=np.linalg.norm(self.hetatoms_apo.coor[:][:]-self.lig_center, axis=1)
        mask_apo = dist_apo < 10.0#self.options.distance
        sel_residue_apo = self.hetatoms_apo.resi[mask_apo]
        sel_chain_apo = self.hetatoms_apo.chain[mask_apo]
        #print('ligands overlap:')
        for chain in set(sel_chain_apo):
            mask_lig = select_chain_apo == chain
            sel_residue_apo = sel_residue_apo[mask_lig]
            for residue in sel_residue2:
                try:
                    res_atoms = self.hetatoms_apo.extract(f'chain {chain} and resi {residue}')
                    self.close_hetatoms_apo = close_hetatoms_apo.combine(res_atoms)
                except NameError:
                    self.close_hetatoms_apo = self.hetatoms_apo.extract(f'chain {chain} and resi {residue}')
        return self.close_hetatoms_apo
        
   def select_close_residues(self):
        self.atoms_holo = self.holo_structure.extract('record', 'ATOM', '==')
        self.atoms_apo = self.apo_structure.extract('record', 'ATOM', '==')
        dist_holo=np.linalg.norm(self.atoms_holo.coor[:][:]-self.lig_center, axis=1)
        dist_apo=np.linalg.norm(self.atoms_apo.coor[:][:]-self.lig_center, axis=1)
        mask_holo = dist_holo < self.options.distance
        mask_apo = dist_apo < self.options.distance
        sel_residue_holo = self.atoms_holo.resi[mask_holo]
        sel_chain_holo = self.atoms_holo.chain[mask_holo]
        sel_residue_apo = self.atoms_apo.resi[mask_apo]
        sel_chain_apo = self.atoms_apo.chain[mask_apo]
        for chain in set(sel_chain_holo):
            mask2 = sel_chain_holo == chain
            sel_residue2 = sel_residue_holo[mask2]
            for residue in sel_residue2:
                try:
                    res_atoms=self.atoms_holo.extract(f'chain {chain} and resi {residue}')
                    self.close_atoms_chain_holo=close_atoms_chain_holo.combine(res_atoms)
                except NameError:
                    self.close_atoms_chain_holo=self.atoms_holo.extract(f'chain {chain} and resi {residue}')
        for chain in set(sel_chain_apo):
            mask2 = sel_chain_apo == chain
            sel_residue2 = sel_residue_apo[mask2]
            for residue in sel_residue2:
                try:
                    res_atoms=self.atoms_apo.extract(f'chain {chain} and resi {residue}')
                    self.close_atoms_chain_apo=close_atoms_chain_apo.combine(res_atoms)
                except NameError:
                    self.close_atoms_chain_apo=self.atoms_apo.extract(f'chain {chain} and resi {residue}')
        fname_holo = (self.pdb_holo+"_subset.pdb") #self.options.distance + pdbname
        return self.close_atoms_chain_apo, self.close_atoms_chain_holo

def main():
    args = parse_args()
    print(args)
    try:
        os.mkdir(args.directory)
    except OSError:
        pass
    # Load structure and prepare it
    apo_structure = Structure.fromfile(args.apo_structure).reorder() #put H20 on the bottom
    print(apo_structure)
    apo_structure = apo_structure.extract('e', 'H', '!=')
    holo_structure = Structure.fromfile(args.holo_structure).reorder() #put H20 on the bottom
    holo_structure = holo_structure.extract('e', 'H', '!=')
    options_multi = QFitMultiResOptions()
    options_multi.apply_command_args(args)
    time0 = time.time()
    sub_structure = QFitMultiResidue(holo_structure, apo_structure, options_multi)
    substructure = sub_structure.run()
    print(f"Total time: {time.time() - time0}s")
