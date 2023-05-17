#!/usr/bin/env python

"""
This script will take in 2 pdbs and a ligand or geometric point in the PDB and the PDB names and output a list of overlapping ligands and a list of close residues (determined by -distance).
INPUT: 2 PDB, 2 PDB names, ligand [optional: distance]
OUTPUT: Text file with list of close residue, text file with list of overlapping ligands 

example:
subset_structure_AH.py holo_pdb.pdb apo_pdb.pdb --holo_name {holo name} --apo_name {holo name} -ls {ligand name}
"""

import os
import os.path
import time
from argparse import ArgumentParser

import numpy as np
from qfit.structure import Structure
from qfit.qfit import QFitOptions


def parse_args():
    p = ArgumentParser(description=__doc__)
    p.add_argument("apo_structure", type=str, help="Apo Structure.")
    p.add_argument("holo_structure", type=str, help="Holo structure post alignment")

    # Output options
    p.add_argument(
        "-d",
        "--directory",
        type=os.path.abspath,
        default=".",
        metavar="<dir>",
        help="Directory to store results.",
    )
    p.add_argument("--pdb_holo", help="Name of the input Holo PDB.")
    p.add_argument("--pdb_apo", help="Name of the input Apo PDB.")

    # New subset arguments
    p.add_argument(
        "-dis",
        "--distance",
        type=float,
        default="5.0",
        metavar="<float>",
        help="Distance from start site to identify ",
    )
    p.add_argument(
        "-sta",
        "--starting_site",
        type=float,
        default="1.0",
        metavar="<float>",
        help="Distance from start site to identify ",
    )
    p.add_argument(
        "-ls",
        "--ligand_start",
        help="Ligand in which you want to measure your distance from",
    )
    args = p.parse_args()
    return args


class QFitMultiResOptions(QFitRotamericResidueOptions):
    def __init__(self):
        super().__init__()
        self.ligand_start = None
        self.distance = None
        self.pdb_holo = None
        self.pdb_apo = None


class subset_str:
    def __init__(self, holo_structure, apo_structure, options):
        self.holo_structure = holo_structure  # PDB with HOH at the bottom
        self.apo_structure = apo_structure  # PDB with HOH at the bottom
        self.close_atoms_chain_holo = None
        self.close_atoms_chain_apo = None
        self.close_hetatoms_apo = None
        self.pdb_holo = ""
        self.pdb_apo = ""
        self.options = options

    def run(self):
        if not self.options.pdb_holo is None:
            self.pdb_holo = self.options.pdb_holo + "_"

        if not self.options.pdb_apo is None:
            self.pdb_apo = self.options.pdb_apo + "_"

        lig_structure = self.select_lig()
        lig_overlap = self.select_close_ligands()
        if not lig_overlap == None:
            with open(self.pdb_apo + "ligand_overlap.txt", "w") as file:
                file.write(lig_overlap)
        substructure_apo, substructure_holo = self.select_close_residues()
        fname = self.pdb_holo + "_subset.pdb"
        substructure_holo.tofile(fname)
        fname = self.pdb_apo + "_subset.pdb"
        substructure_apo.tofile(fname)
        return substructure_apo, substructure_holo

    def select_lig(self):
        """
        Select the residue IDs of the ligands you want to extract; get a central value of all atoms in that ligand
        """
        # first we are going to check which resiudes are ligands
        lig_structure = self.holo_structure.extract(
            "resn", self.options.ligand_start
        )  #
        # calculate center distance structure.residue.calc_coordinates
        center_x = np.mean(lig_structure.coor[:, 0])
        center_y = np.mean(lig_structure.coor[:, 1])
        center_z = np.mean(lig_structure.coor[:, 2])
        self.lig_center = [center_x, center_y, center_z]
        return lig_structure

    def select_close_ligands(self):
        self.hetatoms_apo = self.apo_structure.extract("record", "HETATOM", "==")
        dist_apo = np.linalg.norm(
            self.hetatoms_apo.coor[:][:] - self.lig_center, axis=1
        )
        mask_apo = dist_apo < 10.0  # self.options.distance
        sel_residue_apo = self.hetatoms_apo.resi[mask_apo]
        sel_chain_apo = self.hetatoms_apo.chain[mask_apo]
        for chain in set(sel_chain_apo):
            mask_lig = select_chain_apo == chain
            sel_residue_apo = sel_residue_apo[mask_lig]
            for residue in sel_residue2:
                try:
                    res_atoms = self.hetatoms_apo.extract(
                        f"chain {chain} and resi {residue}"
                    )
                    self.close_hetatoms_apo = close_hetatoms_apo.combine(res_atoms)
                except NameError:
                    self.close_hetatoms_apo = self.hetatoms_apo.extract(
                        f"chain {chain} and resi {residue}"
                    )
        return self.close_hetatoms_apo

    def select_close_residues(self):
        self.atoms_holo = self.holo_structure.extract("record", "ATOM", "==")
        self.atoms_apo = self.apo_structure.extract("record", "ATOM", "==")
        dist_holo = np.linalg.norm(self.atoms_holo.coor[:][:] - self.lig_center, axis=1)
        dist_apo = np.linalg.norm(self.atoms_apo.coor[:][:] - self.lig_center, axis=1)
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
                    res_atoms = self.atoms_holo.extract(
                        f"chain {chain} and resi {residue}"
                    )
                    self.close_atoms_chain_holo = close_atoms_chain_holo.combine(
                        res_atoms
                    )
                except NameError:
                    self.close_atoms_chain_holo = self.atoms_holo.extract(
                        f"chain {chain} and resi {residue}"
                    )
        for chain in set(sel_chain_apo):
            mask2 = sel_chain_apo == chain
            sel_residue2 = sel_residue_apo[mask2]
            for residue in sel_residue2:
                try:
                    res_atoms = self.atoms_apo.extract(
                        f"chain {chain} and resi {residue}"
                    )
                    self.close_atoms_chain_apo = close_atoms_chain_apo.combine(
                        res_atoms
                    )
                except NameError:
                    self.close_atoms_chain_apo = self.atoms_apo.extract(
                        f"chain {chain} and resi {residue}"
                    )
        fname_holo = self.pdb_holo + "_subset.pdb"
        return self.close_atoms_chain_apo, self.close_atoms_chain_holo


def main():
    args = parse_args()
    print(args)
    try:
        os.mkdir(args.directory)
    except OSError:
        pass
    # Load structure and prepare it
    apo_structure = Structure.fromfile(
        args.apo_structure
    ).reorder()  # put H20 on the bottom
    holo_structure = Structure.fromfile(
        args.holo_structure
    ).reorder()  # put H20 on the bottom
    options_multi = QFitMultiResOptions()
    options_multi.apply_command_args(args)
    sub_structure = subset_str(holo_structure, apo_structure, options_multi)
    substructure = sub_structure.run()


if __name__ == "__main__":
    main()
