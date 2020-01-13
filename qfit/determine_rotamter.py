"""For each residue/alt conf determines the rotatmer stat"""
import numpy as np
import argparse
import logging
import os
import sys
import time
import math
from string import ascii_uppercase
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
    args = p.parse_args()
    return args

chi_atoms = dict(
        chi1=dict(
            ARG=['N', 'CA', 'CB', 'CG'],
            ASN=['N', 'CA', 'CB', 'CG'],
            ASP=['N', 'CA', 'CB', 'CG'],
            CYS=['N', 'CA', 'CB', 'SG'],
            GLN=['N', 'CA', 'CB', 'CG'],
            GLU=['N', 'CA', 'CB', 'CG'],
            HIS=['N', 'CA', 'CB', 'CG'],
            ILE=['N', 'CA', 'CB', 'CG1'],
            LEU=['N', 'CA', 'CB', 'CG'],
            LYS=['N', 'CA', 'CB', 'CG'],
            MET=['N', 'CA', 'CB', 'CG'],
            PHE=['N', 'CA', 'CB', 'CG'],
            PRO=['N', 'CA', 'CB', 'CG'],
            SER=['N', 'CA', 'CB', 'OG'],
            THR=['N', 'CA', 'CB', 'OG1'],
            TRP=['N', 'CA', 'CB', 'CG'],
            TYR=['N', 'CA', 'CB', 'CG'],
            VAL=['N', 'CA', 'CB', 'CG1'],
        ),
        altchi1=dict(
            VAL=['N', 'CA', 'CB', 'CG2'],
        ),
        chi2=dict(
            ARG=['CA', 'CB', 'CG', 'CD'],
            ASN=['CA', 'CB', 'CG', 'OD1'],
            ASP=['CA', 'CB', 'CG', 'OD1'],
            GLN=['CA', 'CB', 'CG', 'CD'],
            GLU=['CA', 'CB', 'CG', 'CD'],
            HIS=['CA', 'CB', 'CG', 'ND1'],
            ILE=['CA', 'CB', 'CG1', 'CD1'],
            LEU=['CA', 'CB', 'CG', 'CD1'],
            LYS=['CA', 'CB', 'CG', 'CD'],
            MET=['CA', 'CB', 'CG', 'SD'],
            PHE=['CA', 'CB', 'CG', 'CD1'],
            PRO=['CA', 'CB', 'CG', 'CD'],
            TRP=['CA', 'CB', 'CG', 'CD1'],
            TYR=['CA', 'CB', 'CG', 'CD1'],
        ),
        altchi2=dict(
            ASP=['CA', 'CB', 'CG', 'OD2'],
            LEU=['CA', 'CB', 'CG', 'CD2'],
            PHE=['CA', 'CB', 'CG', 'CD2'],
            TYR=['CA', 'CB', 'CG', 'CD2'],
        ),
        chi3=dict(
            ARG=['CB', 'CG', 'CD', 'NE'],
            GLN=['CB', 'CG', 'CD', 'OE1'],
            GLU=['CB', 'CG', 'CD', 'OE1'],
            LYS=['CB', 'CG', 'CD', 'CE'],
            MET=['CB', 'CG', 'SD', 'CE'],
        ),
        chi4=dict(
            ARG=['CG', 'CD', 'NE', 'CZ'],
            LYS=['CG', 'CD', 'CE', 'NZ'],
        ),
        chi5=dict(
            ARG=['CD', 'NE', 'CZ', 'NH1'],
        ),
    )


def get_angle(v1, v2): 
          """Return angle between two vectors.""" 
          v1_u = v1 / np.linalg.norm(v1)
          v2_u = v2 / np.linalg.norm(v2)


def calc_dihedral(N, CA, CB, CG): #these should be the four points that define the dihedral angle
    print('starting calc dihedral')
    #n_ca = (np.subtract(N, CA)/np.linalg.norm(np.subtract(CB,CA)))
    ca_n = np.subtract(CA,N)
    cb_ca = np.subtract(CB,CA)
    cg_cb = np.subtract(CG,CB)
    ca_n = ca_n/np.linalg.norm(ca_n)
    cb_ca = cb_ca/np.linalg.norm(cb_ca)
    cg_cb = cg_cb/np.linalg.norm(cg_cb)
    n1 = np.cross(ca_n, cb_ca) #cross product
    n2 = np.cross(cb_ca, cg_cb)
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    m1 = np.cross(n1, cb_ca)
    m2 = np.cross(n2, cg_cb)
    x1 = np.dot(n1, n2)
    y1 = np.dot(m1, n2)
    angle1 = -math.atan2(y1, x1)
    print(angle1)
    return angle1
    

def main():
    args = parse_args()
    try:
        os.makedirs(args.directory)
    except OSError:
        pass

    #print(chi_atoms)
    #print(type(chi_atoms))
    structure = Structure.fromfile(args.structure).reorder()
    structure = structure.extract("record", "ATOM", "==")
    for chain in np.unique(structure.chain):
            select2 = structure.extract('chain', chain, '==')
            residues = set(list(select2.resi))
            residue_ids = []
            for i in residues:
                tmp_i = str(i)
                if ':' in tmp_i:
                    resi = int(tmp_i.split(':')[1][1:])
                else:
                    resi = tmp_i
                residue_ids.append(resi)
     
    for id in residue_ids:
           chi_names = list()
           res_tmp = select2.extract('resi', int(id), '==') #this is seperating each residues
           resn_name = (np.array2string(np.unique(res_tmp.resi)), np.array2string(np.unique(res_tmp.resn)),np.array2string(np.unique(res_tmp.chain)))
           if len(np.unique(res_tmp.altloc))>1:
              noalt_atoms = res_tmp.extract('altloc', "", '==')
              res_name=''.join(map(str, np.unique(res_tmp.resn)))
              for alt in np.unique(res_tmp.altloc):
                if alt=='':
                   continue
                print(res_name)
                print(alt)
                mask = np.isin(res_tmp.altloc, [' ', alt])
                res_tmp2 = res_tmp.extract('altloc', alt, '==')
                res_tmp2 = res_tmp2.combine(noalt_atoms)
                for i in chi_atoms.keys():
                  for y in chi_atoms[i].keys():
                      if y==res_name:
                         atom_list = chi_atoms[i][res_name]#res_tmp.resn[0]]
                         print(atom_list)
                         mask_atom = np.isin(res_tmp2.name, atom_list)
                         axis_coor = res_tmp2.coor[mask_atom]
                         calc_dihedral(*axis_coor)
