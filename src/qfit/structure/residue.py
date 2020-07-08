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

import numpy as np
import copy
import math
from .base_structure import _BaseStructure
from .math import *
from .rotamers import ROTAMERS
import time

_PROTEIN_BACKBONE_ATOMS = ['N', 'CA', 'C']
_NUCLEOTIDE_BACKBONE_ATOMS = ['P', "O5'", "C5'", "C4'", "C3'", "O3"]
_SOLVENTS = ['HOH']


def residue_type(residue):
    """Check residue type.

    The following types are recognized:
        rotamer-residue
            Amino acid for which rotamer information is available
        aa-residue
            Amino acid residue
        residue
            DNA / RNA residue
        solvent
            HOH
        ligand
            Other
    """

    # RNA and DNA backbone atoms
    if residue.resn[0] in ROTAMERS:
        return 'rotamer-residue'
    bb_atoms = _PROTEIN_BACKBONE_ATOMS
    if residue.select('name', bb_atoms).size == 3:
        return 'aa-residue'
    xna_atoms = _NUCLEOTIDE_BACKBONE_ATOMS
    if residue.select('name', xna_atoms).size == 6:
        return 'residue'
    if residue.resn[0] in _SOLVENTS:
        return 'solvent'
    return 'ligand'


class _BaseResidue(_BaseStructure):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = (kwargs['resi'], kwargs['icode'])
        self.type = kwargs['type']

    def __repr__(self):
        resi, icode = self.id
        string = "Residue: {}".format(resi)
        if self.id[1] != "":
            string += ":{}".format(icode)
        return string


class _Residue(_BaseResidue):
    pass


class _RotamerResidue(_BaseResidue):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        resname = self.resn[0]
        self._rotamers = ROTAMERS[resname]
        self.nchi = self._rotamers['nchi']
        self.nrotamers = len(self._rotamers['rotamers'])
        self.rotamers = self._rotamers['rotamers']
        self._init_clash_detection()
        self.i=0

    def _init_clash_detection(self, scaling_factor=0.75, covalent_bonds=None):
        # Setup the condensed distance based arrays for clash detection and fill them
        self._ndistances = self.natoms * (self.natoms - 1) // 2
        self._clash_mask = np.ones(self._ndistances, bool)
        self._clash_radius2 = np.zeros(self._ndistances, float)
        radii = self.covalent_radius
        bonds = self._rotamers['bonds']
        if covalent_bonds:
            for bond in covalent_bonds:
                bonds.append(bond)
        offset = self.natoms * (self.natoms - 1) // 2
        for i in range(self.natoms - 1):
            #starting_index = int(offset - sp_comb(self.natoms - i, 2)) - i - 1
            natoms_left = self.natoms - i
            starting_index = offset - natoms_left * (natoms_left - 1) // 2 - i - 1
            name1 = self.name[i]
            covrad1 = radii[i]
            for j in range(i + 1, self.natoms):
                bond1 = [name1, self.name[j]]
                bond2 = bond1[::-1]
                covrad2 = radii[j]
                index = starting_index + j
                self._clash_radius2[index] = covrad1 + covrad2 + 0.4
                if bond1 in bonds or bond2 in bonds:
                    self._clash_mask[index] = False

        self._clash_radius2 *= self._clash_radius2
        self._clash_radius2 *= scaling_factor
        self._clashing = np.zeros(self._ndistances, bool)
        self._dist2_matrix = np.empty(self._ndistances, float)

        # All atoms are active from the start
        self.active = np.ones(self.natoms, bool)
        self._active_mask = np.ones(self._ndistances, bool)

    def update_clash_mask(self):
        offset = self.natoms * (self.natoms - 1) // 2 - 1
        for i, active in enumerate(self.active[:-1]):
            natoms_left = self.natoms - i
            starting_index = offset - natoms_left * (natoms_left - 1) // 2 - i
            end = starting_index + self.natoms - i + 1
            self._active_mask[starting_index: end] = active

    def clashes(self):
        """Checks if there are any internal clashes.
        Deactivated atoms are not taken into account.
        """

        dm = self._dist2_matrix
        coor = self.coor
        dot = np.dot
        k = 0
        for i in range(self.natoms - 1):
            u = coor[i]
            for j in range(i + 1, self.natoms):
                u_v = u - coor[j]
                dm[k] = dot(u_v, u_v)
                k += 1

        np.less_equal(dm, self._clash_radius2 , self._clashing)
        self._clashing &= self._clash_mask
        self._clashing &= self._active_mask
        nclashes = self._clashing.sum()
        return nclashes

    def get_chi(self, chi_index):
        atoms = self._rotamers['chi'][chi_index]
        selection = self.select('name', atoms)
        ordered_sel = []
        for atom in atoms:
            for sel in selection:
                if atom == self._name[sel]:
                    ordered_sel.append(sel)
                    break
        coor = self._coor[ordered_sel]
        angle = dihedral_angle(coor)
        return angle

    def set_chi(self, chi_index, value, covalent=None, length=None):
        atoms = self._rotamers['chi'][chi_index]
        selection = self.select('name', atoms)

        # Translate coordinates to center on coor[1]
        coor = self._coor[selection]
        origin = coor[1].copy()
        coor -= origin

        # Make an orthogonal axis system based on 3 atoms
        backward = gram_schmidt_orthonormal_zx(coor)
        forward = backward.T

        # Complete selection to be rotated
        atoms_to_rotate = self._rotamers['chi-rotate'][chi_index]
        selection = self.select('name', atoms_to_rotate)
        if covalent in atoms_to_rotate:
            # If we are rotating the atom that is covalently bonded
            # to the ligand, we should also rotate the ligand.
            atoms_to_rotate2 = self.name[length:]
            selection2 = self.select('name', atoms_to_rotate2)
            selection = np.array(list(selection) + list(selection2), dtype=int)

        # Create transformation matrix
        angle = np.deg2rad(value - self.get_chi(chi_index))
        rotation = Rz(angle)
        R = forward @ rotation @ backward

        # Apply transformation
        coor_to_rotate = self._coor[selection]
        coor_to_rotate -= origin
        coor_to_rotate = np.dot(coor_to_rotate, R.T)
        coor_to_rotate += origin
        self._coor[selection] = coor_to_rotate

    def print_residue(self):
        for atom, coor, element, b, q in zip(
          self.name, self.coor, self.e, self.b, self.q):
            print("{} {} {} {} {}".format(atom, coor, element, b, q))

    def complete_residue(self):
        if residue_type(self) != "rotamer-residue":
            msg = (f"Cannot complete non-aminoacid residue {self}. "
                   f"Please complete the missing atoms of the residue before "
                   f"running qFit again!")
            raise RuntimeError(msg)

        for atom, position in zip(self._rotamers['atoms'],
                                  self._rotamers['positions']):
            # Found a missing atom!
            if atom not in self.name:
                self.complete_residue_recursive(atom)

    def complete_residue_recursive(self, atom):
        if atom in ['N', 'C', 'CA', 'O']:
            msg = (f"{self} is missing backbone atom {atom}. "
                   f"qFit cannot complete missing backbone atoms. "
                   f"Please complete the missing backbone atoms of "
                   f"the residue before running qFit again!")
            raise RuntimeError(msg)

        ref_atom = self._rotamers['connectivity'][atom][0]
        if ref_atom not in self.name:
            self.complete_residue_recursive(ref_atom)
        idx = np.argwhere(self.name == ref_atom)[0]
        ref_coor = self.coor[idx]
        bond_length, bond_length_sd = (
            self._rotamers['bond_dist'][ref_atom][atom])
        # Identify a suitable atom for the bond angle:
        for angle in self._rotamers['bond_angle']:
            if angle[0][1] == ref_atom and angle[0][2] == atom:
                if angle[0][0][0] == "H":
                    continue
                bond_angle_atom = angle[0][0]
                bond_angle, bond_angle_sd = angle[1]
                if bond_angle_atom not in self.name:
                    self.complete_residue_recursive(bond_angle_atom)
                bond_angle_coor = self.coor[np.argwhere(
                  self.name == bond_angle_atom)[0]]
                dihedral_atom = None
                # If the atom's position is dependent on a rotamer,
                # identify the fourth dihedral angle atom:
                for i, chi in self._rotamers['chi'].items():
                    if (chi[1] == bond_angle_atom and
                       chi[2] == ref_atom and
                       chi[3] == atom):
                        dihedral_atom = chi[0]
                        dihed_angle = self._rotamers['rotamers'][0][i-1]

                # If the atom's position is not dependent on a rotamer,
                # identify the fourth dihedral angle atom:
                if dihedral_atom is None:
                    for dihedral in self._rotamers['dihedral']:
                        if (dihedral[0][1] == bond_angle_atom and
                           dihedral[0][2] == ref_atom and
                           dihedral[0][3] == atom):
                            dihedral_atom = dihedral[0][0]
                            if dihedral[1][0] in self._rotamers['atoms']:
                                other_dihedral_atom = dihedral[1][0]
                                if dihedral_atom not in self.name:
                                    self.complete_residue_recursive(
                                     dihedral_atom)
                                dihedral_atom_coor = self.coor[np.argwhere(
                                 self.name == dihedral_atom)[0]]
                                if other_dihedral_atom not in self.name:
                                    self.complete_residue_recursive(
                                     other_dihedral_atom)
                                other_dihedral_atom_coor = self.coor[
                                 np.argwhere(
                                  self.name == other_dihedral_atom)[0]]
                                try:
                                    dihed_angle = dihedral[1][1]
                                    dihed_angle += dihedral_angle([
                                                    dihedral_atom_coor[0],
                                                    bond_angle_coor[0],
                                                    ref_coor[0],
                                                    other_dihedral_atom_coor[0]
                                                    ])
                                except:
                                    dihed_angle = 180
                                    dihed_angle += dihedral_angle([
                                                    dihedral_atom_coor[0],
                                                    bond_angle_coor[0],
                                                    ref_coor[0],
                                                    other_dihedral_atom_coor[0]
                                                    ])
                            else:
                                dihed_angle = dihedral[1][0]
                            break
                if dihedral_atom is not None:
                    if dihedral_atom not in self.name:
                        self.complete_residue_recursive(dihedral_atom)
                    dihedral_atom_coor = self.coor[np.argwhere(
                                            self.name == dihedral_atom)[0]]
                    break
        new_coor = self.calc_coordinates(dihedral_atom_coor[0],
                                         bond_angle_coor[0],
                                         ref_coor[0],
                                         bond_length,
                                         bond_angle,
                                         dihed_angle)
        new_coor = [round(x, 3) for x in new_coor]
        self.add_atom(atom, atom[0], new_coor)

    def calc_coordinates(self,u, v, Origin, L, bond_angle, dihedral):
        a=u-Origin
        b=v-Origin
        # Find the normal vector that defines the plane between a and b
        n = np.cross(a, b)
        AX,AY,AZ = list(a) ## a vector
        BX,BY,BZ = list(b) ## b vector
        A ,B ,C  = list(n) ## n vector
        # CALCULATE COORDINATES OF ATOM IF POSITIONED COPLANAR TO BOND ATOMS AT LENGTH L AND ANGLE BOND_ANGLE
        F = np.linalg.norm(b)  * np.cos(np.deg2rad(bond_angle))
        denom= (B*B)*(BX*BX+BZ*BZ)+ (A*A)*(BY*BY+BZ*BZ) + (BX*BX+BY*BY)*(C*C) - (2*A*BX*BZ*C) - (2*B*BY)*(A*BX+BZ*C)
        const= L * np.abs(B*BZ-BY*C) * np.sqrt(-F * F * np.inner(n,n)+denom)
        X= (  L*F*((B*B*BX)-(A*B*BY)+C*(-A*BZ+BX*C)) + const )/denom
        if((B==0 or BZ==0) and (BY==0 or C==0)):
            const1=math.sqrt( C*C*(-A*A*X*X+(B*B+C*C)*(L-X)*(L+X)))
            Y= ((-A*B*X)+const1)/(B*B+C*C)
            Z= -(A*C*C*X+B*const1)/(C*(B*B+C*C))
        else:
            Y= ((A*A*BY*F*L)*(B*BZ-BY*C)+ C*( -F*L*math.pow(B*BZ-BY*C,2) + BX*const) - A*( B*B*BX*BZ*F*L- B*BX*BY*F*L*C + BZ*const)) / ((B*BZ-BY*C)*denom)
            Z= ((A*A*BZ*F*L)*(B*BZ-BY*C) + (B*F*L)*math.pow(B*BZ-BY*C,2) + (A*BX*F*L*C)*(-B*BZ+BY*C) - B*BX*const + A*BY*const) / ((B*BZ-BY*C)*denom)
        # Translate the new vector to the correct coordinate
        D=np.array([X, Y, Z]) + Origin
        # Calculate how much we need to rotate the dihedral angle:
        dihedral=dihedral-dihedral_angle([u, v, Origin, D])
        # Calculate the rotation matrix
        Rotation = Rv(Origin-v,np.deg2rad(dihedral))
        # Rotate the dihedral angle by 'dihedral' degrees and translate the vector back:
        return np.squeeze(np.asarray(np.dot(Rotation,(D-v))+v))

    def add_atom(self, name, element, coor):
        index = self._selection[-1]
        if index < len(self.data['record']):
            index = len(self.data['record'])-1
        for attr in self.data:
            if attr == "e":
                setattr(self, '_'+attr, np.append(getattr(self, '_'+attr),
                                                  element))
            elif attr == "atomid":
                setattr(self, '_'+attr, np.append(getattr(self, '_'+attr),
                                                  index+1))
            elif attr == "name":
                setattr(self, '_'+attr, np.append(getattr(self, '_'+attr),
                                                  name))
            elif attr == 'coor':
                setattr(self, '_'+attr, np.append(getattr(self, '_'+attr),
                                                  np.expand_dims(coor, axis=0),
                                                  axis=0))
            else:
                setattr(self, '_'+attr, np.append(getattr(self, '_'+attr),
                                                  getattr(self, attr)[-1]))
        setattr(self, '_selection', np.append(self.__dict__['_selection'],
                                              index+1 ))
        setattr(self, 'natoms', self.natoms+1)

    def reorder(self):
        for idx,atom2 in enumerate(self._rotamers['atoms']+self._rotamers['hydrogens']):
            if self.name[idx] != atom2:
                    idx2 = np.argwhere(self.name == atom2)[0]
                    index = np.ndarray((1,),dtype='int')
                    index[0,]=np.array(self.__dict__['_selection'][0],dtype='int')
                    for attr in ["record", "name", "b", "q", "coor", "resn", "resi","icode", "e", "charge", "chain", "altloc"]:
                        self.__dict__['_'+attr][index+idx],self.__dict__['_'+attr][index+idx2]=self.__dict__['_'+attr][index+idx2],self.__dict__['_'+attr][index+idx]
