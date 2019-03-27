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

import os.path

import numpy as np


class BackboneRotator:

    """Rotate around phi, psi angles."""

    def __init__(self, segment):

        self.segment = segment
        self.ndofs = 2 * len(segment.residues)
        self._starting_coor = segment.coor

        # Check for each rotation which atoms are affected. Start with the last residue.
        self._aligners = []
        self._origins = []
        selections = []
        for n, residue in enumerate(self.segment.residues[::-1]):
            psi_sel = residue.select('name', ('O', 'OXT'))
            if n > 0:
                psi_sel = np.concatenate((psi_sel, self.segment.residues[-n]._selection))
            phi_sel = residue.select('name', ('N', 'CA', 'O', 'OXT','H','HA'), '!=')
            selections += [psi_sel, phi_sel]

            N = residue.extract('name', 'N')
            CA = residue.extract('name' ,'CA')
            C = residue.extract('name', 'C')
            axis = C.coor[0] - CA.coor[0]
            psi_aligner = ZAxisAligner(axis)
            axis = CA.coor[0] - N.coor[0]
            phi_aligner = ZAxisAligner(axis)

            self._aligners += [psi_aligner, phi_aligner]
            self._origins += [C.coor[0], CA.coor[0]]

        self._atoms_to_rotate = []
        for selection in selections:
            atoms_to_rotate = []
            if self._atoms_to_rotate:
                atoms_to_rotate = np.concatenate((self._atoms_to_rotate))
            atoms_to_rotate = np.concatenate((atoms_to_rotate, selection)).astype(np.int32)
            self._atoms_to_rotate.append(np.unique(atoms_to_rotate))

    def __call__(self, torsions):

        assert len(torsions) == self.ndofs, "Number of torsions should equal "
        " degrees of freedom"

        # We start with the last torsion as this is more efficient
        torsions = np.deg2rad(torsions[::-1])

        self.segment.coor = self._starting_coor
        atoms_to_rotate = []
        iterator = zip(torsions, self._origins, self._aligners, self._atoms_to_rotate)
        for torsion, origin, aligner, atoms_to_rotate in iterator:
            if torsion == 0.0:
                continue
            coor = self.segment._coor[atoms_to_rotate]
            coor -= origin
            R = np.asarray(aligner.forward_rotation * Rz(torsion) * aligner.backward_rotation)
            coor = np.dot(coor, R.T)
            coor += origin
            self.segment._coor[atoms_to_rotate] = coor


class Translator:

    def __init__(self, ligand):
        self.ligand = ligand
        self.coor_to_translate = self.ligand.coor

    def __call__(self, trans):
        self.ligand.coor = self.coor_to_translate + np.asarray(trans)


class CBAngleRotator:

    def __init__(self, residue):
        self.residue = residue
        angle_selection = residue.select('name', ('N', 'CA', 'CB'))
        if angle_selection.size != 3:
            raise RuntimeError("Residue does not have N, CA and CB atom for rotation.")
        self.atoms_to_rotate = residue.select('name', ('N', 'CA', 'C', 'O','H','HA'), '!=')
        self._origin = self.residue.extract('name', 'CA').coor[0]
        self._coor_to_rotate = self.residue._coor[self.atoms_to_rotate]
        self._coor_to_rotate -= self._origin
        axis_coor = residue.extract('name', ('N', 'CB')).coor
        axis_coor -= self._origin
        axis = np.cross(axis_coor[0], axis_coor[1])
        axis /= np.linalg.norm(axis)
        aligner = ZAxisAligner(axis)
        self._forward = aligner.forward_rotation
        self._coor_to_rotate = (aligner.backward_rotation *
                np.asmatrix(self._coor_to_rotate.T)).T

    def __call__(self, angle):
        # Since the axis of rotation is already aligned with the z-axis, we can
        # freely rotate the coordinates and perform the inverse operation to realign the
        # axis to the real world frame.
        R = self._forward * np.asmatrix(Rz(np.deg2rad(angle)))
        self.residue._coor[self.atoms_to_rotate] = (R * self._coor_to_rotate.T).T + self._origin


class GlobalRotator:

    """Rotate ligand around its center."""

    def __init__(self, ligand, center=None):

        self.ligand = ligand
        self._center = center
        ligand_coor = self.ligand.coor
        if self._center is None:
            self._center = ligand_coor.mean(axis=0)
        self._coor_to_rotate = np.asmatrix(ligand_coor - self._center)
        self._intermediate = np.zeros_like(ligand_coor)

    def __call__(self, rotmat):
        np.dot(rotmat, self._coor_to_rotate.T, self._intermediate.T)
        self._intermediate += self._center
        self.ligand.coor = self._intermediate


class PrincipalAxisRotator:

    """Rotate ligand along the principal axes."""

    def __init__(self, ligand):
        self.ligand = ligand
        self._center = ligand.coor.mean(axis=0)
        self._coor_to_rotate = self.ligand.coor - self._center
        gyration_tensor = np.asmatrix(self._coor_to_rotate).T * np.asmatrix(self._coor_to_rotate)
        eig_values, eig_vectors = np.linalg.eigh(gyration_tensor)
        # Sort eigenvalues such that lx <= ly <= lz
        sort_ind = np.argsort(eig_values)
        self.principal_axes = np.asarray(eig_vectors[:, sort_ind].T)

        self.aligners = [ZAxisAligner(axis) for axis in self.principal_axes]

    def __call__(self, angle, axis=2):
        aligner = self.aligners[axis]
        R = aligner.forward_rotation * np.asmatrix(Rz(angle)) * aligner.backward_rotation
        self.ligand.coor[:] = (R * self._coor_to_rotate.T).T + self._center


# TODO Make a super class combining the BondRotator with the AngleRotator or at
# refactorize code.
class BondAngleRotator:

    """Rotate ligand along a bond angle defined by three atoms."""

    def __init__(self, ligand, a1, a2, a3, key='name'):
        # Atoms connected to a1 will stay fixed.
        self.ligand = ligand
        self.atom1 = a1
        self.atom2 = a2
        self.atom3 = a3

        # Determine which atoms will be moved by the rotation.
        self._root = getattr(ligand, key).tolist().index(a2)
        self._conn = ligand.connectivity
        self.atoms_to_rotate = [self._root]
        self._foundroot = 0
        curr = getattr(ligand, key).tolist().index(a3)
        self._find_neighbours_recursively(curr)
        if self._foundroot > 1:
            raise ValueError("Atoms are part of a ring. Bond angle cannot be rotated.")

        # Find the rigid motion that aligns the axis of rotation onto the z-axis.
        self._coor_to_rotate = self.ligand.coor[self.atoms_to_rotate]
        # Move root to origin
        self._t = self.ligand.coor[self._root]
        self._coor_to_rotate -= self._t
        # The rotation axis is the cross product between a1 and a3.
        a1_coor = self.ligand.coor[getattr(ligand, key).tolist().index(a1)]
        axis = np.cross(a1_coor - self._t, self._coor_to_rotate[1])

        # Align the rotation axis to the z-axis for the coordinates
        aligner = ZAxisAligner(axis)
        self._forward = aligner.forward_rotation
        self._coor_to_rotate = (aligner.backward_rotation *
                np.asmatrix(self._coor_to_rotate.T)).T

    def _find_neighbours_recursively(self, curr):
        self.atoms_to_rotate.append(curr)
        bonds = np.flatnonzero(self._conn[curr])
        for b in bonds:
            if b == self._root:
                self._foundroot += 1
            if b not in self.atoms_to_rotate:
                self._find_neighbours_recursively(b)

    def __call__(self, angle):

        # Since the axis of rotation is already aligned with the z-axis, we can
        # freely rotate the coordinates and perform the inverse operation to realign the
        # axis to the real world frame.
        R = self._forward * np.asmatrix(Rz(angle))
        self.ligand.coor[self.atoms_to_rotate] = (R * self._coor_to_rotate.T).T + self._t


class ChiRotator:

    """Rotate a residue around a chi-angle"""

    def __init__(self, residue, chi_index, covalent=None):
        self.residue = residue
        self.chi_index = chi_index
        # Get the coordinates that define the torsion angle
        torsion_atoms = self.residue._rotamers['chi'][chi_index]
        selection = self.residue.select('name', torsion_atoms)
        new_selection = []
        for atom in torsion_atoms:
            for sel in selection:
                if atom == self.residue._name[sel]:
                    new_selection.append(sel)
                    break
        selection = new_selection

        # Build a coordinate frame around it using Gram-Schmidt orthogonalization
        norm = np.linalg.norm
        coor = self.residue._coor[selection]
        self._origin = coor[1].copy()
        coor -= self._origin
        zaxis = coor[2]
        zaxis /= norm(zaxis)
        yaxis = coor[0] - np.inner(coor[0], zaxis) * zaxis
        yaxis /= norm(yaxis)
        xaxis = np.cross(yaxis, zaxis)
        self._backward = np.asmatrix(np.zeros((3, 3), float))
        self._backward[0] = xaxis
        self._backward[1] = yaxis
        self._backward[2] = zaxis
        self._forward= self._backward.T.copy()

        # Save the coordinates aligned along the Z-axis for fast future rotation
        atoms_to_rotate = self.residue._rotamers['chi-rotate'][chi_index]
        self._atom_selection = self.residue.select('name', atoms_to_rotate)
        if covalent in atoms_to_rotate:
            # If we are rotating the atom that is covalently bonded
            # to the ligand, we should also rotate the ligand.
            atoms_to_rotate2 = self.residue.name[6:]
            atom_selection2 = self.residue.select('name', atoms_to_rotate2)
            tmp = list(self._atom_selection)
            tmp += list(atom_selection2)
            self._atom_selection = np.array(tmp, dtype=int)
        self._coor_to_rotate = np.dot(
            self.residue._coor[self._atom_selection] - self._origin, self._backward.T)
        self._tmp = np.zeros_like(self._coor_to_rotate)

    def __call__(self, angle):
        R = self._forward * Rz(np.deg2rad(angle))
        np.dot(self._coor_to_rotate, R.T, self._tmp)
        self._tmp += self._origin
        self.residue._coor[self._atom_selection] = self._tmp


class BondRotator:

    """Rotate ligand along the bond of two atoms."""

    def __init__(self, ligand, a1, a2, key='name'):
        # Atoms connected to a1 will stay fixed.
        self.ligand = ligand
        self.atom1 = a1
        self.atom2 = a2

        # Determine which atoms will be moved by the rotation.
        self._root = getattr(ligand, key).tolist().index(a1)
        self._conn = ligand.connectivity
        self.atoms_to_rotate = [self._root]
        self._foundroot = 0
        curr = getattr(ligand, key).tolist().index(a2)
        self._find_neighbours_recursively(curr)
        #if self._foundroot > 1:
        #    raise ValueError("Atoms are part of a ring. Bond cannot be rotated.")

        # Find the rigid motion that aligns the axis of rotation onto the z-axis.
        self._coor_to_rotate = self.ligand.coor[self.atoms_to_rotate].copy()
        # Move root to origin
        self._t = self.ligand.coor[self._root]
        self._coor_to_rotate -= self._t
        # Find angle between rotation axis and x-axis
        axis = self._coor_to_rotate[1] / np.linalg.norm(self._coor_to_rotate[1,:-1])
        aligner = ZAxisAligner(axis)

        # Align the rotation axis to the z-axis for the coordinates
        self._forward = aligner.forward_rotation
        self._coor_to_rotate = (aligner.backward_rotation *
                np.asmatrix(self._coor_to_rotate.T)).T

    def _find_neighbours_recursively(self, curr):
        self.atoms_to_rotate.append(curr)
        bonds = np.flatnonzero(self._conn[curr])
        for b in bonds:
            if b == self._root:
                self._foundroot += 1
            if b not in self.atoms_to_rotate:
                self._find_neighbours_recursively(b)

    def __call__(self, angle):

        # Since the axis of rotation is already aligned with the z-axis, we can
        # freely rotate them and perform the inverse operation to realign the
        # axis to the real world frame.
        R = self._forward * np.asmatrix(Rz(angle))
        self.ligand.coor[self.atoms_to_rotate] = (R * self._coor_to_rotate.T).T + self._t


class ZAxisAligner:

    """Find the rotation that aligns a vector to the Z-axis."""

    def __init__(self, axis):
        # Find angle between rotation axis and x-axis
        axis = axis / np.linalg.norm(axis[:-1])
        xaxis_angle = np.arccos(axis[0])
        if axis[1] < 0:
            xaxis_angle *= -1
        # Rotate around Z-axis
        self._Rz = Rz(xaxis_angle)
        axis = np.dot(self._Rz.T, axis.reshape(3, -1)).ravel()
        # Find angle between rotation axis and z-axis
        zaxis_angle = np.arccos(axis[2] / np.linalg.norm(axis))
        if axis[0] < 0:
            zaxis_angle *= -1
        self._Ry = Ry(zaxis_angle)
        # Check whether the transformation is correct.
        # Rotate around the Y-axis to align to the Z-axis.
        axis = np.dot(self._Ry.T, axis.reshape(3, -1)).ravel() / np.linalg.norm(axis)
        if not np.allclose(axis, [0, 0, 1]):
            print(axis)
            raise ValueError("Axis is not aligned to z-axis.")
        self.backward_rotation = np.asmatrix(self._Ry).T * np.asmatrix(self._Rz).T
        self.forward_rotation = np.asmatrix(self._Rz) * np.asmatrix(self._Ry)


def Rz(theta):
    """Rotate along z-axis."""
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.asarray([[cos_theta, -sin_theta, 0],
                       [sin_theta,  cos_theta, 0],
                       [        0,          0, 1]])


def Ry(theta):
    """Rotate along y-axis."""
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.asarray([[ cos_theta, 0, sin_theta],
                       [         0, 1,         0],
                       [-sin_theta, 0, cos_theta]])


def aa_to_rotmat(axis, angle):
    """Axis angle to rotation matrix."""

    kx, ky, kz = axis
    K = np.asmatrix([[0, -kz, ky],
                     [kz, 0, -kx],
                     [-ky, kx, 0]])
    K2 = K * K
    R = np.identity(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K2
    return R


class RotationSets:

    LOCAL = (('local_5_10.npy', 10, 5.00),
             ('local_5_100.npy', 100, 5.00),
             ('local_5_1000.npy', 1000, 5.00),
             ('local_10_10.npy', 10, 10.00),
             ('local_10_100.npy', 100, 10.00),
             ('local_10_1000.npy', 1000, 10.00),
             )

    _DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), 'data')

    @classmethod
    def get_set(cls, angle):
        angles = zip(*cls.SETS)[-1]
        diff = [abs(a - angle) for a in angles]
        fname = cls.SETS[diff.index(min(diff))][0]
        with open(os.path.join(cls._DATA_DIRECTORY, fname)) as f:
            quat_weights = np.load(f)
        return cls.quat_to_rotmat(quat_weights[:, :4])

    @classmethod
    def get_local_set(cls, fname='local_10_10.npy'):
        quats = np.load(os.path.join(cls._DATA_DIRECTORY, fname))
        return cls.quat_to_rotmat(quats)

    @classmethod
    def local(cls, max_angle, nrots=100):
        quats = []
        radian_max_angle = np.deg2rad(max_angle)
        while len(quats) < nrots - 1:
            quat = cls.random_rotmat(matrix=False)
            angle = 2 * np.arccos(quat[0])
            if angle <= radian_max_angle:
                quats.append(quat)
        quats.append(np.asarray([1, 0, 0, 0], dtype=np.float64))
        return np.asarray(quats)

    @staticmethod
    def quat_to_rotmat(quaternions):

        quaternions = np.asarray(quaternions)

        w = quaternions[:, 0]
        x = quaternions[:, 1]
        y = quaternions[:, 2]
        z = quaternions[:, 3]

        Nq = w**2 + x**2 + y**2 + z**2
        s = np.zeros(Nq.shape, dtype=np.float64)
        s[Nq >  0.0] = 2.0/Nq[Nq > 0.0]
        s[Nq <= 0.0] = 0

        X = x*s
        Y = y*s
        Z = z*s

        rotmat = np.zeros((quaternions.shape[0],3,3), dtype=np.float64)
        rotmat[:,0,0] = 1.0 - (y*Y + z*Z)
        rotmat[:,0,1] = x*Y - w*Z
        rotmat[:,0,2] = x*Z + w*Y

        rotmat[:,1,0] = x*Y + w*Z
        rotmat[:,1,1] = 1.0 - (x*X + z*Z)
        rotmat[:,1,2] = y*Z - w*X

        rotmat[:,2,0] = x*Z - w*Y
        rotmat[:,2,1] = y*Z + w*X
        rotmat[:,2,2] = 1.0 - (x*X + y*Y)

        np.around(rotmat, decimals=8, out=rotmat)

        return rotmat

    @classmethod
    def random_rotmat(cls, matrix=True):
        """Return a random rotation matrix"""

        s1 = 1
        while s1 >= 1.0:
            e1 = np.random.random() * 2 - 1
            e2 = np.random.random() * 2 - 1
            s1 = e1**2 + e2**2

        s2 = 1
        while s2 >= 1.0:
            e3 = np.random.random() * 2 - 1
            e4 = np.random.random() * 2 - 1
            s2 = e3**2 + e4**2

        q0 = e1
        q1 = e2
        q2 = e3 * np.sqrt((1 - s1)/s2 )
        q3 = e4 * np.sqrt((1 - s1)/s2 )

        quat = [q0, q1, q2, q3]
        if matrix:
            return cls.quat_to_rotmat(np.asarray(quat).reshape(1, 4))[0]
        else:
            return quat
