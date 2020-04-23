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


def gram_schmidt_orthonormal_zx(coords):
    """Create an orthonormal basis from atom vectors z & x.

    This function preserves the direction of the z atom-vector.
    This is based off the Gram-Schmidt process.

    Args:
        coords (np.ndarray[float]): A nx3 matrix of row-vectors,
            with each row containing the position of an atom.

    Returns:
        np.ndarray[float]: A 3x3 orthonormal set of row-vectors,
            where the z-axis is parallel to the input z-atom vector.
    """
    zaxis = coords[2] / np.linalg.norm(coords[2])
    yaxis = coords[0] - np.inner(coords[0], zaxis) * zaxis
    yaxis /= np.linalg.norm(yaxis)
    xaxis = np.cross(yaxis, zaxis)
    return np.vstack((xaxis, yaxis, zaxis))


def Rz(theta):
    """Create a rotation matrix for rotating about z-axis.

    Args:
        theta (float): Angle of rotation in radians.

    Returns:
         np.ndarray[float]: A 3x3 rotation matrix for rotation about z.
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.array([[cos_theta, -sin_theta, 0],
                     [sin_theta,  cos_theta, 0],
                     [        0,          0, 1]])


def Ry(theta):
    """Create a rotation matrix for rotating about y-axis.

    Args:
        theta (float): Angle of rotation in radians.

    Returns:
         np.ndarray[float]: A 3x3 rotation matrix for rotation about y.
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.array([[ cos_theta, 0, sin_theta],
                     [         0, 1,         0],
                     [-sin_theta, 0, cos_theta]])


def Rv(vector, theta):
    """Create a rotation matrix for rotating about a vector.

    Args:
        vector (np.ndarray[np.float]): A (3,) vector about which to rotate.
        theta (float): Angle of rotation in radians.

    Returns:
         np.ndarray[float]: A 3x3 rotation matrix for rotation about z.
    """
    (x, y, z) = vector / np.linalg.norm(vector)
    K = np.array([[ 0, -z,  y],
                  [ z,  0, -x],
                  [-y,  x,  0]])
    rot = np.cos(theta) * np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.outer(vector, vector)
    return rot


def aa_to_rotmat(axis, angle):
    """Create a rotation matrix for a given Euler axis, angle rotation.

    Args:
        axis (np.ndarray[np.float]): A (3,) vector about which to rotate.
        theta (float): Angle of rotation in radians.

    Returns:
         np.ndarray[float]: A 3x3 rotation matrix for rotation about axis.
    """
    kx, ky, kz = axis
    K = np.array([[  0, -kz,  ky],
                  [ kz,   0, -kx],
                  [-ky,  kx,   0]])
    K2 = K @ K
    R = np.identity(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K2
    return R


def dihedral_angle(coor):
    """Calculate dihedral angle starting from four points."""
    b1 = coor[0] - coor[1]
    b2 = coor[3] - coor[2]
    b3 = coor[2] - coor[1]
    n1 = np.cross(b3, b1)
    n2 = np.cross(b3, b2)
    m1 = np.cross(n1, n2)

    norm = np.linalg.norm
    normfactor = norm(n1) * norm(n2)
    sinv = norm(m1) / normfactor
    cosv = np.inner(n1, n2) / normfactor
    angle = np.rad2deg(np.arctan2(sinv, cosv))

    # Check sign of angle
    u = np.cross(n1, n2)
    if np.inner(u, b3) < 0:
        angle *= -1
    return angle
