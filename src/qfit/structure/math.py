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


def Rz(theta):
    """Rotate along z-axis."""
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.array([[cos_theta, -sin_theta, 0],
                     [sin_theta,  cos_theta, 0],
                     [        0,          0, 1]])


def Rv(vector, theta):
    """Rotate along a vector."""
    vector = vector / np.linalg.norm(vector)
    x, y, z = vector
    K = np.asmatrix([[ 0, -z,  y],
                     [ z,  0, -x],
                     [-y,  x,  0]])
    rot = np.cos(theta) * np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.outer(vector, vector)
    return rot


def aa_to_rotmat(axis, angle):
    """Axis angle to rotation matrix."""
    kx, ky, kz = axis
    K = np.asmatrix([[  0, -kz,  ky],
                     [ kz,   0, -kx],
                     [-ky,  kx,   0]])
    K2 = K * K
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
