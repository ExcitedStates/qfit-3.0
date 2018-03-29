import numpy as np

def Rz(theta):
    """Rotate along z-axis."""
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.asarray([[cos_theta, -sin_theta, 0],
                       [sin_theta,  cos_theta, 0],
                       [        0,          0, 1]])


def aa_to_rotmat(axis, angle):
    """Axis angle to rotation matrix."""

    kx, ky, kz = axis
    K = np.asmatrix([[0, -kz, ky],
                     [kz, 0, -kx],
                     [-ky, kx, 0]])
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
