import math

import numpy as np
import scitbx.matrix


# TODO use scitbx.math.orthonormal_basis
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
    return _rotation_around_unit_axis(0.0, 0.0, 1.0, theta)


def Ry(theta):
    """Create a rotation matrix for rotating about y-axis.

    Args:
        theta (float): Angle of rotation in radians.

    Returns:
         np.ndarray[float]: A 3x3 rotation matrix for rotation about y.
    """
    return _rotation_around_unit_axis(0.0, 1.0, 0.0, theta)


def get_rotation_around_vector(vector, theta):
    """Create a rotation matrix for rotating about a vector.

    Args:
        vector (np.ndarray[np.float]): A (3,) vector about which to rotate.
        theta (float): Angle of rotation in radians.

    Returns:
         np.ndarray[float]: A 3x3 rotation matrix for rotation around the vector.
    """
    axis = scitbx.matrix.col(tuple(vector))
    rot_mat = axis.axis_and_angle_as_r3_rotation_matrix(theta, deg=False)
    return np.array(rot_mat).reshape((3,3))


def _rotation_around_unit_axis(u, v, w, theta):
    """Rotation matrix about the unit axis (u, v, w).

    This performs exactly the same floating-point operations as
    scitbx.matrix.col((u, v, w)).axis_and_angle_as_r3_rotation_matrix(theta)
    (axis-angle to unit quaternion to matrix), so results are identical, but
    without the overhead of building scitbx.matrix objects (~10x faster).
    The axis must already be normalized, e.g. a coordinate axis.
    """
    if not theta % (2 * math.pi):
        q0, q1, q2, q3 = 1, 0, 0, 0
    else:
        h = theta * 0.5
        c, s = math.cos(h), math.sin(h)
        q0, q1, q2, q3 = c, u * s, v * s, w * s
    return np.array((
        (2*(q0*q0+q1*q1)-1, 2*(q1*q2-q0*q3),   2*(q1*q3+q0*q2)),
        (2*(q1*q2+q0*q3),   2*(q0*q0+q2*q2)-1, 2*(q2*q3-q0*q1)),
        (2*(q1*q3-q0*q2),   2*(q2*q3+q0*q1),   2*(q0*q0+q3*q3)-1),
    ), dtype=np.float64)


def dihedral_angle(coor):
    """Calculate dihedral angle starting from four points."""
    if len(coor) != 4:
        raise ValueError(f"Computing a dihedral angle requires exactly 4 points")
    if isinstance(coor, np.ndarray):
        coor = coor.tolist()
    # adding np.round() gets us closer to the floating-point behavior of the
    # previous all-numpy implementation
    return np.round(scitbx.matrix.dihedral_angle(coor, deg=True), decimals=6)


def calc_rmsd(coor_a, coor_b):
    """Determine root-mean-square distance between two structures.

    Args:
        coor_a (np.ndarray[(n_atoms, 3), dtype=np.float]):
            Coordinates for structure a.
        coor_b (np.ndarray[(n_atoms, 3), dtype=np.float]):
            Coordinates for structure b.

    Returns:
        np.float:
            Distance between two structures.
    """
    return np.sqrt(np.mean((coor_a - coor_b) ** 2))


def adp_ellipsoid_axes(U_ij):
    """Calculate principal axes of ADP ellipsoid.

    Args:
        U_ij (np.ndarray[float]): square symmetric matrix of anisotropic
            displacement parameters (ADPs) in cartesian coordinates.
            In a PDB, ANISOU cards contain the parameters
                [u_11, u_22, u_33, u_12, u_13, u_23] / 1e-4 Å^2.

    Returns:
        List[np.ndarray[float]]: principal axes of the anisotropic
            displacement ellipsoid, from largest to smallest.
    """
    # Unscale ADP parameters to Å^2
    # U_ij = U_ij * 1e-4

    # Eigendecompose U_ij matrix with lower-triangle eigh
    eigvals, eigvecs = np.linalg.eigh(U_ij)  # pylint: disable=unused-variable

    # Scale unit eigenvectors by associated eigenvalues to return principal axes
    #   TODO: Should we? This would mean -bba/-bbs would not behave as expected.
    # directions = [e for e in (eigvals * eigvecs).T]
    return list(eigvecs.T)
