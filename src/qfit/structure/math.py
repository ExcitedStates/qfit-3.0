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
    return get_rotation_around_vector([0,0,1], theta)


def Ry(theta):
    """Create a rotation matrix for rotating about y-axis.

    Args:
        theta (float): Angle of rotation in radians.

    Returns:
         np.ndarray[float]: A 3x3 rotation matrix for rotation about y.
    """
    return get_rotation_around_vector([0,1,0], theta)


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
