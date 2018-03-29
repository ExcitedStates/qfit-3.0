
import numpy as np


def compute_jacobian5d(bb_coor):
    """Compute the 5D Jacobian for null space computation.

    bb_coor : Coordinates of sequential N, CA, and C atoms.

    endpoint : Coordinate of fixed endpoint.
    """

    nresidues = bb_coor.shape[0] // 3
    N_coor = bb_coor[::3]
    CA_coor = bb_coor[1::3]
    C_coor = bb_coor[2::3]

    # Use notations as used in Budday, Lyendecker and Van den Bedem (2016).
    fh = bb_coor[0]
    fd = bb_coor[1]
    fa = bb_coor[-1]
    faa = bb_coor[-2]
    fh_fa = fh + fa
    ndofs = nresidues * 2
    #jacobian = np.zeros((ndofs, 5), dtype=np.float64)
    jacobian = np.zeros((5, ndofs), dtype=np.float64).T
    norm = np.linalg.norm
    # N -> CA rotations
    # Relative distance constraints
    r = CA_coor - N_coor
    r /= norm(r, axis=1).reshape(-1, 1)
    jacobian[::2, :3] = np.cross(r, fh_fa - N_coor)
    # Orientation constraints
    f = np.asmatrix(fa - fh)
    dfh_dq = np.cross(r, fh - N_coor)
    dfd_dq = np.cross(r, fd - N_coor)
    jacobian[::2, 3] = f * np.asmatrix(dfh_dq - dfd_dq).T

    f = np.asmatrix(fa - faa)
    dfa_dq = np.cross(r, fa - N_coor)
    jacobian[::2, 4] = f * np.asmatrix(dfh_dq - dfa_dq).T

    # C -> CA rotations
    # Relative distance constraints
    r = C_coor - CA_coor
    r /= norm(r, axis=1).reshape(-1, 1)
    jacobian[1::2, :3] = np.cross(r, fh_fa - C_coor)
    # Orientation constraints
    f = np.asmatrix(fa - fh)
    dfh_dq = np.cross(r, fh - CA_coor)
    dfd_dq = np.cross(r, fd - CA_coor)
    jacobian[1::2, 3] = f * np.asmatrix(dfh_dq - dfd_dq).T

    f = np.asmatrix(fa - faa)
    dfa_dq = np.cross(r, fa - CA_coor)
    jacobian[1::2, 4] = f * np.asmatrix(dfh_dq - dfa_dq).T

    return jacobian.T


def compute_jacobian(bb_coor):
    """Compute the 6D Jacobian for null space computation.

    bb_coor : Coordinates of sequential N, CA, and C atoms.
    """

    nresidues = bb_coor.shape[0] // 3
    N_coor = bb_coor[::3]
    CA_coor = bb_coor[1::3]
    C_coor = bb_coor[2::3]

    ndofs = nresidues * 2
    jacobian = np.zeros((6, ndofs), dtype=np.float64).T
    norm = np.linalg.norm
    # N -> CA rotations
    # Relative distance constraints
    t1 = CA_coor - N_coor
    t1 /= norm(t1, axis=1).reshape(-1, 1)
    c1 = np.cross(t1, bb_coor[-1] - CA_coor)
    jacobian[::2, :3] = t1
    jacobian[::2, 3:] = c1

    # C -> CA rotations
    # Relative distance constraints
    t1 = C_coor - CA_coor
    t1 /= norm(t1, axis=1).reshape(-1, 1)
    c1 = np.cross(t1, bb_coor[-1] - C_coor)
    jacobian[1::2, :3] = t1
    jacobian[1::2, 3:] = c1

    return jacobian.T


def compute_null_space(jacobian):

    _, s, v = np.linalg.svd(jacobian)
    nonredundant = (s > 1e-10).sum()
    return v[:, nonredundant:]


def project_on_null_space(null_space, gradients):
    null_space = np.asmatrix(null_space)
    projection = null_space * null_space.T
    return projection * gradients


def test_jacobian():
    nresidues = 10
    bb_coor = np.random.rand(3 * 3 * nresidues).reshape(nresidues * 3, 3)
    jacobian = compute_jacobian(bb_coor)
    return jacobian


def test_compute_null_space():
    np.random.seed(0)
    nresidues = 10
    bb_coor = np.random.rand(3 * 3 * nresidues).reshape(nresidues * 3, 3)
    endpoint = bb_coor[-1]
    jacobian = compute_jacobian(bb_coor)
    null_space = compute_null_space(jacobian)


if __name__ == '__main__':
    jacobian = test_jacobian()
    test_compute_null_space()
