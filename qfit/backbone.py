import numpy as np
import scipy as sp

from .samplers import BackboneRotator


def move_direction_adp(u_matrix, unit_cell):

    u_matrix = np.asmatrix(u_matrix)
    orth_matrix = np.asmatrix(unit_cell.frac_to_orth)
    metric_tensor = orth_matrix.T * orth_matrix
    u_orth = metric_tensor * u_matrix * metric_tensor
    eigval, eigvec = np.linalg.eigh(u_orth)
    order = np.argsort(eigval)
    eigvec = eigvec[order]
    if np.linalg.det(eigvec) < 0:
        rotmat[:, 0] *= -1
    eigensum = np.zeros(3)
    for i in range(3):
        eigensum[i] = (eigval[i] * eigvec[i]).sum()
    eigensum /= np.linalg.norm(eigensum)

    directions = [eigvec[0], eigvec[1], eigvec[2], eigensum]
    return directions


def compute_jacobian5d(bb_coor):
    """Compute the 5D Jacobian for null space computation.

    bb_coor : Coordinates of sequential N, CA, and C atoms.
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


def project_on_null_space(null_space, gradients):
    null_space = np.asmatrix(null_space)
    projection = null_space * null_space.T
    return projection * gradients


class CBMoveFunctional:

    """Functional for obtaining energy and gradient to move a CB atom"""

    def __init__(self, segment, residue_index, endpoint):
        self.segment = segment
        self.residue_index = residue_index
        residue = self.segment[residue_index]
        self._cb_index = residue.select('name', 'CB')[0]
        self.endpoint = endpoint

    def target(self):

        current = self.segment._coor[self._cb_index]
        diff = current - self.endpoint
        energy = np.dot(diff, diff)
        return energy

    def gradient(self):

        """Return the gradient on the CB atom."""

        current = self.segment._coor[self._cb_index]
        diff = current - self.endpoint
        return 2 * diff

    def target_and_gradient(self):
        current = self.segment._coor[self._cb_index]
        diff = current - self.endpoint
        energy = np.dot(diff, diff)
        gradient = 2 * diff
        return energy, gradient

    def target_and_gradients_phi_psi(self):

        """Return the gradients by rotating along each phi and psi backbone angle."""

        target, gradient = self.target_and_gradient()
        normal = gradient / np.linalg.norm(gradient)
        gradients = np.zeros((len(self.segment) * 2, 3), float)
        current = self.segment._coor[self._cb_index]
        for n, residue in enumerate(self.segment.residues):
            # Residues after the selected CB residue have no impact on the CB
            # position. The backbone torsion gradients will be zero.
            if n > self.residue_index:
                continue
            N = residue.extract('name', 'N')
            CA = residue.extract('name', 'CA')
            C = residue.extract('name', 'C')
            origin = N.coor[0]
            phi_axis = CA.coor[0] - origin
            phi_axis /= np.linalg.norm(phi_axis)
            phi_gradient = np.cross(phi_axis, current - origin)
            phi_gradient_unit = phi_gradient / np.linalg.norm(phi_gradient)
            gradients[2 * n] = np.dot(phi_gradient_unit, normal) * phi_gradient

            if n == self.residue_index:
                continue
            origin = CA.coor[0]
            psi_axis = C.coor[0] - origin
            psi_axis /= np.linalg.norm(psi_axis)
            psi_gradient = np.cross(psi_axis, current - origin)
            psi_gradient_unit = psi_gradient / np.linalg.norm(phi_gradient)
            gradients[2 * n + 1] = np.dot(psi_gradient_unit, normal) * psi_gradient
        return target, gradients


class NullSpaceOptimizer:

    def __init__(self, segment, endpoint):

        self.segment = segment
        self.ndofs = len(segment) * 2
        self._bb_selection = np.sort(self.segment.select('name', ('N', 'CA', 'C')))
        self.rotator = BackboneRotator(segment)
        self._starting_coor = self.segment.coor

        residue_index = int(len(self.segment) / 2.0)
        self._functional = CBMoveFunctional(segment, residue_index, endpoint)

    def optimize(self):

        torsions = np.zeros(self.ndofs, float)

        options = {'disp': True}
        minimize = sp.optimize.minimize
        result = minimize(self.target_and_gradient, torsions,
                          method='L-BFGS-B', jac=True, options=options)
        return result

    def target_and_gradient(self, torsions):

        self.rotator(torsions)
        #target, gradient = self._functional.target_and_gradients_phi_psi()
        target = self._functional.target()

        tmp = torsions.copy()
        delta = 1e-4
        gradients = np.zeros(torsions.size)
        for n in range(torsions.size):
            tmp[n] += delta
            self.rotator(tmp)
            fp = self._functional.target()

            tmp[n] -= 2 * delta
            self.rotator(tmp)
            fn = self._functional.target()

            gradients[n] = (fp - fn) / (2 * delta)
            tmp[n] += delta

        bb_coor = self.segment._coor[self._bb_selection]
        jacobian = compute_jacobian(bb_coor)
        null_space = sp.linalg.null_space(jacobian)
        null_space = np.asmatrix(null_space)
        projector = np.asarray(null_space * null_space.T)
        null_space_gradients = np.dot(projector, gradients)
        self.segment.coor = self._starting_coor
        return target, null_space_gradients
