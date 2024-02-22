import unittest
import math

import numpy as np
import pytest

from qfit.structure.math import (dihedral_angle,
                                 get_rotation_around_vector,
                                 Ry,
                                 Rz,
                                 gram_schmidt_orthonormal_zx,
                                 adp_ellipsoid_axes)


def assert_matrix_approx_equals(actual, expected):
    for i in range(3):
        for j in range(3):
            assert actual[i][j] == pytest.approx(expected[i][j], abs=0.00000001)


class TestStructureMath(unittest.TestCase):
    IDENTITY_MATRIX = np.array([1,0,0,0,1,0,0,0,1]).reshape(3,3)

    def test_structure_math_dihedral_angle(self):
        coor = np.array([[-1, 1, -1], [0, 0, 0], [1, 0, 0], [2, 1, 1]])
        assert dihedral_angle(coor) == 90.0
        #assert dihedral_angle(coor.tolist()) == 90.0
        coor = np.array([[-1, 1, 0], [0, 0, 0], [1, 0, 0], [2, 1, 0]])
        assert dihedral_angle(coor) == 0

    def test_structure_math_Ry(self):
        m = Ry(math.pi/2)
        assert m.shape == (3,3)
        assert_matrix_approx_equals(m, [[0,0,1],[0,1,0],[-1,0,0]])
        m = Ry(0)
        assert_matrix_approx_equals(m, self.IDENTITY_MATRIX)

    def test_structure_math_Rz(self):
        m = Rz(math.pi/2)
        assert m.shape == (3,3)
        assert_matrix_approx_equals(m, [[0,-1,0],[1,0,0],[0,0,1]])
        m = Rz(0)
        assert_matrix_approx_equals(m, self.IDENTITY_MATRIX)

    def test_get_rotation_around_vector(self):
        # equivalent to Ry(math.pi/2)
        m = get_rotation_around_vector([0, 1, 0], math.pi/2)
        assert m.shape == (3,3)
        assert_matrix_approx_equals(m, [[0,0,1],[0,1,0],[-1,0,0]])
        m = get_rotation_around_vector(np.array([0, 1, 0]), math.pi/2)
        assert_matrix_approx_equals(m, [[0,0,1],[0,1,0],[-1,0,0]])
        # equivalent to Rz(math.pi/2)
        m = get_rotation_around_vector([0, 0, 1], math.pi/2)
        assert_matrix_approx_equals(m, [[0,-1,0],[1,0,0],[0,0,1]])
        # 45 degrees around (1,1,1) vector
        m = get_rotation_around_vector([1, 1, 1], math.pi/4)
        expected = [( 0.80473785, -0.31061721,  0.50587936),
                    ( 0.50587936,  0.80473785, -0.31061721),
                    (-0.31061721,  0.50587936,  0.80473785)]
        assert_matrix_approx_equals(m, expected)
        m = get_rotation_around_vector([1, 1, 1], 0)
        assert_matrix_approx_equals(m, self.IDENTITY_MATRIX)
        m = get_rotation_around_vector([0, 0, 0], 0)
        assert_matrix_approx_equals(m, self.IDENTITY_MATRIX)

    def test_gram_schmidt_orthonormal_zx(self):
        m = gram_schmidt_orthonormal_zx([[-1,0,0],[0,0,0],[1,1,0]])
        expected = [[ 0.        ,  0.        , -1.        ],
                    [-0.70710678,  0.70710678,  0.        ],
                    [ 0.70710678,  0.70710678,  0.        ]]
        assert_matrix_approx_equals(m, expected)
        m = gram_schmidt_orthonormal_zx([[-1,0,-1],[0,0,0],[1,1,1]])
        expected = [[ 0.70710678,  0.        , -0.70710678],
                    [-0.40824829,  0.81649658, -0.40824829],
                    [ 0.57735027,  0.57735027,  0.57735027]]
        assert_matrix_approx_equals(m, expected)

    def test_adp_ellipsoid_axes(self):
        uij = [[1465.0, -684.0, 354.0],
               [-684.0, 1150.0, -298.0],
               [354.0, -298.0, 1600.0]]
        direction = adp_ellipsoid_axes(uij)
        np.testing.assert_array_almost_equal(direction,
            [[0.61876973, 0.78542739, 0.01509452],
             [-0.43646415,  0.32774904,  0.83790191],
             [-0.65316389,  0.52505655, -0.54561209]])
