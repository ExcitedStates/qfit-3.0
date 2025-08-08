import unittest
import math

import numpy as np
import pytest

from qfit.structure.math import (dihedral_angle,
                                 Ry,
                                 Rz,
                                 gram_schmidt_orthonormal_zx,
                                 )


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
