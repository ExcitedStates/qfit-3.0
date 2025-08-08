import os.path as op

import numpy as np
import pytest

from qfit.structure import Structure
from qfit.backbone import (compute_jacobian5d, compute_jacobian, AtomMoveFunctional, NullSpaceOptimizer)

from .base_test_case import UnitBase

class TestBackbone(UnitBase):

    def test_backbone_compute_jacobian(self):
        # from model AWA_single.pdb
        bb_coor = np.array([[0.666, 2.016, 2.644],
                            [2.029, 1.773, 2.187],
                            [3.023, 2.662, 2.928]])
        jacobian = compute_jacobian(bb_coor)
        np.testing.assert_array_almost_equal(
            jacobian,
            np.array([[ 0.93486358,  0.65154936],
                      [-0.16667047,  0.58272373],
                      [-0.31345022,  0.48571235],
                      [ 0.15515443,  0.        ],
                      [-1.00430344,  0.        ],
                      [ 0.99676417,  0.        ]]))

    # XXX this function is unused so it's not even clear if the test is correct
    def test_backbone_compute_jacobian_5d(self):
        # from model AWA_single.pdb
        bb_coor = np.array([[0.666, 2.016, 2.644],
                            [2.029, 1.773, 2.187],
                            [3.023, 2.662, 2.928]])
        jacobian = compute_jacobian5d(bb_coor)
        np.testing.assert_array_almost_equal(
            jacobian,
            np.array([[ 3.46393363e-01,  5.61525429e-01],
                      [-3.68484059e+00, -1.39921209e+00],
                      [ 2.99245168e+00,  9.25429513e-01],
                      [ 0.00000000e+00, -1.66533454e-16],
                      [-1.11022302e-16, -2.22044605e-16]]))

    def test_backbone_atom_move_functional(self):
        s = Structure.fromfile(op.join(self.DATA, "GNNAFNS_single.pdb"))
        ss = s.extract("resi 4 and name CB")
        endpoint = ss.coor[0] + [0.1,0.1,0.1]
        seg = list(s.segments)[0]
        amf = AtomMoveFunctional(seg, 3, "CB", endpoint)
        assert amf.target() == pytest.approx(0.03, abs=0.0000001)
        np.testing.assert_array_almost_equal(amf.gradient(), [-0.2, -0.2, -0.2])
        t, g = amf.target_and_gradient()
        assert t == pytest.approx(0.03, abs=0.0000001)
        np.testing.assert_array_almost_equal(g, [-0.2, -0.2, -0.2])
        t, g = amf.target_and_gradients_phi_psi()
        assert t == pytest.approx(0.03, abs=0.0000001)
        np.testing.assert_array_almost_equal(
            g,
            [[-0.25243894, -3.83085508,  1.14856205],
             [-0.79223472, -0.3292945 , -2.54322371],
             [-0.39439932,  0.35764835, -2.4550739 ],
             [ 0.08692137, -0.55717302,  0.31931969],
             [ 0.34165661, -1.49555218,  0.21906015],
             [-0.52578009, -0.25108976, -0.70016494],
             [-0.40773945, -1.12152905, -0.35088628],
             [ 0.        ,  0.        ,  0.        ],
             [ 0.        ,  0.        ,  0.        ],
             [ 0.        ,  0.        ,  0.        ],
             [ 0.        ,  0.        ,  0.        ],
             [ 0.        ,  0.        ,  0.        ],
             [ 0.        ,  0.        ,  0.        ],
             [ 0.        ,  0.        ,  0.        ]])

    def test_backbone_null_space_optimizer(self):
        s = Structure.fromfile(op.join(self.DATA, "GNNAFNS_single.pdb"))
        ss = s.extract("resi 4 and name CB")
        endpoint = ss.coor[0] + [0.1,0.1,0.1]
        seg = list(s.segments)[0]
        nso = NullSpaceOptimizer(seg)
        assert nso.ndofs == 14
        result = nso.optimize("CB", endpoint)
        assert result.success
        assert result.message.replace("_", " ") == 'CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL'
        np.testing.assert_array_almost_equal(
            result.x,
            [-1.15240586,  0.48723203,  0.2311406 , -0.59913063, -1.28003427,
              1.4423638 ,  1.64459587,  2.19054824,  2.77618331, -3.20287006,
              2.87392182, -3.30959218, -0.22161363, -2.57777175])
        t, g = nso.target_and_gradient(np.zeros(nso.ndofs, float))
        assert t == pytest.approx(0.03, abs=0.0000001)
        np.testing.assert_array_almost_equal(
            g,
            [ 0.00166658, -0.00087116, -0.00145954, -0.00443921, -0.00597848,
             -0.00602664, -0.00603923,  0.0013946 ,  0.00126556,  0.00923786,
              0.00180946,  0.0046722 ,  0.00263691,  0.00450347])
