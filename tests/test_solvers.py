import pytest

from qfit.solvers import (QPSolver, MIQPSolver)


def setup_module(module):  # pylint: disable=unused-argument
    import qfit.solvers

    # Manually set CPLEX to False
    # in this module ONLY
    qfit.solvers.CPLEX = False


class TestForcedNoSolvers:
    # QPSolver.__new__ should raise ImportError
    # when CPLEX flags are false
    def test_QPSolvers_fail(self):
        with pytest.raises(ImportError):
            QPSolver(None, None, use_cplex=False)
        with pytest.raises(ImportError):
            QPSolver(None, None, use_cplex=True)

    # MIQPSolver.__new__ should raise ImportError
    # when CPLEX flags are false
    def test_MIQPSolvers_fail(self):
        with pytest.raises(ImportError):
            MIQPSolver(None, None, use_cplex=False)
        with pytest.raises(ImportError):
            MIQPSolver(None, None, use_cplex=True)
