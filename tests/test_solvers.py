import inspect

import pytest

import qfit.solvers
from qfit.solvers import (
    available_miqp_solvers,
    available_qp_solvers,
    get_miqp_solver_class,
    get_qp_solver_class,
)


def test_missing_solvers() -> None:
    with pytest.raises(KeyError):
        get_qp_solver_class("NotASolver")
    with pytest.raises(KeyError):
        get_miqp_solver_class("NotASolver")


def test_get_qp_solver() -> None:
    qp_solver_class = get_qp_solver_class(next(iter(available_qp_solvers.keys())))
    assert inspect.isclass(qp_solver_class)
    assert issubclass(qp_solver_class, qfit.solvers.QPSolver)


def test_get_miqp_solver() -> None:
    miqp_solver_class = get_qp_solver_class(next(iter(available_miqp_solvers.keys())))
    assert inspect.isclass(miqp_solver_class)
    assert issubclass(miqp_solver_class, qfit.solvers.MIQPSolver)
