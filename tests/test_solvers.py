import inspect

import numpy as np
import pytest
import scipy.sparse
from numpy.typing import NDArray

import qfit.solvers
from qfit.solvers import (
    available_miqp_solvers,
    available_qp_solvers,
    get_miqp_solver_class,
    get_qp_solver_class,
    is_none_or_empty,
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


def test_is_none_or_empty() -> None:
    """If the array is empty, this function should return True."""
    assert is_none_or_empty(None) is True

    assert is_none_or_empty([]) is True
    assert is_none_or_empty([False]) is False
    assert is_none_or_empty([object()]) is False

    assert is_none_or_empty(np.array([])) is True
    assert is_none_or_empty(np.array([0.1])) is False

    assert (
        is_none_or_empty(
            scipy.sparse.csc_matrix(
                ([], ([], [])),
                shape=(1, 0),
            )
        )
        is True
    )
    assert (
        is_none_or_empty(
            scipy.sparse.csc_matrix(
                ([1.0], ([0], [0])),
                shape=(1, 1),
            )
        )
        is False
    )


@pytest.mark.parametrize("solver_class", available_qp_solvers.values())
class TestQPSolver:
    target = np.array([2.0, 3.0, 7.0])
    models = np.array([[6.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 21.0]])

    def test_qp_solver(self, solver_class: type[qfit.solvers.QPSolver]) -> None:
        solver = solver_class(self.target, self.models)
        solver.solve_qp()

        assert np.allclose(solver.weights, [1 / 3, 1 / 3, 1 / 3], atol=1e-3)
        assert np.isclose(solver.objective_value, 0.0, atol=1e-6)


@pytest.mark.parametrize("solver_class", available_miqp_solvers.values())
class TestMIQPSolver:
    target = np.array([2.0, 3.0, 7.0])
    models = np.array([[6.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 21.0]])

    def expected_objective(self, expected_weights: NDArray[np.float_]) -> float:
        return np.sum(np.square(np.inner(self.models, expected_weights) - self.target))

    def test_miqp_solver_with_threshold(
        self, solver_class: type[qfit.solvers.MIQPSolver]
    ) -> None:
        solver = solver_class(self.target, self.models)
        solver.solve_miqp(threshold=0.4)

        expected_weights = np.array([0.0, 0.4, 0.4])
        expected_objective = self.expected_objective(expected_weights)

        assert np.allclose(solver.weights, expected_weights, atol=1e-3)
        assert np.isclose(solver.objective_value, expected_objective, atol=1e-6)

    def test_miqp_solver_with_cardinality_3(
        self, solver_class: type[qfit.solvers.MIQPSolver]
    ) -> None:
        solver = solver_class(self.target, self.models)
        solver.solve_miqp(cardinality=3)

        expected_weights = np.array([1 / 3, 1 / 3, 1 / 3])
        expected_objective = 0.0
        assert np.allclose(solver.weights, expected_weights, atol=1e-3)
        assert np.isclose(solver.objective_value, expected_objective, atol=1e-6)

    def test_miqp_solver_with_cardinality_2(
        self, solver_class: type[qfit.solvers.MIQPSolver]
    ) -> None:
        solver = solver_class(self.target, self.models)
        solver.solve_miqp(cardinality=2)

        expected_weights = np.array([0.0, 1 / 3, 1 / 3])
        expected_objective = self.expected_objective(expected_weights)
        assert np.allclose(solver.weights, expected_weights, atol=1e-3)
        assert np.isclose(solver.objective_value, expected_objective, atol=1e-6)

    def test_miqp_solver_with_cardinality_1(
        self, solver_class: type[qfit.solvers.MIQPSolver]
    ) -> None:
        solver = solver_class(self.target, self.models)
        solver.solve_miqp(cardinality=1)

        expected_weights = np.array([0.0, 0.0, 1 / 3])
        expected_objective = self.expected_objective(expected_weights)
        assert np.allclose(solver.weights, expected_weights, atol=1e-3)
        assert np.isclose(solver.objective_value, expected_objective, atol=1e-6)

    def test_miqp_solver_with_threshold_and_cardinality_1(
        self, solver_class: type[qfit.solvers.MIQPSolver]
    ) -> None:
        solver = solver_class(self.target, self.models)
        solver.solve_miqp(threshold=0.4, cardinality=1)

        expected_weights = np.array([0.0, 0.0, 0.4])
        expected_objective = self.expected_objective(expected_weights)
        assert np.allclose(solver.weights, expected_weights, atol=1e-3)
        assert np.isclose(solver.objective_value, expected_objective, atol=1e-6)


@pytest.mark.parametrize("solver_class", available_miqp_solvers.values())
class TestMIQPSolverReuse:
    target = np.array([2.0, 3.0, 7.0])
    models = np.array([[6.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 21.0]])

    @pytest.fixture
    def solver(
        self, solver_class: type[qfit.solvers.MIQPSolver]
    ) -> qfit.solvers.MIQPSolver:
        """Instantiate a solver within the scope of this class, to be re-used."""
        return solver_class(self.target, self.models)

    def expected_objective(self, expected_weights: NDArray[np.float_]) -> float:
        return np.sum(np.square(np.inner(self.models, expected_weights) - self.target))

    def test_miqp_solver_with_threshold_and_cardinality_1(
        self, solver_class: type[qfit.solvers.MIQPSolver]
    ) -> None:
        solver = solver_class(self.target, self.models)
        solver.solve_miqp(threshold=0.4, cardinality=1)

        expected_weights = np.array([0.0, 0.0, 0.4])
        expected_objective = self.expected_objective(expected_weights)
        assert np.allclose(solver.weights, expected_weights, atol=1e-3)
        assert np.isclose(solver.objective_value, expected_objective, atol=1e-6)

    def test_miqp_solver_with_cardinality_1(
        self, solver: type[qfit.solvers.MIQPSolver]
    ) -> None:
        solver.solve_miqp(cardinality=1)

        expected_weights = np.array([0.0, 0.0, 1 / 3])
        expected_objective = self.expected_objective(expected_weights)
        assert np.allclose(solver.weights, expected_weights, atol=1e-3)
        assert np.isclose(solver.objective_value, expected_objective, atol=1e-6)

    def test_miqp_solver_with_threshold(
        self, solver: type[qfit.solvers.MIQPSolver]
    ) -> None:
        solver.solve_miqp(threshold=0.4)

        expected_weights = np.array([0.0, 0.4, 0.4])
        expected_objective = self.expected_objective(expected_weights)

        assert np.allclose(solver.weights, expected_weights, atol=1e-3)
        assert np.isclose(solver.objective_value, expected_objective, atol=1e-6)
