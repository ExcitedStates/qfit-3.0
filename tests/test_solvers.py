import inspect

import numpy as np
import pytest
from numpy.typing import NDArray

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
        assert np.isclose(solver.objective_value, expected_objective, atol=1e-4)

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


def _reference_find_redundant(models, threshold=1e-6):
    """Previous CVXPYSolver.find_redundant_conformers (pairwise loop)."""
    n = len(models)
    redundant, valid = set(), []
    for i in range(n):
        if i in redundant:
            continue
        valid.append(i)
        for j in range(i + 1, n):
            if j in redundant:
                continue
            diff = models[j] - models[i]
            if diff @ diff < threshold * threshold:
                redundant.add(j)
    return valid, sorted(redundant)


def _models_with_duplicates(n=60, nvoxels=500, seed=0):
    rng = np.random.default_rng(seed)
    models = rng.random((n, nvoxels)) * 10
    models[7] = models[3]                    # exact duplicate
    models[20] = models[3]                   # duplicate of a duplicate
    models[11] = models[5] + 1e-8            # inside the 1e-6 threshold
    models[12] = models[5] + 1e-4            # outside the threshold
    models[40] = models[30] + 2e-7 / np.sqrt(nvoxels)  # just inside
    models[41] = models[30] + 2e-6 / np.sqrt(nvoxels)  # just outside
    return models


@pytest.mark.parametrize("solver_class", available_miqp_solvers.values())
class TestRedundantConformers:

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_reference(self, solver_class, seed) -> None:
        models = _models_with_duplicates(seed=seed)
        solver = solver_class(np.zeros(models.shape[1]), models)
        solver.find_redundant_conformers()
        valid, redundant = _reference_find_redundant(models)
        assert solver.valid_indices == valid
        assert sorted(solver.redundant_indices) == redundant
        assert sorted(redundant) == [7, 11, 20, 40]

    def test_quadratic_coeffs_match_reference(self, solver_class) -> None:
        models = _models_with_duplicates()
        target = np.random.default_rng(3).random(models.shape[1])
        solver = solver_class(target, models)
        solver.compute_quadratic_coeffs()
        valid, _ = _reference_find_redundant(models)
        assert np.allclose(solver.quad_obj, models[valid] @ models[valid].T)
        assert np.allclose(solver.lin_obj, -models[valid] @ target)

    def test_weights_of_redundant_conformers_are_zero(self, solver_class) -> None:
        target = np.array([2.0, 3.0, 7.0])
        models = np.array([[6.0, 0.0, 0.0], [6.0, 0.0, 0.0],
                           [0.0, 9.0, 0.0], [0.0, 0.0, 21.0]])
        solver = solver_class(target, models)
        solver.solve_qp()
        assert solver.weights.shape == (4,)
        assert solver.weights[1] == 0
        assert np.allclose(solver.weights[[0, 2, 3]], 1 / 3, atol=1e-3)
