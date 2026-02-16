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

_torch_available = "TorchSolver" in available_qp_solvers
_skip_no_torch = pytest.mark.skipif(
    not _torch_available, reason="torch not installed"
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


@_skip_no_torch
class TestTorchSolverGradientFlow:
    """Verify gradients propagate through the TorchSolver optimization."""

    def test_gradient_flow_through_softmax_qp(self) -> None:
        import torch

        target = torch.tensor([2.0, 3.0, 7.0], dtype=torch.float64)
        models = torch.tensor(
            [[6.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 21.0]],
            dtype=torch.float64,
        )

        alpha = torch.zeros(4, dtype=torch.float64, requires_grad=True)  # 3 + 1 slack
        omega = torch.softmax(alpha, dim=0)[:3]

        P = models @ models.T
        q = -models @ target
        obj = 0.5 * omega @ P @ omega + q @ omega
        obj.backward()

        assert alpha.grad is not None
        assert not torch.all(alpha.grad == 0)


@_skip_no_torch
class TestTorchSolverLargeProblem:
    """Verify TorchSolver convergence on a larger problem."""

    def test_large_qp_convergence(self) -> None:
        from qfit.solvers import get_qp_solver_class

        TorchSolver = get_qp_solver_class("TorchSolver")

        rng = np.random.default_rng(42)
        n_voxels = 500
        n_conformers = 50

        models = rng.standard_normal((n_conformers, n_voxels))
        # Create a target from a known mixture
        true_weights = np.zeros(n_conformers)
        true_weights[:3] = [0.3, 0.3, 0.3]
        target = models.T @ true_weights + 0.01 * rng.standard_normal(n_voxels)

        solver = TorchSolver(target, models)
        solver.solve_qp()

        assert solver.weights is not None
        assert np.all(solver.weights >= -1e-6)
        assert np.sum(solver.weights) <= 1.0 + 1e-6


@_skip_no_torch
class TestTorchSolverSparsity:
    """Verify MIQP cardinality constraint produces the expected sparsity."""

    def test_sparsity_with_cardinality_2(self) -> None:
        from qfit.solvers import get_miqp_solver_class

        TorchSolver = get_miqp_solver_class("TorchSolver")

        rng = np.random.default_rng(42)
        n_voxels = 100
        n_conformers = 10

        models = rng.standard_normal((n_conformers, n_voxels))
        # Create a target from exactly 2 conformers
        true_weights = np.zeros(n_conformers)
        true_weights[0] = 0.5
        true_weights[1] = 0.4
        target = models.T @ true_weights

        solver = TorchSolver(target, models)
        solver.solve_miqp(cardinality=2)

        assert solver.weights is not None
        assert np.sum(solver.weights > 1e-6) <= 2
