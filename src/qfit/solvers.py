from __future__ import annotations

import inspect
import logging
import sys
from abc import ABC, abstractmethod
from types import ModuleType
from typing import TYPE_CHECKING, Any, Optional, cast  # pylint: disable=unused-import

import numpy as np
import scipy as sci  # pylint: disable=unused-import
import scipy.sparse  # pylint: disable=unused-import
from numpy.typing import NDArray

import cvxpy as cp

from .utils.optional_lazy_import import lazy_load_module_if_available

logger = logging.getLogger(__name__)

SolverError: tuple[type[Exception], ...] = RuntimeError

__all__ = [
    "available_qp_solvers",
    "available_miqp_solvers",
    "get_qp_solver_class",
    "get_miqp_solver_class",
]


###############################
# Define solver "interfaces" / Abstractions
#   All solvers in this module must subclass either QPSolver or MIQPSolver, and conform to the interface.
#   The functions in the "Helper Methods" section depend on the subclass membership of these ABCs.
###############################


class GenericSolver(ABC):
    # Class variables
    driver_pkg_name: str
    driver: Optional[ModuleType]

    # Instance variables
    target: NDArray[np.float_]  # 1D array of shape (n_voxels,)
    models: NDArray[np.float_]  # 2D array of shape (n_models, n_voxels,)

    weights: Optional[NDArray[np.float_]] = None
    objective_value: Optional[float] = None

    def __init__(self, target, models, in_model=None, nthreads=1):
        self.target = target
        self.models = models
        self.in_model = in_model

        self.quad_obj = None
        self.lin_obj = None

        self.nconformers = models.shape[0]
        self.valid_indices = []
        self.redundant_indices = []

        self._weights = None
        self._objective_value = 0
        self.weights = None

    def find_redundant_conformers(self, threshold=1e-6):
        """Find redundant conformers using vectorized distance calculations.

        Two conformers are redundant if their model density vectors have
        L2 norm difference less than threshold.

        Optimized using set operations and efficient numpy comparisons.
        """
        redundant_set = set()
        threshold_sq = threshold * threshold

        for i in range(self.nconformers):
            if i in redundant_set:
                continue
            self.valid_indices.append(i)

            # Vectorized comparison with all remaining conformers
            remaining_indices = np.arange(i + 1, self.nconformers)
            if len(remaining_indices) == 0:
                continue

            # Filter out already-known redundant indices
            mask = np.array([j not in redundant_set for j in remaining_indices])
            candidates = remaining_indices[mask]

            if len(candidates) == 0:
                continue

            # Compute squared distances efficiently
            diff = self.models[candidates] - self.models[i]
            dist_sq = np.einsum('ij,ij->i', diff, diff)

            # Find redundant conformers
            redundant_mask = dist_sq < threshold_sq
            new_redundant = candidates[redundant_mask]
            redundant_set.update(new_redundant)

        self.redundant_indices = list(redundant_set)
        assert len(self.valid_indices) + len(self.redundant_indices) == self.nconformers

    def compute_quadratic_coeffs(self):
        # minimize 0.5 x.T P x + q.T x
        #   where P = self.quad_obj =   ρ_model.T ρ_model
        #         q = self.lin_obj  = - ρ_model.T ρ_obs
        # note that ρ_model is the transpose of self.models
        self.find_redundant_conformers()
        self.quad_obj = (
            self.models[self.valid_indices] @ self.models[self.valid_indices].T
        )
        self.lin_obj = -1 * self.models[self.valid_indices] @ self.target

    def construct_weights(self):
        self.weights = []
        j = 0
        for i in range(self.nconformers):
            if i in self.redundant_indices:
                self.weights.append(0)
            else:
                self.weights.append(self._weights[j])
                j += 1
        self.weights = np.array(self.weights)
        self.objective_value = self._objective_value
        assert len(self.weights) == self.nconformers


class QPSolver(GenericSolver):
    """Finds the combination of conformer-occupancies that minimizes difference density.

    Problem statement
    -----------------
    We have observed density ρ^o from the user-provided map (target).
    We also have a set of conformers, each with modelled/calculated density ρ^c_i.
    We want find the vector of occupancies ω = <ω_0, ..., ω_n> that minimizes
        the difference between the observed and modelled density --- that minimizes
        a residual sum-of-squares function, rss(ω).
    Mathematically, we wish to minimize:
        min_ω rss(ω) = min_ω || ρ^c ω - ρ^o ||^2

    Expanding & rearranging rss(ω):
        rss(ω) = ( ρ^c ω - ρ^o ).T  ( ρ^c ω - ρ^o )
                = ω.T ρ^c.T ρ^c ω - 2 ρ^o.T ρ^c ω + ρ^o.T ρ^o
    We can rewrite this as
        rss(ω) = ω.T P ω + 2 q.T ω + C
    where
        P =  ρ^c.T ρ^c
        q = -ρ^c.T ρ^o
        C =  ρ^o.T ρ^o

    Noting that the canonical QP objective function is
        g(x) = 1/2 x.T P x + q.T x
    we can use a QP solver to find min_x g(x), which, by equivalence,
        will provide the solution to min_ω rss(ω).

    Solution constraints
    --------------------
    Furthermore, these occupancies are meaningful parameters, so we require
    that their sum is within the unit interval:
        Σ ω_i ≤ 1
    and that each individual occupancy is a positive fractional number:
        0 ≤ ω_i ≤ 1
    """

    @abstractmethod
    def solve_qp(self) -> None: ...


class MIQPSolver(GenericSolver):
    """Finds the combination of conformer-occupancies that minimizes difference density.

    Problem statement
    -----------------
    We have observed density ρ^o from the user-provided map (target).
    We also have a set of conformers, each with modelled/calculated density ρ^c_i.
    We want find the vector of occupancies ω = <ω_0, ..., ω_n> that minimizes
        the difference between the observed and modelled density --- that minimizes
        a residual sum-of-squares function, rss(ω).
    Mathematically, we wish to minimize:
        min_ω rss(ω) = min_ω || ρ^c ω - ρ^o ||^2

    Expanding & rearranging rss(ω):
        rss(ω) = ( ρ^c ω - ρ^o ).T  ( ρ^c ω - ρ^o )
                = ω.T ρ^c.T ρ^c ω - 2 ρ^o.T ρ^c ω + ρ^o.T ρ^o
    We can rewrite this as
        rss(ω) = ω.T P ω + 2 q.T ω + C
    where
        P =  ρ^c.T ρ^c
        q = -ρ^c.T ρ^o
        C =  ρ^o.T ρ^o

    Noting that the canonical QP objective function is
        g(x) = 1/2 x.T P x + q.T x
    we can use a QP solver to find min_x g(x), which, by equivalence,
        will provide the solution to min_ω rss(ω).

    Solution constraints
    --------------------
    Furthermore, these occupancies are meaningful parameters, so we require
    that their sum is within the unit interval:
        Σ ω_i ≤ 1

    We also want to have either:
        (a) a set of conformers of known size (cardinality), or
        (b) a set of conformers with _at least_ threshold occupancy, or else zero (threshold).
    This can be achieved with a mixed-integer linear constraint:
        z_i t_min ≤ ω_i ≤ z_i
    where
        z_i ∈ {0, 1}
        t_min is the minimum-allowable threshold value for ω.
    """

    @abstractmethod
    def solve_miqp(
        self,
        threshold: Optional[float] = None,
        cardinality: Optional[int] = None,
        exact: bool = False,
    ) -> None: ...


###############################
# Define solver implementations
###############################


class CVXPYSolver(QPSolver, MIQPSolver):
    driver_pkg_name = "cvxpy"
    driver = lazy_load_module_if_available(driver_pkg_name)

    def solve_miqp(self, threshold=0, cardinality=0):
        if self.quad_obj is None or self.lin_obj is None:
            self.compute_quadratic_coeffs()

        m = len(self.valid_indices)
        P = self.quad_obj
        q = self.lin_obj

        w = cp.Variable(m)
        z = cp.Variable(m, boolean=True)
        objective = cp.Minimize(0.5 * cp.quad_form(w, cp.psd_wrap(P)) + q.T @ w)
        constraints = [np.ones(m).T @ w <= 1]
        if threshold:
            constraints += [w - z <= 0, w >= threshold * z]
            # The first constraint requires w_i to be zero if z_i is
            # The second requires that each non-zero w_i is greater than the threshold
        if cardinality:
            constraints += [w - z <= 0, np.ones(m).T @ z <= cardinality]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver="SCIP")
        self.objective_value = prob.value
        # I'm not sure why objective_values is calculated this way, but doing
        # so to be compatible with the former CPLEXSolver class
        self._objective_value = 2 * prob.value + self.target.T @ self.target
        self._weights = w.value
        self.construct_weights()

    def rscc_solve_miqp(self, threshold=0, cardinality=0):
        if self.quad_obj is None or self.lin_obj is None:
            self.compute_quadratic_coeffs()

        m = len(self.valid_indices)
        P = self.quad_obj
        q = self.lin_obj

        w = cp.Variable(m)
        z = cp.Variable(m, boolean=True)
        objective = cp.Minimize(0.5 * cp.quad_form(w, cp.psd_wrap(P)) + q.T @ w)
        constraints = [np.ones(m).T @ w <= 1]
        if threshold:
            constraints += [w - z <= 0, w >= threshold * z]
            # The first constraint requires w_i to be zero if z_i is
            # The second requires that each non-zero w_i is greater than the threshold
        if cardinality:
            constraints += [w - z <= 0, np.ones(m).T @ z <= cardinality]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver="SCIP")
        self.objective_value = prob.value
        # I'm not sure why objective_values is calculated this way, but doing
        # so to be compatible with the former CPLEXSolver class
        self._objective_value = 2 * prob.value + self.target.T @ self.target
        self._weights = w.value

        # output the correlation coefficient between the QFIT MODEL density and the target density 
        cutoff=0.002
        filterarray = self._weights >= cutoff
        filtered_weights = self._weights[filterarray]
        filtered_models = self.models[filterarray, :]

        combined_model = np.dot(filtered_weights, filtered_models)
        corr = np.corrcoef(combined_model, self.target)[0, 1]
        print(f"RSCC for model of interest: {corr}")

        # output correlations coefficient between INPUT MODEL density and target density 
        if self.in_model is not None and self.in_model.size > 0:
            input_corr = np.corrcoef(self.in_model, self.target)[0, 1]
            print(f"RSCC for comparison model: {input_corr}")
            
        self.construct_weights()



    def solve_qp(self, split_threshold=3000):
        if self.quad_obj is None or self.lin_obj is None:
            self.compute_quadratic_coeffs()

        valid_conformers = len(self.valid_indices)
        self._weights = np.zeros(valid_conformers)
        splits = valid_conformers // split_threshold + 1  # number of splits
        for split in range(splits):
            # take every splits-th element with split as an offset, guaranteeing full coverage
            P = self.quad_obj[split::splits, split::splits]
            q = self.lin_obj[split::splits]
            m = len(P)
            w = cp.Variable(m)
            objective = cp.Minimize(0.5 * cp.quad_form(w, cp.psd_wrap(P)) + q.T @ w)
            constraints = [w >= np.zeros(m), np.ones(m).T @ w <= 1]
            prob = cp.Problem(objective, constraints)
            prob.solve()
            # I'm not sure why objective_values is calculated this way, but doing
            # so to be compatible with the former CPLEXSolver class
            self._objective_value += 2 * prob.value + self.target.T @ self.target
            self._objective_value /= splits
            self._weights[split::splits] = w.value / splits
        self.construct_weights()


class TorchSolver(QPSolver, MIQPSolver):
    """QP/MIQP solver using PyTorch with differentiable optimization.

    Uses softmax reparameterization for QP and Gumbel-Sigmoid annealing
    for MIQP to provide end-to-end differentiability.
    """

    driver_pkg_name = "torch"
    driver = lazy_load_module_if_available(driver_pkg_name)

    def solve_qp(self, split_threshold=3000):
        if self.quad_obj is None or self.lin_obj is None:
            self.compute_quadratic_coeffs()

        torch = self.driver

        valid_conformers = len(self.valid_indices)
        self._weights = np.zeros(valid_conformers)
        self._objective_value = 0
        splits = valid_conformers // split_threshold + 1  # number of splits
        for split in range(splits):
            # take every splits-th element with split as an offset
            P_np = self.quad_obj[split::splits, split::splits]
            q_np = self.lin_obj[split::splits]
            m = len(P_np)

            P = torch.tensor(P_np, dtype=torch.float64)
            q = torch.tensor(q_np, dtype=torch.float64)

            # Unconstrained logits with slack dimension (m+1).
            # softmax(alpha)[:m] guarantees omega_i >= 0 and sum(omega) <= 1.
            alpha = torch.zeros(m + 1, dtype=torch.float64, requires_grad=True)

            optimizer = torch.optim.LBFGS(
                [alpha],
                lr=1.0,
                max_iter=20,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-12,
            )

            prev_obj = float("inf")
            for _ in range(100):

                def closure():
                    optimizer.zero_grad()
                    omega = torch.softmax(alpha, dim=0)[:m]
                    obj = 0.5 * omega @ P @ omega + q @ omega
                    obj.backward()
                    return obj

                obj_val = optimizer.step(closure)
                current_obj = obj_val.item()
                if abs(current_obj - prev_obj) < 1e-14:
                    break
                prev_obj = current_obj

            with torch.no_grad():
                omega = torch.softmax(alpha, dim=0)[:m]
                qp_value = (0.5 * omega @ P @ omega + q @ omega).item()

            # Accumulate objective using same convention as CVXPYSolver
            self._objective_value += 2 * qp_value + self.target.T @ self.target
            self._objective_value /= splits
            self._weights[split::splits] = omega.detach().numpy() / splits
        self.construct_weights()

    def _select_conformers_annealing(self, P_np, q_np, m, cardinality=0, threshold=0):
        """Phase 1: Binary Concrete relaxation + temperature annealing.

        Uses sigmoid(logit / tau) for relaxed binary selection and
        softmax with slack for continuous weights. Temperature is
        annealed from 2.0 to 0.01 over 30 log-spaced steps with
        200 Adam steps per temperature.
        """
        torch = self.driver

        P = torch.tensor(P_np, dtype=torch.float64)
        q = torch.tensor(q_np, dtype=torch.float64)

        # Learnable parameters
        log_alpha_z = torch.zeros(m, dtype=torch.float64, requires_grad=True)
        alpha_w = torch.zeros(m + 1, dtype=torch.float64, requires_grad=True)

        # Temperature schedule: log-spaced from 2.0 to 0.01
        temperatures = torch.logspace(
            float(np.log10(2.0)),
            float(np.log10(0.01)),
            steps=30,
            dtype=torch.float64,
        )

        for t_idx, tau in enumerate(temperatures):
            progress = t_idx / max(len(temperatures) - 1, 1)
            lambda_penalty = 1.0 + 99.0 * progress  # ramp from 1 to 100

            optimizer = torch.optim.Adam([log_alpha_z, alpha_w], lr=0.01)

            for _ in range(200):
                optimizer.zero_grad()

                # Relaxed binary selection (deterministic Binary Concrete)
                z = torch.sigmoid(log_alpha_z / tau)

                # Continuous weights via softmax with slack
                gamma = torch.softmax(alpha_w, dim=0)[:m]

                # Gated occupancies
                omega = z * gamma

                # QP objective
                loss = 0.5 * omega @ P @ omega + q @ omega

                # Cardinality penalty: penalize having more than K conformers
                if cardinality:
                    loss = (
                        loss
                        + lambda_penalty
                        * torch.relu(z.sum() - cardinality) ** 2
                    )

                # Threshold penalty: penalize omega_i < t_min * z_i
                if threshold:
                    loss = loss + lambda_penalty * torch.relu(
                        threshold * z - omega
                    ).sum()

                loss.backward()
                optimizer.step()

        # Binarize selections
        with torch.no_grad():
            z_hard = (torch.sigmoid(log_alpha_z / 0.01) > 0.5).numpy()

        # Enforce cardinality: if more than K selected, keep top K by activation
        if cardinality and np.sum(z_hard) > cardinality:
            with torch.no_grad():
                activations = torch.sigmoid(log_alpha_z / 0.01).numpy()
            top_k = np.argsort(activations)[-cardinality:]
            z_hard = np.zeros(m, dtype=bool)
            z_hard[top_k] = True

        return z_hard

    def _solve_qp_subset(self, P_np, q_np, selected, threshold=0):
        """Phase 2: Clean QP (softmax + L-BFGS) over selected conformers.

        When threshold > 0, uses shifted reparameterization:
            w_i = t_min + gamma_i * (1 - K * t_min)
        to guarantee w_i >= t_min for all selected conformers.
        """
        torch = self.driver

        sel_indices = np.where(selected)[0]
        k = len(sel_indices)

        if k == 0:
            return np.zeros(len(P_np)), 0.0

        P_sub = P_np[np.ix_(sel_indices, sel_indices)]
        q_sub = q_np[sel_indices]

        P = torch.tensor(P_sub, dtype=torch.float64)
        q = torch.tensor(q_sub, dtype=torch.float64)

        if threshold > 0 and k * threshold <= 1.0:
            # Shifted reparameterization: w_i = t_min + gamma_i * slack_budget
            # ensures w_i >= t_min and sum(w) <= 1
            slack_budget = 1.0 - k * threshold
            alpha = torch.zeros(k + 1, dtype=torch.float64, requires_grad=True)

            optimizer = torch.optim.LBFGS(
                [alpha],
                lr=1.0,
                max_iter=20,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-12,
            )

            prev_obj = float("inf")
            for _ in range(100):

                def closure():
                    optimizer.zero_grad()
                    gamma = torch.softmax(alpha, dim=0)[:k]
                    w = threshold + gamma * slack_budget
                    obj = 0.5 * w @ P @ w + q @ w
                    obj.backward()
                    return obj

                obj_val = optimizer.step(closure)
                current_obj = obj_val.item()
                if abs(current_obj - prev_obj) < 1e-14:
                    break
                prev_obj = current_obj

            with torch.no_grad():
                gamma = torch.softmax(alpha, dim=0)[:k]
                w_sub = (threshold + gamma * slack_budget).numpy()
        else:
            # Standard softmax QP (no threshold or infeasible threshold)
            alpha = torch.zeros(k + 1, dtype=torch.float64, requires_grad=True)

            optimizer = torch.optim.LBFGS(
                [alpha],
                lr=1.0,
                max_iter=20,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-12,
            )

            prev_obj = float("inf")
            for _ in range(100):

                def closure():
                    optimizer.zero_grad()
                    w = torch.softmax(alpha, dim=0)[:k]
                    obj = 0.5 * w @ P @ w + q @ w
                    obj.backward()
                    return obj

                obj_val = optimizer.step(closure)
                current_obj = obj_val.item()
                if abs(current_obj - prev_obj) < 1e-14:
                    break
                prev_obj = current_obj

            with torch.no_grad():
                w_sub = torch.softmax(alpha, dim=0)[:k].numpy()

        # Enforce threshold: zero out any w_i < t_min
        if threshold > 0:
            w_sub[w_sub < threshold] = 0.0

        # Build full weight vector over all valid conformers
        w_full = np.zeros(len(P_np))
        w_full[sel_indices] = w_sub

        # Compute QP objective value
        qp_value = 0.5 * w_full @ P_np @ w_full + q_np @ w_full
        return w_full, qp_value

    def solve_miqp(self, threshold=0, cardinality=0):
        if self.quad_obj is None or self.lin_obj is None:
            self.compute_quadratic_coeffs()

        m = len(self.valid_indices)
        P = self.quad_obj
        q = self.lin_obj

        # Determine if conformer selection is needed
        need_selection = (threshold > 0) or (0 < cardinality < m)
        if need_selection:
            # Phase 1: Select conformers via annealing
            selected = self._select_conformers_annealing(
                P, q, m, cardinality, threshold
            )
        else:
            selected = np.ones(m, dtype=bool)

        # Phase 2: Clean QP over selected conformers
        self._weights, qp_value = self._solve_qp_subset(P, q, selected, threshold)

        # Compute objective value using same convention as CVXPYSolver
        self._objective_value = 2 * qp_value + self.target.T @ self.target
        self.construct_weights()

    def rscc_solve_miqp(self, threshold=0, cardinality=0):
        if self.quad_obj is None or self.lin_obj is None:
            self.compute_quadratic_coeffs()

        m = len(self.valid_indices)
        P = self.quad_obj
        q = self.lin_obj

        need_selection = (threshold > 0) or (0 < cardinality < m)
        if need_selection:
            selected = self._select_conformers_annealing(
                P, q, m, cardinality, threshold
            )
        else:
            selected = np.ones(m, dtype=bool)

        self._weights, qp_value = self._solve_qp_subset(P, q, selected, threshold)
        self._objective_value = 2 * qp_value + self.target.T @ self.target

        # RSCC reporting (identical to CVXPYSolver logic)
        cutoff = 0.002
        filterarray = self._weights >= cutoff
        filtered_weights = self._weights[filterarray]
        filtered_models = self.models[self.valid_indices][filterarray, :]

        combined_model = np.dot(filtered_weights, filtered_models)
        corr = np.corrcoef(combined_model, self.target)[0, 1]
        print(f"RSCC for model of interest: {corr}")

        if self.in_model is not None and self.in_model.size > 0:
            input_corr = np.corrcoef(self.in_model, self.target)[0, 1]
            print(f"RSCC for comparison model: {input_corr}")

        self.construct_weights()


###############################
# Helper methods
###############################


def _available_qp_solvers() -> dict[str, type]:
    """List all available QP solver classes in this module."""
    available_solvers = {}

    # Get all classes defined in this module
    #   use module.__dict__ because it preserves order
    #     (unlike dir(module) or inspect.getmembers(module))
    for name, obj in sys.modules[__name__].__dict__.items():
        if inspect.isclass(obj) and obj.__module__ == __name__:
            # Check the class implements QPSolver
            if obj in QPSolver.__subclasses__():
                # Check the driver module is loadable
                if obj.driver is not None:
                    available_solvers[name] = obj
    return available_solvers


def _available_miqp_solvers() -> dict[str, type]:
    """List all available MIQP solver classes in this module."""
    available_solvers = {}

    # Get all classes defined in this module
    #   use module.__dict__ because it preserves order
    #     (unlike dir(module) or inspect.getmembers(module))
    for name, obj in sys.modules[__name__].__dict__.items():
        if inspect.isclass(obj) and obj.__module__ == __name__:
            # Check the class implements MIQPSolver
            if obj in MIQPSolver.__subclasses__():
                # Check the driver module is loadable
                if obj.driver is not None:
                    available_solvers[name] = obj
    return available_solvers


available_qp_solvers = _available_qp_solvers()
available_miqp_solvers = _available_miqp_solvers()
if not available_qp_solvers:
    msg = (
        "Could not find any QP solver engines.\n"
        + "Please ensure that at least one of:\n  "
        + str([solver.driver_pkg_name for solver in QPSolver.__subclasses__()])
        + "\n"
        + "is installed."
    )
    raise ImportError(msg)
if not available_miqp_solvers:
    msg = (
        "Could not find any MIQP solver engines.\n"
        + "Please ensure that at least one of:\n  "
        + str([solver.driver_pkg_name for solver in MIQPSolver.__subclasses__()])
        + "\n"
        + "is installed."
    )
    raise ImportError(msg)


def get_qp_solver_class(solver_type: str) -> type[QPSolver]:
    """Return the class of the requested solver type, or raise a KeyError."""
    return available_qp_solvers[solver_type]


def get_miqp_solver_class(solver_type: str) -> type[MIQPSolver]:
    """Return the class of the requested solver type, or raise a KeyError."""
    return available_miqp_solvers[solver_type]
