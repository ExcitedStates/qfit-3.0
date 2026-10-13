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

    def _gram_matrix(self):
        """Pairwise inner products of the model densities, ρ_model.T ρ_model."""
        if getattr(self, "_gram", None) is None:
            self._gram = self.models @ self.models.T
        return self._gram

    def find_redundant_conformers(self, threshold=1e-6):
        """Find redundant conformers.

        Conformer j is redundant if an earlier, non-redundant conformer i < j
        has a model density vector within L2 distance `threshold` of it.

        Squared distances are obtained from the Gram matrix,
            ||m_i - m_j||^2 = G_ii + G_jj - 2 G_ij,
        which costs O(n^2) once G is known (G is needed for the QP anyway).
        Because that expression suffers from cancellation, it is only used as
        a generous prefilter; every candidate pair is confirmed exactly.
        """
        threshold_sq = threshold * threshold
        n = self.nconformers
        self.valid_indices = []
        self.redundant_indices = []
        if n == 0:
            return

        gram = self._gram_matrix()
        sq_norms = np.diag(gram).copy()
        # Candidate pairs (i < j), computed in row blocks to bound memory.
        candidates = [None] * n
        block = 1024
        for start in range(0, n, block):
            rows = slice(start, min(start + block, n))
            norm_sum = sq_norms[rows, None] + sq_norms[None, :]
            dist_sq = norm_sum - 2 * gram[rows]
            close = dist_sq <= threshold_sq + 1e-8 * norm_sum
            for offset, row in enumerate(close):
                i = start + offset
                candidates[i] = np.flatnonzero(row[i + 1:]) + i + 1

        redundant = np.zeros(n, dtype=bool)
        for i in range(n):
            if redundant[i]:
                continue
            self.valid_indices.append(i)
            for j in candidates[i]:
                if redundant[j]:
                    continue
                diff = self.models[j] - self.models[i]
                if diff @ diff < threshold_sq:
                    redundant[j] = True

        self.redundant_indices = np.flatnonzero(redundant).tolist()
        assert len(self.valid_indices) + len(self.redundant_indices) == self.nconformers

    def compute_quadratic_coeffs(self):
        # minimize 0.5 x.T P x + q.T x
        #   where P = self.quad_obj =   ρ_model.T ρ_model
        #         q = self.lin_obj  = - ρ_model.T ρ_obs
        # note that ρ_model is the transpose of self.models
        self.find_redundant_conformers()
        valid = np.asarray(self.valid_indices, dtype=int)
        self.quad_obj = self._gram_matrix()[np.ix_(valid, valid)]
        self.lin_obj = -1 * self.models[valid] @ self.target

    def construct_weights(self):
        self.weights = np.zeros(self.nconformers)
        self.weights[self.valid_indices] = self._weights
        self.objective_value = self._objective_value
        assert len(self.weights) == self.nconformers

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
