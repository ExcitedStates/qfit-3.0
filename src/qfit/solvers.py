from __future__ import annotations

import inspect
import logging
import sys
from abc import ABC, abstractmethod
from types import ModuleType
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np
import scipy as sci
import scipy.sparse
from numpy.typing import NDArray

from .utils.optional_lazy_import import lazy_load_module_if_available

logger = logging.getLogger(__name__)

SolverError: tuple[type[Exception], ...] = (RuntimeError,)

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
    def solve_qp(self) -> None:
        ...


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
    ) -> None:
        ...


###############################
# Define solver implementations
###############################


class CVXOPTSolver(QPSolver):
    driver_pkg_name = "cvxopt"
    driver = lazy_load_module_if_available(driver_pkg_name)
    if TYPE_CHECKING:
        import cvxopt

    def __init__(self, target: NDArray[np.float_], models: NDArray[np.float_]) -> None:
        if TYPE_CHECKING:
            self.driver = self.cvxopt
            assert self.driver is not None

        # Initialize variables
        self.target = target
        self.models = models

        # self.quad_obj: self.driver.matrix
        # self.lin_obj: self.driver.matrix
        # self.le_constraints: self.driver.spmatrix
        # self.le_bounds: self.driver.matrix

        self.weights: Optional[NDArray[np.float_]] = None
        self.objective_value: Optional[float] = None

        self.solution: dict[str, Any]
        self.nconformers = models.shape[0]

        # Get the driver & set options
        self.driver.solvers.options["show_progress"] = False
        self.driver.solvers.options["abstol"] = 1e-8
        self.driver.solvers.options["reltol"] = 1e-7
        self.driver.solvers.options["feastol"] = 1e-8

    def solve_qp(self) -> None:
        if TYPE_CHECKING:
            self.driver = self.cvxopt
            assert self.driver is not None

        # Set up the matrices and restraints
        logger.debug(
            "Building cvxopt matrix, size: (%i,%i)",
            self.nconformers,
            self.nconformers,
        )

        # minimize 0.5 x.T P x + q.T x
        #   where P = self.quad_obj        =   ρ_model.T ρ_model
        #         q = self.lin_obj         = - ρ_model.T ρ_obs
        self.quad_obj = self.driver.matrix(np.inner(self.models, self.models), tc="d")
        self.lin_obj = self.driver.matrix(-np.inner(self.models, self.target), tc="d")

        # subject to:
        #   G x ≤ h
        #   where G = self.le_constraints
        #         h = self.le_bounds
        #
        #   Each weight x falls in the closed interval [0..1], and the sum of all weights is <= 1.
        #   This corresponds to (2 * nconformers + 1) constraints, imposed on (nconformers) variables:
        #     the lower bound accounts for (nconformers) constraints,
        #     the upper bound accounts for (nconformers) constraints,
        #     the summation constraint accounts for 1.
        #   We construct G with a sparse matrix.
        #         constraint idx=[0..N)           constraint idx=[N..2N)                             constraint idx=2N
        rowidxs = [*range(0, self.nconformers)] + [*range(self.nconformers, 2 * self.nconformers)] + (self.nconformers * [2 * self.nconformers])  # fmt:skip
        #         -x_i ≤ 0                        x_i ≤ 1                                            Σ_i x_i ≤ 1
        coeffs  = (self.nconformers * [-1.0])   + (self.nconformers * [1.0])                       + (self.nconformers * [1.0])  # fmt:skip
        colidxs = [*range(0, self.nconformers)] + [*range(0, self.nconformers)]                    + [*range(0, self.nconformers)]  # fmt:skip
        threshs = (self.nconformers * [0.0])    + (self.nconformers * [1.0])                       + [1.0]  # fmt:skip
        self.le_constraints = self.driver.spmatrix(coeffs, rowidxs, colidxs, tc="d")
        self.le_bounds = self.driver.matrix(threshs, tc="d")

        # Solve
        self.solution = self.driver.solvers.qp(
            self.quad_obj, self.lin_obj, self.le_constraints, self.le_bounds
        )

        # Store the density residual and the weights
        self.objective_value = (
            2 * self.solution["primal objective"] +
            np.inner(self.target, self.target)
        )  # fmt:skip
        self.weights = np.asarray(self.solution["x"]).ravel()


class CPLEXSolver(QPSolver, MIQPSolver):
    driver_pkg_name = "cplex"
    driver = lazy_load_module_if_available(driver_pkg_name)
    if TYPE_CHECKING:
        import cplex

    def __init__(
        self, target: NDArray[np.float_], models: NDArray[np.float_], nthreads: int = 1
    ) -> None:
        if TYPE_CHECKING:
            self.driver = self.cplex
            assert self.driver is not None

        # Initialize variables
        self.target = target
        self.models = models

        self.quad_obj: list[object] = []
        self.lin_obj: list[tuple[int, float]] = []

        self.weights: Optional[NDArray[np.float_]] = None
        self.objective_value: Optional[float] = None

        self.nconformers = models.shape[0]
        self.nthreads = nthreads

        # Get the driver & append raisable Exceptions to SolverError class in module (global) scope
        CplexSolverError: type[Exception] = self.driver.exceptions.CplexSolverError
        global SolverError
        SolverError += (CplexSolverError,)

    def compute_quadratic_coeffs(self) -> None:
        """Precompute the quadratic coefficients (P, q).

        These values don't depend on threshold/cardinality.
        Having these objectives pre-computed saves a few cycles when MIQP
        is evaluated for multiple values of threshold/cardinality.
        """
        if TYPE_CHECKING:
            self.driver = self.cplex
            assert self.driver is not None

        # minimize 0.5 x.T P x + q.T x
        #   where P = self.quad_obj =   ρ_model.T ρ_model
        #         q = self.lin_obj  = - ρ_model.T ρ_obs
        quad_obj_coeffs = np.inner(self.models, self.models)
        lin_obj_coeffs = -np.inner(self.models, self.target)

        # We have to unpack the arrays for CPLEX into CSR format (even though they're dense)
        self.quad_obj = []
        for row in quad_obj_coeffs:
            idxs, vals = zip(*enumerate(row.tolist()))
            self.quad_obj.append(self.driver.SparsePair(ind=idxs, val=vals))

        # CPLEX requires linear objectives as a list of (idx, val) tuples
        self.lin_obj = list(enumerate(lin_obj_coeffs))

    def solve_miqp(
        self,
        threshold: Optional[float] = None,
        cardinality: Optional[int] = None,
        exact: bool = False,
    ) -> None:
        if TYPE_CHECKING:
            self.driver = self.cplex
            assert self.driver is not None

        if not (self.quad_obj and self.lin_obj):
            self.compute_quadratic_coeffs()

        # Create and configure the cplex object
        miqp = self.driver.Cplex()
        miqp.set_results_stream(None)
        miqp.set_log_stream(None)
        miqp.set_warning_stream(None)
        miqp.set_error_stream(None)
        if self.nthreads is not None:
            miqp.parameters.threads.set(self.nthreads)

        # Create variables and set linear constraints
        # w_i ≤ 1
        variable_names = [f"w{n}" for n in range(self.nconformers)]
        upper_bounds = self.nconformers * [1.0]
        miqp.variables.add(names=variable_names, ub=upper_bounds)

        # Σ_i w_i ≤ 1
        ind = [f"w{n}" for n in range(self.nconformers)]
        val = self.nconformers * [1.0]
        miqp.linear_constraints.add(
            lin_expr=[self.driver.SparsePair(ind=ind, val=val)],
            senses=["L"],
            rhs=[1.0],
        )

        # Setup quadratic objective of the MIQP
        miqp.objective.set_quadratic(self.quad_obj)
        miqp.objective.set_linear(self.lin_obj)
        miqp.objective.set_sense(miqp.objective.sense.minimize)

        # If cardinality or threshold is specified, the problem is a MIQP,
        # so we need to add binary integer variables z_i.
        if cardinality not in (None, 0) or threshold not in (None, 0):
            # z_i ∈ {0, 1}
            integer_names = [f"z{n}" for n in range(self.nconformers)]
            variable_types = self.nconformers * miqp.variables.type.binary
            miqp.variables.add(names=integer_names, types=variable_types)

            # Only count weights for which z_i is 1
            # w_i - z_i ≤ 0
            # (∵ z_i ∈ {0,1} and 0 ≤ w_i ≤ 1, this is only true when z_i = 1)
            lin_expr = [
                self.driver.SparsePair(ind=[f"w{n}", f"z{n}"], val=[1, -1])
                for n in range(self.nconformers)
            ]
            senses = self.nconformers * ["L"]
            rhs = self.nconformers * [0.0]
            miqp.linear_constraints.add(lin_expr=lin_expr, senses=senses, rhs=rhs)

        # Set the threshold constraint if applicable
        if threshold not in (None, 0):
            # tdmin z_i - w_i ≤ 0, i.e. w_i ≥ tdmin
            lin_expr = [
                self.driver.SparsePair(ind=[f"z{n}", f"w{n}"], val=[threshold, -1])
                for n in range(self.nconformers)
            ]
            senses = self.nconformers * ["L"]
            rhs = self.nconformers * [0.0]
            miqp.linear_constraints.add(lin_expr=lin_expr, senses=senses, rhs=rhs)

        # Set the cardinality constraint if applicable
        if cardinality not in (None, 0):
            # Σ z_i ≤ cardinality
            cardinality = cast(int, cardinality)  # typing noop
            ind = [f"z{n}" for n in range(self.nconformers)]
            val = self.nconformers * [1.0]
            lin_expr = [self.driver.SparsePair(ind=ind, val=val)]
            if exact:
                senses = ["E"]
                rhs = [min(cardinality, self.nconformers)]
            else:
                senses = ["L"]
                rhs = [cardinality]
            miqp.linear_constraints.add(lin_expr=lin_expr, senses=senses, rhs=rhs)

        # Solve
        miqp.solve()

        # Store the density residual and the weights
        self.objective_value = (
            2 * miqp.solution.get_objective_value()
            + np.inner(self.target, self.target)
        )  # fmt:skip
        self.weights = np.array(miqp.solution.get_values()[: self.nconformers])

        # Close the cplex object
        miqp.end()

    def solve_qp(self) -> None:
        if TYPE_CHECKING:
            self.driver = self.cplex
            assert self.driver is not None

        # We can re-use the MIQP code above, provided no cardinality or threshold.
        self.solve_miqp(threshold=None, cardinality=None, exact=False)


class OSQPSolver(QPSolver):
    driver_pkg_name = "osqp"
    driver = lazy_load_module_if_available(driver_pkg_name)
    if TYPE_CHECKING:
        import osqp

    OSQP_SETTINGS = {
        "eps_abs": 1e-06,
        "eps_rel": 1e-06,
        "eps_prim_inf": 1e-07,
        "verbose": False,
    }

    def __init__(self, target: NDArray[np.float_], models: NDArray[np.float_]) -> None:
        self.target = target
        self.models = models

        self.nconformers = models.shape[0]

        self.weights: Optional[NDArray[np.float_]] = None
        self.objective_value: Optional[float] = None

    def _setup_Pq(self) -> None:
        P = np.zeros((self.nconformers, self.nconformers), np.float64)
        q = np.zeros(self.nconformers, np.float64)
        for i in range(self.nconformers):
            for j in range(i, self.nconformers):
                P[i, j] = np.inner(self.models[i], self.models[j])
            q[i] = -np.inner(self.models[i], self.target)

        # OSQP requires quadratic objectives in a scipy CSC sparse matrix
        self.P = sci.sparse.csc_matrix(P)
        self.q = q

    def _setup_constraints(self) -> None:
        self.l = np.zeros(self.nconformers + 1)
        self.u = np.ones(self.nconformers + 1)

        data = np.ones(2 * self.nconformers)
        row_ind = (
            list(range(self.nconformers)) + [self.nconformers] * self.nconformers
        )
        col_ind = list(range(self.nconformers)) * 2
        shape = (self.nconformers + 1, self.nconformers)
        self.A = sci.sparse.csc_matrix((data, (row_ind, col_ind)), shape=shape)

    def solve_qp(self) -> None:
        if TYPE_CHECKING:
            self.driver = self.osqp
            assert self.driver is not None

        self._setup_Pq()
        self._setup_constraints()

        qp = self.driver.OSQP()
        qp.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, **self.OSQP_SETTINGS)
        result = qp.solve()

        self.weights = np.array(result.x).ravel()
        self.objective_value = 2 * result.info.obj_val + np.inner(self.target, self.target)


class MIOSQPSolver(MIQPSolver):
    driver_pkg_name = "miosqp"
    driver = lazy_load_module_if_available(driver_pkg_name)
    if TYPE_CHECKING:
        import miosqp

    MIOSQP_SETTINGS = {
        # integer feasibility tolerance
        "eps_int_feas": 1e-06,
        # maximum number of iterations
        "max_iter_bb": 10000,
        # tree exploration rule
        #   [0] depth first
        #   [1] two-phase: depth first until first incumbent and then best bound
        "tree_explor_rule": 0,
        # branching rule
        #   [0] max fractional part
        "branching_rule": 0,
        "verbose": False,
        "print_interval": 1,
    }

    OSQP_SETTINGS = {
        "eps_abs": 1e-06,
        "eps_rel": 1e-06,
        "eps_prim_inf": 1e-07,
        "verbose": False,
    }

    def __init__(self, target: NDArray[np.float_], models: NDArray[np.float_]) -> None:
        self.target = target
        self.models = models

        self.nconformers = models.shape[0]

        self.weights: Optional[NDArray[np.float_]] = None
        self.objective_value: Optional[float] = None

        # Append raisable Exceptions to SolverError class in module (global) scope
        global SolverError
        SolverError += (ValueError,)

    def _setup_Pq(self) -> None:
        shape = (2 * self.nconformers, 2 * self.nconformers)
        data = []
        row_idx = []
        col_idx = []
        q = np.zeros(2 * self.nconformers, np.float64)
        for i in range(self.nconformers):
            model_i = self.models[i]
            q[i] = -np.inner(model_i, self.target)
            value = np.inner(model_i, model_i)
            data.append(value)
            row_idx.append(i)
            col_idx.append(i)
            for j in range(i + 1, self.nconformers):
                value = np.inner(model_i, self.models[j])
                data.append(value)
                row_idx.append(i)
                col_idx.append(j)
                data.append(value)
                row_idx.append(j)
                col_idx.append(i)
        self.P = sci.sparse.csc_matrix((data, (row_idx, col_idx)), shape=shape)
        self.q = q

    def solve_miqp(
        self,
        threshold: Optional[float] = None,
        cardinality: Optional[int] = None,
        exact: bool = False,
    ) -> None:
        if TYPE_CHECKING:
            self.driver = self.miosqp
            assert self.driver is not None

        if cardinality is threshold is None:
            raise ValueError("Set either cardinality or threshold.")

        self._setup_Pq()

        # The number of restraints are the upper and lower boundaries on
        # each variable plus one for the sum(w_i) <= 1, plus nconformers to
        # set a threshold constraint plus 1 for a cardinality constraint
        # We set first the weights upper and lower bounds, then the sum
        # constraint, then the binary variables upper and lower boundary
        # and then the coupling restraints followed by the threshold
        # contraints and finally a cardinality constraint.
        # A_row effectively contains the constraint indices
        # A_col holds which variables are involved in the constraint
        A_data = [1.0] * (2 * self.nconformers)
        A_row = list(range(self.nconformers)) + [self.nconformers] * self.nconformers
        A_col = list(range(self.nconformers)) * 2
        nconstraints = self.nconformers + 1

        i_l = np.zeros(self.nconformers, np.int32)
        i_u = np.ones(self.nconformers, np.int32)
        i_idx = np.arange(self.nconformers, 2 * self.nconformers, dtype=np.int32)

        # Introduce an implicit cardinality constraint
        # 0 <= zi - wi <= 1
        A_data += [-1.0] * self.nconformers + [1.0] * self.nconformers
        # The wi and zi indices
        start_row = self.nconformers + 1
        A_row += list(range(start_row, start_row + self.nconformers)) * 2
        A_col += list(range(2 * self.nconformers))
        nconstraints += self.nconformers
        if threshold is not None:
            # Introduce threshold constraint
            # 0 <= wi - t * zi <= 1
            A_data += [1.0] * self.nconformers + [-threshold] * self.nconformers
            start_row += self.nconformers
            A_row += list(range(start_row, start_row + self.nconformers)) * 2
            A_col += list(range(2 * self.nconformers))
            nconstraints += self.nconformers

        if cardinality is not None:
            # Introduce explicit cardinality constraint
            # 0 <= sum(zi) <= cardinality
            A_data += [1] * self.nconformers
            A_row += [nconstraints] * self.nconformers
            A_col += list(range(self.nconformers, self.nconformers * 2))
            nconstraints += 1

        l = np.zeros(nconstraints)
        u = np.ones(nconstraints)
        if cardinality is not None:
            u[-1] = cardinality
        A = sci.sparse.csc_matrix((A_data, (A_row, A_col)))

        miqp = self.driver.MIOSQP()
        miqp.setup(self.P, self.q, A, l, u, i_idx, i_l, i_u,
                   self.MIOSQP_SETTINGS, self.OSQP_SETTINGS)  # fmt:skip
        result = miqp.solve()

        self.weights = np.asarray(result.x[0 : self.nconformers])
        self.objective_value = 2 * result.upper_glob + np.inner(self.target, self.target)


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
