from __future__ import annotations

import inspect
import logging
import sys
from abc import ABC, abstractmethod
from types import ModuleType
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np
import scipy as sci
import scipy.sparse  # pylint: disable=unused-import
from numpy.typing import NDArray
from cplex.exceptions import CplexSolverError

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
        global SolverError
        SolverErrors = (SolverError, CplexSolverError)

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

        try:
            result = miqp.solve()
        except CplexSolverError:
            raise SolverError(
                "CPLEX encountered an error: Non-convex objective function"
            )

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

        self.quad_obj: sci.sparse.csc_matrix
        self.lin_obj: NDArray[np.float_]
        self.constraints: sci.sparse.csc_matrix
        self.lower_bounds: NDArray[np.float_]
        self.upper_bounds: NDArray[np.float_]

        self.weights: Optional[NDArray[np.float_]] = None
        self.objective_value: Optional[float] = None

    def compute_quadratic_coeffs(self) -> None:
        # minimize 0.5 x.T P x + q.T x
        #   where P = self.quad_obj =   ρ_model.T ρ_model
        #         q = self.lin_obj  = - ρ_model.T ρ_obs
        quad_obj_coeffs = np.inner(self.models, self.models)
        lin_obj_coeffs = -np.inner(self.models, self.target)

        # OSQP requires quadratic objectives in a scipy CSC sparse matrix
        self.quad_obj = sci.sparse.csc_matrix(quad_obj_coeffs)
        self.lin_obj = lin_obj_coeffs

    def compute_constraints(self) -> None:
        # subject to:
        #   l ≤ A x ≤ u
        #   where l = self.lower_bounds
        #         A = self.constraints
        #         u = self.upper_bounds
        #
        #   Each weight x falls in the closed interval [0..1], and the sum of all weights is <= 1.
        #   This corresponds to (1 * nconformers + 1) constraints, imposed on (nconformers) variables:
        #     the lower and upper bounds account for (nconformers) constraints,
        #     the summation constraint accounts for 1.
        shape = (self.nconformers + 1, self.nconformers)
        #   We construct A with a sparse matrix.
        #         constraint idx=[0..N)           constraint idx=N
        rowidxs = [*range(0, self.nconformers)] + (self.nconformers * [self.nconformers])  # fmt:skip
        #         0 ≤ x_i ≤ 1                     0 ≤ Σ_i x_i ≤ 1
        coeffs  = (self.nconformers * [1.0])    + (self.nconformers * [1.0])  # fmt:skip
        colidxs = [*range(0, self.nconformers)] + [*range(0, self.nconformers)]  # fmt:skip
        lowers  = (self.nconformers * [0.0])    + [0.0]  # fmt:skip
        uppers  = (self.nconformers * [1.0])    + [1.0]  # fmt:skip

        self.constraints = sci.sparse.csc_matrix(
            (coeffs, (rowidxs, colidxs)),
            shape=shape,
        )
        self.lower_bounds = np.array(lowers)
        self.upper_bounds = np.array(uppers)

    def solve_qp(self) -> None:
        if TYPE_CHECKING:
            self.driver = self.osqp
            assert self.driver is not None

        self.compute_quadratic_coeffs()
        self.compute_constraints()

        qp = self.driver.OSQP()
        qp.setup(
            P=self.quad_obj,
            q=self.lin_obj,
            A=self.constraints,
            l=self.lower_bounds,
            u=self.upper_bounds,
            **self.OSQP_SETTINGS,
        )
        result = qp.solve()

        self.weights = np.array(result.x).ravel()
        self.objective_value = (
            2 * result.info.obj_val
            + np.inner(self.target, self.target)
        )  # fmt:skip


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

        self.quad_obj: Optional[sci.sparse.csc_matrix] = None
        self.lin_obj: Optional[NDArray[np.float_]] = None
        self.constraints: sci.sparse.csc_matrix
        self.lower_bounds: NDArray[np.float_]
        self.upper_bounds: NDArray[np.float_]
        self.binary_vars: NDArray[np.int_]
        self.lower_ints: NDArray[np.int_]
        self.upper_ints: NDArray[np.int_]

        self.weights: Optional[NDArray[np.float_]] = None
        self.objective_value: Optional[float] = None

        # Append raisable Exceptions to SolverError class in module (global) scope
        global SolverError
        SolverErrors = (SolverError, CplexSolverError)

    def compute_quadratic_coeffs(self) -> None:
        """Precompute the quadratic coefficients (P, q).

        These values don't depend on threshold/cardinality.
        Having these objectives pre-computed saves a few cycles when MIQP
        is evaluated for multiple values of threshold/cardinality.
        """
        # Since this is an MIQP problem, the objective function f(x)
        # takes the vector x = [w_0 .. w_i, z_0 .. z_i],
        #   where w_i are the weights,
        #         z_i are the integer selections.

        # We wish to minimize 0.5 x.T P x + q.T x
        #   where P = self.quad_obj =   ρ_model.T ρ_model
        #         q = self.lin_obj  = - ρ_model.T ρ_obs
        quad_obj_coeffs = np.inner(self.models, self.models)
        lin_obj_coeffs = -np.inner(self.models, self.target)

        # This is sufficient for the non-integer weights,
        #   but we must extend our matrix so that
        #     P is of shape (2*nconfs, 2*nconfs), and
        #     q is of len (2*nconfs),
        #   to match the dimensions of vector x.
        P_shape = (2 * self.nconformers, 2 * self.nconformers)

        # MIOSQP requires quadratic objectives in a scipy CSC sparse matrix
        # Get an index into the dense _quad_obj_coeffs array,
        #   and construct P with csc_array((data, (row_ind, col_ind)), [shape=(M, N)]).
        rowidx, colidx = np.indices(quad_obj_coeffs.shape)
        self.quad_obj = sci.sparse.csc_matrix(
            (quad_obj_coeffs.ravel(), (rowidx.ravel(), colidx.ravel())),
            shape=P_shape,
        )
        # Extend q to appropriate shape
        self.lin_obj = np.append(lin_obj_coeffs, np.zeros((self.nconformers,)))

    def compute_mixed_int_constraints(
        self,
        threshold: Optional[float],
        cardinality: Optional[int],
    ) -> None:
        # subject to:
        #   l ≤ A x ≤ u
        #   x[i] ∈ Z, for i in i_idx
        #   il[i] <= x[i] <= iu[i], for i in i_idx
        #   where l = self.lower_bounds
        #         A = self.constraints
        #         u = self.upper_bounds
        #         i_idx = self.binary_vars: a vector of indices of which variables are integer
        #         il = self.lower_ints: the lower bounds on the integer variables
        #         iu = self.upper_ints: the upper bounds on the integer variables.

        # We will construct A with a sparse matrix.

        # Presently, we have no constraints.
        n_constraints = 0
        # Our constraints will be applied over the variables [w_0 .. w_i, z_0 .. z_i],
        #   where w_i are the weights,
        #         z_i are the integer selections.
        n_vars = 2 * self.nconformers

        # Since cardinality or threshold will be specified, the problem is a MIQP,
        # so we need to add binary integer variables z_i.
        #     z_i ∈ {0, 1}
        self.binary_vars = np.arange(self.nconformers, 2 * self.nconformers)
        self.lower_ints = np.array(self.nconformers * [0])
        self.upper_ints = np.array(self.nconformers * [1])

        # fmt:off

        #   Each weight w_i falls in the closed interval [0..1], and the sum of all weights is <= 1.
        #   This corresponds to (1 * nconformers + 1) constraints, imposed on (nconformers) variables:
        #     the lower and upper bounds account for (nconformers) constraints,
        #     the summation constraint accounts for 1.
        #        constraint idx=[0..N)           constraint idx=N
        rowidx = [*range(0, self.nconformers)] + (self.nconformers * [self.nconformers])
        #        0 ≤ w_i ≤ 1                     0 ≤ Σ_i w_i ≤ 1
        coeffs = (self.nconformers * [1.0])    + (self.nconformers * [1.0])
        colidx = [*range(0, self.nconformers)] + [*range(0, self.nconformers)]
        lowers = (self.nconformers * [0.0])    + [0.0]
        uppers = (self.nconformers * [1.0])    + [1.0]
        n_constraints += self.nconformers + 1

        # Introduce an implicit cardinality constraint
        # Only count weights for which z_i is 1
        #     0 <= z_i - w_i <= 1
        #     (∵ z_i ∈ {0,1} and 0 ≤ w_i ≤ 1, this is only true when z_i = 1)
        # constraint idx=[N+1..2N+1)
        rowidx += (
            [*range(n_constraints, n_constraints + self.nconformers)] +
            [*range(n_constraints, n_constraints + self.nconformers)]
        )
        #    0 <= -w_i                           + z_i <= 1
        coeffs += ((self.nconformers * [-1.0])   + (self.nconformers * [1.0]))
        colidx += ([*range(0, self.nconformers)] + [*range(self.nconformers, 2 * self.nconformers)])
        lowers += self.nconformers * [0.0]
        uppers += self.nconformers * [1.0]
        n_constraints += self.nconformers

        # Set the threshold constraint if applicable
        #     tdmin z_i - w_i ≤ 0, i.e. w_i ≥ tdmin
        if threshold is not None:
            # constraint idx=[2N+1..3N+1)
            rowidx += (
                [*range(n_constraints, n_constraints + self.nconformers)] +
                [*range(n_constraints, n_constraints + self.nconformers)]
            )
            #    0 <= wi                            - t * zi <= 1
            coeffs += (self.nconformers * [1.0])    + (self.nconformers * [-threshold])
            colidx += [*range(0, self.nconformers)] + [*range(self.nconformers, 2 * self.nconformers)]
            lowers += (self.nconformers * [0.0])
            uppers += (self.nconformers * [1.0])
            n_constraints += self.nconformers

        # fmt:on

        # Set the cardinality constraint if applicable
        #     Σ z_i ≤ cardinality
        if cardinality is not None:
            # constraint idx=2N+2, if no threshold constraint; or
            # constraint idx=3N+2, if a threshold constraint was applied
            rowidx += self.nconformers * [n_constraints]
            #    0 <= Σ z_i <= cardinality
            coeffs += self.nconformers * [1]
            colidx += [*range(self.nconformers, 2 * self.nconformers)]
            lowers += [0]
            uppers += [cardinality]
            n_constraints += 1

        # MIOSQP requires constraints as a sparse matrix
        self.constraints = sci.sparse.csc_matrix(
            (coeffs, (rowidx, colidx)),
            shape=(n_constraints, n_vars),
        )
        self.lower_bounds = np.array(lowers)
        self.upper_bounds = np.array(uppers)

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

        if not (self.quad_obj and self.lin_obj):
            self.compute_quadratic_coeffs()
        self.compute_mixed_int_constraints(threshold, cardinality)

        # Construct the MIOSQP solver & solve!
        miqp = self.driver.MIOSQP()
        miqp.setup(
            P=self.quad_obj,
            q=self.lin_obj,
            A=self.constraints,
            l=self.lower_bounds,
            u=self.upper_bounds,
            i_idx=self.binary_vars,
            i_l=self.lower_ints,
            i_u=self.upper_ints,
            settings=self.MIOSQP_SETTINGS,
            qp_settings=self.OSQP_SETTINGS,
        )
        try:
            result = miqp.solve()
        except CplexSolverError:
            raise SolverError(
                "CPLEX encountered an error: Non-convex objective function"
            )

        # Destructure results
        self.weights = np.array(result.x[0 : self.nconformers])
        self.objective_value = (
            2 * result.upper_glob
            + np.inner(self.target, self.target)
        )  # fmt:skip


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
