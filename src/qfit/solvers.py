import logging

import numpy as np

logger = logging.getLogger(__name__)
SolverError = ImportError("Could not load a valid solver.")


# Try load solver sets
try:
    import cvxopt
    import cplex
except ImportError:
    CPLEX = False
    msg = (
        "CPLEX or CVXOPT is not install. Please re-install these packages and qFit to run"
          )
    raise RuntimeError(msg)
else:
    CPLEX = True
    SolverError = cplex.exceptions.CplexSolverError


# Only these classes should be "public"
__all__ = ["QPSolver", "MIQPSolver", "SolverError"]


# Define the required functions for (MI)QP solver objects
class _Base_QPSolver(object):
    """Base class for Quadratic Programming solvers.

    Declares required interface functions for child classes to overwrite.
    """

    def solve(self):
        raise NotImplementedError


# If we can load the CPLEX solver set,
# provide class definitions for a QP and MIQP solver.
if CPLEX:
    cvxopt.solvers.options["show_progress"] = False
    cvxopt.solvers.options["abstol"] = 1e-8
    cvxopt.solvers.options["reltol"] = 1e-7
    cvxopt.solvers.options["feastol"] = 1e-8

    class CPLEX_QPSolver(_Base_QPSolver):
        """Quadratic Programming solver based on CPLEX."""

        def __init__(self, target, models):
            self._target = target
            self._models = models
            self._nconformers = models.shape[0]

            self._solution = None
            self.weights = None

        def solve(self):
            # Set up the matrices and restraints
            logger.debug(
                f"Building cvxopt matrix, size: ({self._nconformers},{self._nconformers})"
            )

            # minimize 0.5 x.T P x + q.T x
            #   where P = self._quad_obj      =   ρ_model.T ρ_model
            #         q = self._lin_obj       = - ρ_model.T ρ_obs
            self._quad_obj = cvxopt.matrix(np.inner(self._models, self._models), tc="d")
            self._lin_obj = cvxopt.matrix(-np.inner(self._models, self._target), tc="d")

            # subject to:
            #   G x ≤ h
            #   where G = self._le_constraints
            #         h = self._le_bounds
            #
            #   Each weight x falls in the closed interval [0..1], and the sum of all weights is <= 1.
            #   This corresponds to 2 * nconformers bounds + 1 constraint.
            #   We construct G with a sparse matrix.
            #         constraint idx=[0..N)            constraint idx=[N..2N)                               constraint idx=2N
            rowidxs = [*range(0, self._nconformers)] + [*range(self._nconformers, 2 * self._nconformers)] + (self._nconformers * [2 * self._nconformers])  # fmt:skip
            #         -x_i ≤ 0                         x_i ≤ 1                                              Σ_i x_i ≤ 1
            coeffs  = (self._nconformers * [-1.0])   + (self._nconformers * [1.0])                        + (self._nconformers * [1.0])  # fmt:skip
            colidxs = [*range(0, self._nconformers)] + [*range(0, self._nconformers)]                     + [*range(0, self._nconformers)]  # fmt:skip
            threshs = (self._nconformers * [0.0])    + (self._nconformers * [1.0])                        + [1.0]  # fmt:skip
            self._le_constraints = cvxopt.spmatrix(coeffs, rowidxs, colidxs, tc="d")
            self._le_bounds = cvxopt.matrix(threshs, tc="d")

            # Solve
            self._solution = cvxopt.solvers.qp(
                self._quad_obj, self._lin_obj, self._le_constraints, self._le_bounds
            )

            # Store the density residual and the weights
            self.obj_value = (
                2 * self._solution["primal objective"] + 
                np.inner(self._target, self._target)
            )  # fmt:skip
            self.weights = np.asarray(self._solution["x"]).ravel()

    class CPLEX_MIQPSolver(_Base_QPSolver):
        """Mixed-Integer Quadratic Programming solver based on CPLEX."""

        def __init__(self, target, models, threads=1):
            self.initialized = False
            self._target = target
            self._models = models
            self._nconformers = models.shape[0]
            self.threads = threads

        def _initialize(self):
            """Precompute two inner products.

            These values don't depend on threshold/cardinality.
            Having these objectives pre-computed saves a few cycles when MIQP
            is evaluated for multiple values of threshold/cardinality.
            """
            self._quad_obj = np.inner(self._models, self._models)
            self._lin_obj = -np.inner(self._models, self._target)

            self.initialized = True

        def solve(self, cardinality=None, exact=False, threshold=None):
            if not self.initialized:
                self._initialize()

            miqp = cplex.Cplex()
            miqp.set_results_stream(None)
            miqp.set_log_stream(None)
            miqp.set_warning_stream(None)
            miqp.set_error_stream(None)

            # Set number of threads to use
            if self.threads is not None:
                miqp.parameters.threads.set(self.threads)

            # Setup QP part of the MIQP
            variable_names = [f"w{n}" for n in range(self._nconformers)]
            upper_bounds = self._nconformers * [1.0]

            miqp.variables.add(names=variable_names, ub=upper_bounds)
            for i in range(self._nconformers):
                for j in range(i, self._nconformers):
                    miqp.objective.set_quadratic_coefficients(
                        i, j, self._quad_obj[i, j]
                    )
                miqp.objective.set_linear(i, self._lin_obj[i])
            ind = range(self._nconformers)
            val = [1] * self._nconformers
            lin_expr = [cplex.SparsePair(ind=ind, val=val)]
            miqp.linear_constraints.add(
                lin_expr=lin_expr,
                rhs=[1],
                senses=["L"],
            )  # Sum of weights is <= 1

            # If cardinality or threshold is specified the problem is a MIQP, else its
            # a regular QP.
            if cardinality not in (None, 0) or threshold not in (None, 0):
                integer_names = [f"z{n}" for n in range(self._nconformers)]
                variable_types = self._nconformers * miqp.variables.type.binary
                miqp.variables.add(names=integer_names, types=variable_types)

                # Only count weights for which zi is 1
                for n in range(self._nconformers):
                    w = f"w{n}"
                    z = f"z{n}"
                    miqp.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind=[w, z], val=[1, -1])],
                        rhs=[0],
                        senses="L",
                    )
                    # Set the threshold constraint
                    if threshold not in (None, 0):
                        miqp.linear_constraints.add(
                            lin_expr=[
                                cplex.SparsePair(ind=[z, w], val=[threshold, -1])
                            ],
                            rhs=[0],
                            senses=["L"],
                        )
                # Set the cardinality constraint
                if cardinality not in (None, 0):
                    senses = "L"
                    if exact:
                        senses = "E"
                        cardinality = min(cardinality, self._nconformers)
                    lin_expr = [
                        [
                            list(range(self._nconformers, 2 * self._nconformers)),
                            self._nconformers * [1],
                        ]
                    ]
                    miqp.linear_constraints.add(
                        lin_expr=lin_expr,
                        rhs=[cardinality],
                        senses=senses,
                    )
            miqp.solve()

            self.obj_value = 2 * miqp.solution.get_objective_value() + np.inner(
                self._target, self._target
            )
            self.weights = np.asarray(miqp.solution.get_values()[: self._nconformers])
            miqp.end()
            q = self._lin_obj.reshape(-1, 1)
            P = self._quad_obj
            w = self.weights.reshape(-1, 1)

            # print("CPLEX MIQP")
            # print('P:', P)
            # print('q:', q)
            # print('w:', w)

            # obj = 0.5 * w.T @ P @ w + q.T @ w
            # print("calculated myself OBJ:", obj)
            # print('from solver OBJ:', self.obj_value)
            # print("TOTAL:", self.obj_value * 2 + np.inner(self._target, self._target))


# Create a "pseudo-class" to abstract the choice of solver.
class QPSolver(object):
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

    def __new__(cls, *args, **kwargs):
        """Return a constructor for the appropriate solver."""

        # Default to using cplex if not specified.
        use_cplex = kwargs.pop("use_cplex", True)

        # Walk through solver decision tree.
        if use_cplex:
            if CPLEX:
                return CPLEX_QPSolver(*args, **kwargs)
            else:
                raise ImportError(
                    "qFit could not load modules for Quadratic Programming solver.\n"
                    "Please install cvxopt & CPLEX."
                )
        else:
            if CPLEX:
                print(
                    "WARNING: A different solver was requested, but only CPLEX solvers found.\n"
                    "         Using CPLEX solver as fallback."
                )
                return CPLEX_QPSolver(*args, **kwargs)
            else:
                raise ImportError(
                    "qFit could not load modules for Quadratic Programming solver.\n"
                    "Please install cvxopt & CPLEX."
                )

    def __init__(self, *args, **kwargs):
        # This pseudo-class does not generate class instances.
        # As there are no instances of this class,
        #   it does not need an instance initialiser.
        raise NotImplementedError


# Create a "pseudo-class" to abstract the choice of solver.
class MIQPSolver(object):
    def __new__(cls, *args, **kwargs):
        """Construct an initialised instance of the appropriate solver."""

        # Default to using cplex if not specified.
        use_cplex = kwargs.pop("use_cplex", True)

        # Walk through solver decision tree.
        if use_cplex:
            if CPLEX:
                return CPLEX_MIQPSolver(*args, **kwargs)
            else:
                raise ImportError(
                    "qFit could not load modules for Quadratic Programming solver.\n"
                    "Please install cvxopt & CPLEX."
                )
        else:
            if CPLEX:
                print(
                    "WARNING: A different solver was requested, but only CPLEX solvers found.\n"
                    "         Using CPLEX solver as fallback."
                )
                return CPLEX_MIQPSolver(*args, **kwargs)
            else:
                raise ImportError(
                    "qFit could not load modules for Quadratic Programming solver.\n"
                    "Please install cvxopt & CPLEX."
                )

    def __init__(self, *args, **kwargs):
        # This pseudo-class does not generate class instances.
        # As there are no instances of this class,
        #   it does not need an instance initialiser.
        raise NotImplementedError
