'''
Excited States software: qFit 3.0

Contributors: Saulo H. P. de Oliveira, Gydo van Zundert, and Henry van den Bedem.
Contact: vdbedem@stanford.edu

Copyright (C) 2009-2019 Stanford University
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

This entire text, including the above copyright notice and this permission notice
shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
'''

import logging

import numpy as np
from scipy import sparse


logger = logging.getLogger(__name__)


# Try load solver sets
try:
    import cvxopt
    import cplex
except ImportError:
    CPLEX = False
else:
    CPLEX = True


# Only these classes should be "public"
__all__ = ['QPSolver', 'MIQPSolver']


# Define the required functions for (MI)QP solver objects
class _Base_QPSolver(object):
    """Base class for Quadratic Programming solvers.

    Declares required interface functions for child classes to overwrite."""

    def initialize(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


# If we can load the CPLEX solver set,
# provide class definitions for a QP and MIQP solver.
if CPLEX:
    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['abstol'] = 1e-8
    cvxopt.solvers.options['reltol'] = 1e-7
    cvxopt.solvers.options['feastol'] = 1e-8

    class CPLEX_QPSolver(_Base_QPSolver):
        """Quadratic Programming solver based on CPLEX."""

        def __init__(self, target, models):
            self._target = target
            self._models = models
            self._nconformers = models.shape[0]
            self.initialized = False

            self._solution = None
            self.weights = None

        def initialize(self):
            # Set up the matrices and restraints
            logger.debug(f"Building cvxopt matrix, size: ({self._nconformers},{self._nconformers})")
            self._quad_obj = cvxopt.matrix(np.inner(self._models, self._models), tc='d')
            self._lin_obj = cvxopt.matrix(-np.inner(self._models, self._target), tc='d')

            # lower-equal constraints.
            # Each weight falls in the closed interval [0..1] and its sum is <= 1.
            # There are 2 * nconformers bounds + 1 constraint
            # Make a sparse matrix to represent this information.
            self._le_constraints = cvxopt.spmatrix(
                self._nconformers * [-1.0] + 2 * self._nconformers * [1.0],
                list(range(2 * self._nconformers)) + self._nconformers * [2 * self._nconformers],
                3 * list(range(self._nconformers)),
            )
            self._le_bounds = cvxopt.matrix(
                self._nconformers * [0.0] + (self._nconformers + 1) * [1.0],
                tc='d')
            self.initialized = True

        def __call__(self):
            if not self.initialized:
                self.initialize()

            self._solution = cvxopt.solvers.qp(
                self._quad_obj, self._lin_obj,
                self._le_constraints, self._le_bounds
            )
            self.obj_value = 2 * self._solution['primal objective'] + np.inner(self._target, self._target)
            self.weights = np.asarray(self._solution['x']).ravel()

    class CPLEX_MIQPSolver(_Base_QPSolver):
        """Mixed-Integer Quadratic Programming solver based on CPLEX."""

        def __init__(self, target, models, threads=1):
            self._target = target
            self._models = models
            self._nconformers = models.shape[0]
            self.initialized = False
            self.threads = threads

        def initialize(self):
            self._quad_obj = np.inner(self._models, self._models)
            self._lin_obj = -np.inner(self._models, self._target)

            self.initialized = True

        def __call__(self, cardinality=None, exact=False, threshold=None):
            if not self.initialized:
                self.initialize()

            miqp = cplex.Cplex()
            miqp.set_results_stream(None)
            miqp.set_log_stream(None)
            miqp.set_warning_stream(None)
            miqp.set_error_stream(None)

            # Set number of threads to use
            if self.threads is not None:
                miqp.parameters.threads.set(self.threads)

            # Setup QP part of the MIQP
            variable_names = [f'w{n}' for n in range(self._nconformers)]
            upper_bounds = self._nconformers * [1.0]

            miqp.variables.add(names=variable_names, ub=upper_bounds)
            for i in range(self._nconformers):
                for j in range(i, self._nconformers):
                    miqp.objective.set_quadratic_coefficients(i, j, self._quad_obj[i, j])
                miqp.objective.set_linear(i, self._lin_obj[i])

            # Sum of weights is <= 1
            ind = range(self._nconformers)
            val = [1] * self._nconformers
            lin_expr = [cplex.SparsePair(ind=ind, val=val)]
            miqp.linear_constraints.add(
                lin_expr=lin_expr,
                rhs=[1],
                senses=["L"],
            )

            # If cardinality or threshold is specified the problem is a MIQP, else its
            # a regular QP.
            if cardinality not in (None, 0) or threshold not in (None, 0):
                integer_names = [f'z{n}' for n in range(self._nconformers)]
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
                            lin_expr=[cplex.SparsePair(ind=[z, w], val=[threshold, -1])],
                            rhs=[0],
                            senses=["L"],
                        )
                # Set the cardinality constraint
                if cardinality not in (None, 0):
                    senses = "L"
                    if exact:
                        senses = "E"
                        cardinality = min(cardinality, self._nconformers)
                    lin_expr = [[list(range(self._nconformers, 2 * self._nconformers)), self._nconformers * [1]]]
                    miqp.linear_constraints.add(
                        lin_expr=lin_expr,
                        rhs=[cardinality],
                        senses=senses,
                    )
            miqp.solve()

            self.obj_value = 2 * miqp.solution.get_objective_value() + np.inner(self._target, self._target)
            self.weights = np.asarray(miqp.solution.get_values()[:self._nconformers])
            miqp.end()
            q = self._lin_obj.reshape(-1, 1)
            P = self._quad_obj
            w = self.weights.reshape(-1, 1)

            #print("CPLEX MIQP")
            #print('P:', P)
            #print('q:', q)
            #print('w:', w)

            #obj = 0.5 * w.T @ P @ w + q.T @ w
            #print("calculated myself OBJ:", obj)
            #print('from solver OBJ:', self.obj_value)
            #print("TOTAL:", self.obj_value * 2 + np.inner(self._target, self._target))


# Create a "pseudo-class" to abstract the choice of solver.
class QPSolver(object):
    def __new__(cls, *args, **kwargs):
        """Return a constructor for the appropriate solver."""

        # Default to using cplex if not specified.
        use_cplex = kwargs.pop("use_cplex", True)

        # Walk through solver decision tree.
        if use_cplex:
            if CPLEX:
                return CPLEX_QPSolver(*args, **kwargs)
            else:
                raise ImportError("qFit could not load modules for Quadratic Programming solver.\n"
                                  "Please install cvxopt & CPLEX.")
        else:
            if CPLEX:
                print("WARNING: A different solver was requested, but only CPLEX solvers found.\n"
                      "         Using CPLEX solver as fallback.")
                return CPLEX_QPSolver(*args, **kwargs)
            else:
                raise ImportError("qFit could not load modules for Quadratic Programming solver.\n"
                                  "Please install cvxopt & CPLEX.")

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
                raise ImportError("qFit could not load modules for Quadratic Programming solver.\n"
                                  "Please install cvxopt & CPLEX.")
        else:
            if CPLEX:
                print("WARNING: A different solver was requested, but only CPLEX solvers found.\n"
                      "         Using CPLEX solver as fallback.")
                return CPLEX_MIQPSolver(*args, **kwargs)
            else:
                raise ImportError("qFit could not load modules for Quadratic Programming solver.\n"
                                  "Please install cvxopt & CPLEX.")

    def __init__(self, *args, **kwargs):
        # This pseudo-class does not generate class instances.
        # As there are no instances of this class,
        #   it does not need an instance initialiser.
        raise NotImplementedError
