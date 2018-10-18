import numpy as np
from scipy import sparse

try:
    import osqp
    import miosqp
    OSQP = True
except ImportError:
    OSQP = False

CPLEX = False
if not OSQP:
    import cvxopt
    import cplex
    CPLEX = True



if OSQP:
    class QPSolver2:

        OSQP_SETTINGS = {
            'eps_abs': 1e-06,
            'eps_rel': 1e-06,
            'eps_prim_inf': 1e-07,
            'verbose': False}

        def __init__(self, target, models):
            self.target = target
            self.models = models
            self.nconformers = models.shape[0]
            self.initialized = False
            self.solution = None
            self.weights = None

        def initialize(self):
            self._setup_Pq()
            self._setup_constraints()
            self.initialized = True

        def _setup_Pq(self):

            shape = (self.nconformers, self.nconformers)
            P = np.zeros(shape, np.float64)
            q = np.zeros(self.nconformers, np.float64)
            for i in range(self.nconformers):
                for j in range(i, self.nconformers):
                    P[i, j] = np.inner(self.models[i], self.models[j])
                q[i] = -np.inner(self.models[i], self.target)
            self.P = sparse.csc_matrix(P)
            self.q = q

        def _setup_constraints(self):
            self.l = np.zeros(self.nconformers + 1)
            self.u = np.ones(self.nconformers + 1)

            data = np.ones(2 * self.nconformers)
            row_ind = list(range(self.nconformers)) + [self.nconformers] * self.nconformers
            col_ind = list(range(self.nconformers))  * 2
            shape = (self.nconformers + 1, self.nconformers)
            self.A = sparse.csc_matrix((data, (row_ind, col_ind)), shape=shape)

        def __call__(self):
            if not self.initialized:
                self.initialize()
            qp = osqp.OSQP()
            qp.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, **self.OSQP_SETTINGS)
            result = qp.solve()
            self.weights = np.asarray(result.x).ravel()
            self.obj_value = result.info.obj_val


    class MIQPSolver2:

        MIOSQP_SETTINGS = {
            # integer feasibility tolerance
            'eps_int_feas': 1e-06,
            # maximum number of iterations
            'max_iter_bb': 10000,
            # tree exploration rule
            #   [0] depth first
            #   [1] two-phase: depth first until first incumbent and then  best bound
            'tree_explor_rule': 0,
            # branching rule
            #   [0] max fractional part
            'branching_rule': 0,
            'verbose': False,
            'print_interval': 1}

        OSQP_SETTINGS = {
            'eps_abs': 1e-06,
            'eps_rel': 1e-06,
            'eps_prim_inf': 1e-07,
            'verbose': False}

        def __init__(self, target, models):
            self.target = target
            self.models = models
            self.nconformers = models.shape[0]
            self.nvariables = 2 * self.nconformers
            self.nbinary = self.nconformers
            self.initialized = False
            self.solution = None
            self.weights = None

        def initialize(self):
            self._setup_Pq()
            self.initialized = True

        def _setup_Pq(self):

            shape = (self.nvariables, self.nvariables)
            data = []
            row_idx = []
            col_idx = []
            q = np.zeros(self.nvariables, np.float64)
            for i in range(self.nconformers):
                for j in range(i, self.nconformers):
                    value = np.inner(self.models[i], self.models[j])
                    data.append(value)
                    row_idx.append(j)
                    col_idx.append(i)
                q[i] = -np.inner(self.models[i], self.target)
            self.P = sparse.csc_matrix((data, (row_idx, col_idx)), shape=shape)
            self.q = q

        def __call__(self, cardinality=None, threshold=None):
            if cardinality is threshold is None:
                raise ValueError("Set either cardinality or threshold.")
            if not self.initialized:
                self.initialize()
            # The number of restraints are the upper and lower boundaries on
            # each variable plus one for the sum(w_i) <= 1, plus nconformers to
            # set a threshold constraint plus 1 for a cardinality constraint
            # We set first the weights upper and lower bounds, then the sum
            # constraint, then the binary variables upper and lower boundary
            # and then the coupling restraints followed by the threshold
            # contraints and finally a cardinality constraint.
            # A_row effectively contains the constraint indices
            # A_col holds which variables are involved in the constraint
            A_data = [1] * (2 * self.nconformers)
            A_row = list(range(self.nconformers)) + [self.nconformers] * self.nconformers
            A_col = list(range(self.nconformers))  * 2
            nconstraints = self.nconformers + 1

            i_l = np.zeros(self.nconformers, np.int32)
            i_u = np.ones(self.nconformers, np.int32)
            i_idx = np.arange(self.nconformers, 2 * self.nconformers, dtype=np.int32)

            # Introduce an implicit cardinality constraint
            # 0 <= zi - wi <= 1
            A_data += [-1] * self.nconformers + [1] * self.nconformers
            # The wi and zi indices
            start_row = self.nconformers + 1
            A_row += list(range(start_row, start_row + self.nconformers)) * 2
            A_col += list(range(2 * self.nconformers))
            nconstraints += self.nconformers
            if threshold is not None:
                # Introduce threshold constraint
                # 0 <= wi - t * zi <= 1
                A_data += [1] * self.nconformers + [-threshold] * self.nconformers
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
            A = sparse.csc_matrix((A_data, (A_row, A_col)))

            miqp = miosqp.MIOSQP()
            miqp.setup(self.P, self.q, A, l, u, i_idx, i_l, i_u,
                       self.MIOSQP_SETTINGS, self.OSQP_SETTINGS)
            result = miqp.solve()
            self.weights = np.asarray(result.x[:self.nconformers])
            self.obj_value = result.upper_glob

            #print("MIOSQP MIQP")
            #w = result.x.reshape(-1, 1)
            #q = self.q.reshape(-1, 1)
            #print('P:', self.P)
            #print('q:', self.q[:self.nconformers])
            #print('w:', w[:self.nconformers])

            #obj = 0.5 * w.T @ self.P @ w + q.T @ w
            #print("calculated myself OBJ:", obj)
            #print('from solver OBJ:', self.obj_value)


if CPLEX:
    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['abstol'] = 1e-8
    cvxopt.solvers.options['reltol'] = 1e-7
    cvxopt.solvers.options['feastol'] = 1e-8


    class QPSolver:

        def __init__(self, target, models):
            self._target = target
            self._models = models
            self._nconformers = models.shape[0]
            self.initialized = False

            self._solution = None
            self.weights = None

        def initialize(self):

            # Set up the matrices and restraints
            self._quad_obj = cvxopt.matrix(0, (self._nconformers, self._nconformers), tc='d')
            self._lin_obj = cvxopt.matrix(self._nconformers * [0], tc='d')
            for i in range(self._nconformers):
                for j in range(i, self._nconformers):
                    self._quad_obj[i,j] = np.inner(self._models[i], self._models[j])
                    # Matrix is symmetric
                    self._quad_obj[j,i] = self._quad_obj[i,j]
                self._lin_obj[i] = -np.inner(self._models[i], self._target)

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
                    self._nconformers * [0.0] + (self._nconformers + 1) * [1.0], tc='d')
            self.initialized = True

        def __call__(self):

            if not self.initialized:
                self.initialize()

            self._solution = cvxopt.solvers.qp(
                    self._quad_obj, self._lin_obj,
                    self._le_constraints, self._le_bounds
                    )
            self.obj_value = self._solution['primal objective']
            self.weights = np.asarray(self._solution['x']).ravel()


    class MIQPSolver:

        """Mixed Integer Quadratic Program based on CPLEX."""

        def __init__(self, target, models, threads=1):
            self._target = target
            self._models = models
            self._nconformers = models.shape[0]
            self.initialized = False
            self.threads = threads

        def initialize(self):

            self._quad_obj = np.zeros((self._nconformers, self._nconformers))
            self._lin_obj = np.zeros(self._nconformers)
            for i in range(self._nconformers):
                for j in range(i, self._nconformers):
                    self._quad_obj[i,j] = np.inner(self._models[i], self._models[j])
                    # Matrix is symmetric
                    self._quad_obj[j,i] = self._quad_obj[i,j]
                self._lin_obj[i] = -np.inner(self._models[i], self._target)

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
                    miqp.objective.set_quadratic_coefficients(i, j, self._quad_obj[i,j])
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

            self.obj_value = miqp.solution.get_objective_value()
            self.weights = np.asarray(miqp.solution.get_values()[:self._nconformers])
            miqp.end()
            #q = self._lin_obj.reshape(-1, 1)
            #P = self._quad_obj
            #w = self.weights.reshape(-1, 1)

            #print("CPLEX MIQP")
            #print('P:', P)
            #print('q:', q)
            #print('w:', w)

            #obj = 0.5 * w.T @ P @ w + q.T @ w
            #print("calculated myself OBJ:", obj)
            #print('from solver OBJ:', self.obj_value)
