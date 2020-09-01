











class Evolver:
    """Parent of OdeSolver used by Qutip quantum system solvers.
    Do not use directly, but use child class.

    Methods
    -------
    set(state, tlist)
    step()
        Create copy of Qobj
    evolve(state, tlist, progress_bar)

    """
    def __init__(self, system, options):
        self.system = system
        self.options = options
        self.name = "undefined"
        #self.progress_bar = progress_bar
        #self.statetype = "dense"
        #self.normalize_func = dummy_normalize
        self._state = None
        self._tlist = None
        self._error_msg = ("ODE integration error: Try to increase "
                           "the allowed number of substeps by increasing "
                           "the nsteps parameter in the Options class.")
        self._dm = False
        self._oper = True

    def evolve(self, state0, tlist, progress_bar):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def set(self, state, tlist):
        pass

    def update_args(self, args):
        self.LH.arguments(args)

    @staticmethod
    def funcwithfloat(func):
        def new_func(t, y):
            y_cplx = y.view(complex)
            dy = func(t, y_cplx)
            return dy.view(np.float64)
        return new_func

    def normalize(self, state):
        # TODO cannot be used for propagator evolution.
        if self._dm:
            norm = _data.trace(state)
        elif self._oper:
            norm = _data.la_norm(state)  / state.shape[0]
        else:
            norm = _data.la_norm(state)
        state /= norms
        return abs(norm-1)


class EvolverScipyZvode(Evolver):
    # -------------------------------------------------------------------------
    # Solve an ODE for func.
    # Calculate the required expectation values or invoke callback
    # function at each time step.
    #
    # Use scipy's Ode, with zvode solver
    #
    def __init__(self, system, options):
        super(OdeScipyZvode, self).__init__(system, options)
        self.name = "scipy_zvode"

    def set(self, state0, tlist):
        self.tlist = tlist
        self._i = 0
        if state0.shape[1] > 1:
            self._mat_state = True
            self._size = state0.shape[0]
            if abs(_data.trace(state0) - 1) > self.options['rtol']:
                self._oper = True
            else:
                self._dm = True

        else:
            self._mat_state = False

        r = ode(self.system.get_mul(state0))
        options_keys = ['atol', 'rtol', 'nsteps', 'method', 'order',
                        'first_step', 'max_step', 'min_step']
        options = {key: self.options[key] for key in options_keys}
        r.set_integrator('zvode', **options)

        r.set_initial_value(_data.column_stack(state).to_array(), t0)
        self._r = r
        self._prepare_normalize_func(state0)

    def step(self):
        if self._i < len(self.tlist):
            self._i += 1
            self._r.integrate(self.tlist[self._i])
            if not self._r.successful():
                raise Exception(self._error_msg)
            state = self.get_state()
            if self._normalize:
                if self.normalize(state):
                    self._r.set_initial_value(state.as_ndarray(),
                                              self.tlist[self._i])
            yield state

    def evolve(self, state, tlist, progress_bar):
        self.set(state, tlist)
        for t in self.tlist[1:]:
            self._r.integrate(t)
            if not self._r.successful():
                raise Exception(self._error_msg)
            progress_bar.update()
            state = self.get_state()
            if self._normalize:
                if self.normalize(state) >= self.options["renorm_tol"]:
                    self._r.set_initial_value(state.as_ndarray(),
                                              self.tlist[self._i])
            states.append(state.copy())
        return states

    def get_state(self):
        if self._mat_state:
            return _data.column_unstack_dense(
                _data.dense.fast_from_numpy(self._r.y),
                self._size,
                inplace=True)
        else:
            return _data.dense.fast_from_numpy(self._r.y)

"""

class EvolverScipyDop853(Evolver):
    # -------------------------------------------------------------------------
    # Solve an ODE for func.
    # Calculate the required expectation values or invoke callback
    # function at each time step.
    #
    # Use scipy's Ode, with dop853 solver
    #
    def __init__(self, LH, options, progress_bar):
        super(OdeScipyDop853, self).__init__(LH, options, progress_bar)
        self.name = "scipy_dop853"

    def run(self, state0, tlist, args={}, e_ops=[]):
        """
        Internal function for solving ODEs.
        """
        opt = self.options
        normalize_func = self.normalize_func
        e_ops = self._prepare_e_ops(e_ops)
        self.LH.arguments(args)
        n_tsteps = len(tlist)
        state_size = np.prod(state0.shape)
        num_saved_state = n_tsteps if opt.store_states else 1

        states = np.zeros((num_saved_state, state_size), dtype=complex)
        e_ops_store = bool(e_ops)
        e_ops.init(tlist)

        self.set(state0, tlist[0])
        r = self._r

        self.progress_bar.start(n_tsteps-1)
        for t_idx, t in enumerate(tlist):
            if not r.successful():
                raise Exception(self._error_msg)
            # get the current state / oper data if needed
            if opt.store_states or opt.normalize_output or e_ops_store:
                cdata = r.y.view(complex)
                if self.normalize_func(cdata) > opt.atol:
                    r.set_initial_value(cdata.view(np.float64), r.t)
                if opt.store_states:
                    states[t_idx, :] = cdata
                e_ops.step(t_idx, cdata)
            if t_idx < n_tsteps - 1:
                r.integrate(tlist[t_idx+1])
                self.progress_bar.update(t_idx)
        self.progress_bar.finished()
        states[-1, :] = r.y.view(complex)
        self.normalize_func(states[-1, :])
        return states, e_ops.finish()

    def step(self, t, reset=False, changed=False):
        if reset:
            self.set(self._r.y.view(complex), self._r.t)
        self._r.integrate(t)
        state = self._r.y.view(complex)
        self.normalize_func(state)
        return state

    def set(self, state0, t0):
        opt = self.options
        func = self.LH._get_mul(state0)
        r = ode(self.funcwithfloat(func))
        options_keys = ['atol', 'rtol', 'nsteps', 'first_step', 'max_step',
                        'ifactor', 'dfactor', 'beta']
        options = {key: getattr(opt, key)
                   for key in options_keys
                   if hasattr(opt, key)}
        r.set_integrator('dop853', **options)
        if isinstance(state0, Qobj):
            initial_vector = state0.full().ravel('F').view(np.float64)
        else:
            initial_vector = state0.view(np.float64)
        r.set_initial_value(initial_vector, t0)
        self._r = r
        self._prepare_normalize_func(state0)


class EvolverScipyIVP(Evolver):
    # -------------------------------------------------------------------------
    # Solve an ODE for func.
    # Calculate the required expectation values or invoke callback
    # function at each time step.
    #
    # Use scipy's Solve_ivp
    #
    def __init__(self, LH, options, progress_bar):
        super(OdeScipyIVP, self).__init__(LH, options, progress_bar)
        self.name = "scipy_ivp"

    def run(self, state0, tlist, args={}, e_ops=[]):
        # TODO: normalization in solver
        # ?> v1: event, step when norm bad
        # ?> v2: extra non-hermitian term to H
        opt = self.options
        normalize_func = self.normalize_func
        e_ops = self._prepare_e_ops(e_ops)
        self.LH.arguments(args)
        n_tsteps = len(tlist)
        state_size = np.prod(state0.shape)
        num_saved_state = n_tsteps if opt.store_states else 1
        states = np.zeros((num_saved_state, state_size), dtype=complex)
        e_ops_store = bool(e_ops)
        e_ops.init(tlist)

        self.set(state0, tlist[0])
        ode_res = solve_ivp(self.func, [tlist[0], tlist[-1]],
                            self._y, t_eval=tlist, **self.ivp_opt)

        e_ops.init(tlist)
        for t_idx, cdata in enumerate(ode_res.y.T):
            y_cplx = cdata.copy().view(complex)
            self.normalize_func(y_cplx)
            if opt.store_states:
                states[t_idx, :] = y_cplx
            e_ops.step(t_idx, y_cplx)
        if not opt.store_states:
            states[0, :] = y_cplx
        return states, e_ops.finish()

    def step(self, t, reset=False, changed=False):
        ode_res = solve_ivp(self.func, [self._t, t], self._y,
                            t_eval=[t], **self.ivp_opt)
        self._y = ode_res.y.T[0]
        state = self._y.copy().view(complex)
        self._t = t
        self.normalize_func(state)
        return state

    def set(self, state0, t0):
        opt = self.options
        self._t = t0
        self.func = self.funcwithfloat(self.LH._get_mul(state0))
        if isinstance(state0, Qobj):
            self._y = state0.full().ravel('F').view(np.float64)
        else:
            self._y = state0.view(np.float64)

        options_keys = ['method', 'atol', 'rtol', 'nsteps']
        self.ivp_opt = {key: getattr(opt, key)
                        for key in options_keys
                        if hasattr(opt, key)}
        self._prepare_normalize_func(state0)


"""
