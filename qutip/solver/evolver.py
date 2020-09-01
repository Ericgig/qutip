# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
"""
This module provides solvers for
"""

import numpy as np
from numpy.linalg import norm as la_norm
from scipy.integrate import solve_ivp, ode
from qutip.core import data as _data

#from qutip.qobjevo_maker import is_dynargs_pattern
#from qutip.cy.spmatfuncs import normalize_inplace, normalize_op_inplace
#from qutip.qobj import Qobj
"""
def normalize_prop(state):
    st = st.resize(())
    norms = la_norm(state, axis=0)
    state /= norms
    return np.mean(norms)


def dummy_normalize(state):
    return 0


normalize_dm = dummy_normalize
"""


class Evolver:
    """
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
        super(EvolverScipyZvode, self).__init__(system, options)
        self.name = "scipy_zvode"
        self._normalize = False

    def set(self, state0, tlist):
        self.tlist = tlist
        self._i = 0
        if state0.shape[1] > 1:
            self._mat_state = True
            self._size = state0.shape[0]
            if abs(state0.tr() - 1) > self.options['rtol']:
                self._oper = True
            else:
                self._dm = True
        else:
            self._mat_state = False

        r = ode(self.system.mul_np_vec)
        options_keys = ['atol', 'rtol', 'nsteps', 'method', 'order',
                        'first_step', 'max_step', 'min_step']
        options = {key: self.options[key] for key in options_keys}
        r.set_integrator('zvode', **options)

        r.set_initial_value(_data.column_stack(state0.data).to_array().ravel('F'), tlist)
        self._r = r

    def step(self, tlist):
        for t in tlist[1:]:
            self._r.integrate(t)
            if not self._r.successful():
                raise Exception(self._error_msg)
            state = self.get_state()
            if self._normalize:
                if self.normalize(state):
                    self._r.set_initial_value(state.as_ndarray(), t)
            yield t, state

    def evolve(self, state, tlist, progress_bar):
        self.set(state, tlist[0])
        states = [state.copy()]
        for t in tlist[1:]:
            self._r.integrate(t)
            if not self._r.successful():
                raise Exception(self._error_msg)
            progress_bar.update()
            state = self.get_state()
            if self._normalize:
                if self.normalize(state) >= self.options["renorm_tol"]:
                    self._r.set_initial_value(state.as_ndarray(), t)
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
        """
        Internal function for solving ODEs.
        """
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






# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# TODO move in data layer?
"""
    def _prepare_normalize_func(self, state0):
        opt = self.options
        size = np.prod(state0.shape)
        if opt.normalize_output and size == self.LH.shape[1]:
            if self.LH.cte.issuper:
                self.normalize_func = normalize_dm
            else:
                self.normalize_func = normalize_inplace
        elif opt.normalize_output and size == np.prod(self.LH.shape):
            self.normalize_func = normalize_op_inplace
        elif opt.normalize_output:
            self.normalize_func = normalize_mixed(state0.shape)
"""
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
