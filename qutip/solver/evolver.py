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
__all__ = ['EvolverScipyZvode', 'EvolverScipyDop853',
           'EvolverVern7', 'get_evolver']


import numpy as np
from numpy.linalg import norm as la_norm
from scipy.integrate import ode
from qutip.core import data as _data
from qutip.solver._solverqevo import SolverQEvo
from qutip.solver.ode.verner7efficientrk import vern7
from qutip.solver.ode.wrapper import QtOdeFuncWrapperSolverQEvo


all_ode_method = ['adams', 'bdf', 'dop853', 'vern7']

def get_evolver(system, options, args, feedback_args):
    if options['method'] in ['adams','bdf']:
        return EvolverScipyZvode(system, options, args, feedback_args)
    elif options['method'] in ['dop853']:
        return EvolverScipyDop853(system, options, args, feedback_args)
    elif options['method'] in ['vern7']:
        return EvolverVern7(system, options, args, feedback_args)


class Evolver:
    """
    Methods
    -------
    set(state, tlist)
    step()
        Create copy of Qobj
    evolve(state, tlist, progress_bar)

    """
    def __init__(self, system, options, args, feedback_args):
        self.system = SolverQEvo(system, options, args, feedback_args)
        self.options = options
        self.name = "undefined"
        self._error_msg = ("ODE integration error: Try to increase "
                           "the allowed number of substeps by increasing "
                           "the nsteps parameter in the Options class.")
        self._normalize = False

    def set_shape(self, example_state):
        if example_state.shape[1] > 1:
            self._mat_state = True
            self._size = example_state.shape[0]
            if abs(example_state.trace() - 1) > self.options['rtol']:
                self._oper = True
            else:
                self._dm = True
        else:
            self._mat_state = False

    def set(self, state, t0):
        pass

    def update_args(self, args):
        self.system.arguments(args)

    def normalize(self, state):
        # TODO cannot be used for propagator evolution.
        if self._oper:
            return 0
        if self._dm:
            norm = _data.trace(state)
        else:
            norm = _data.la_norm(state)
        state /= norms
        return abs(norm-1)

    def step(self, t):
        self._r.integrate(t)
        if not self._r.successful():
            raise Exception(self._error_msg)
        state = self.get_state()
        if self._normalize:
            if self.normalize(state):
                self.set_state(state, t)
        return state

    def run(self, tlist):
        for t in tlist[1:]:
            self._r.integrate(t)
            if not self._r.successful():
                raise Exception(self._error_msg)
            state = self.get_state()
            if self._normalize:
                if self.normalize(state):
                    self.set_state(state, t)
            yield t, state

    def e_op_prepare(self, e_ops):
        return e_ops


class EvolverScipyZvode(Evolver):
    # -------------------------------------------------------------------------
    # Solve an ODE for func.
    # Calculate the required expectation values or invoke callback
    # function at each time step.
    #
    # Use scipy's Ode, with zvode solver
    #
    name = "scipy_zvode"

    def set(self, state0, t0):
        self.set_shape(state0)
        self._t = t0

        r = ode(self.system.mul_np_vec)
        options_keys = ['atol', 'rtol', 'nsteps', 'method', 'order',
                        'first_step', 'max_step', 'min_step']
        options = {key: self.options[key]
                   for key in options_keys
                   if key in self.options}
        r.set_integrator('zvode', **options)
        self._r = r

        self.set_state(state0, t0)

    def get_state(self):
        if self._mat_state:
            return _data.column_unstack_dense(
                _data.dense.fast_from_numpy(self._r.y),
                self._size,
                inplace=True)
        else:
            return _data.dense.fast_from_numpy(self._r.y)

    def set_state(self, state0, t):
        self._r.set_initial_value(
            _data.column_stack(state0).to_array().ravel(),
            t
        )


class EvolverScipyDop853(Evolver):
    # -------------------------------------------------------------------------
    # Solve an ODE for func.
    # Calculate the required expectation values or invoke callback
    # function at each time step.
    #
    # Use scipy's Ode, with dop853 solver
    #
    name = "scipy_dop853"

    def set(self, state0, t0):
        self.set_shape(state0)
        func = self.system.mul_np_vec
        r = ode(self.funcwithfloat)
        options_keys = ['atol', 'rtol', 'nsteps', 'first_step', 'max_step',
                        'ifactor', 'dfactor', 'beta']
        options = {key: self.options[key]
                   for key in options_keys
                   if key in self.options}
        r.set_integrator('dop853', **options)
        self._r = r
        self.set_state(state0, t0)

    def funcwithfloat(self, t, y):
        y_cplx = y.view(complex)
        dy = self.system.mul_np_vec(t, y_cplx)
        return dy.view(np.float64)

    def get_state(self):
        if self._mat_state:
            return _data.column_unstack_dense(
                _data.dense.fast_from_numpy(self._r.y.view(np.complex)),
                self._size,
                inplace=True)
        else:
            return _data.dense.fast_from_numpy(self._r.y.view(np.complex))

    def set_state(self, state0, t):
        self._r.set_initial_value(
            _data.column_stack(state0).to_array().ravel().view(np.float64),
            t
        )


class EvolverVern7(Evolver):
    # -------------------------------------------------------------------------
    # Solve an ODE for func.
    # Calculate the required expectation values or invoke callback
    # function at each time step.
    #
    # Use scipy's Ode, with dop853 solver
    #
    name = "qutip_vern7"

    def set(self, state0, t0):
        self.set_shape(state0)
        func = QtOdeFuncWrapperSolverQEvo(self.system)
        options_keys = ['atol', 'rtol', 'nsteps', 'first_step', 'max_step',
                        'min_step', 'interpolate']
        options = {key: self.options[key]
                   for key in options_keys
                   if key in self.options}
        r = vern7(func, **options)
        self._r = r
        self.set_state(state0, t0)

    def get_state(self):
        return self._r.y.copy()

    def set_state(self, state, t):
        self._r.set_initial_value(_data.to(_data.Dense, state).copy(), t)
