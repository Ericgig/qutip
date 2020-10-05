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
from qutip.solver.ode.verner7efficient import vern7
from qutip.solver.ode.verner9efficient import vern9
from qutip.solver.ode.wrapper import QtOdeFuncWrapperSolverQEvo


all_ode_method = ['adams', 'bdf', 'dop853', 'vern7', 'vern9']


def get_evolver(system, options, args, feedback_args):
    if options['method'] in ['adams','bdf']:
        return EvolverScipyZvode(system, options, args, feedback_args)
    elif options['method'] in ['dop853']:
        return EvolverScipyDop853(system, options, args, feedback_args)
    elif options['method'] in ['vern7', 'vern9']:
        return EvolverVern(system, options, args, feedback_args)
    elif options['method'] in ['diagonalized', 'diag']:
        return EvolverDiag(system, options, args, feedback_args)
    raise ValueError("method options not recognized")


class Evolver:
    """ A wrapper around ODE solvers.
    Ensure a common interface for Solver usage.
    Take and return states as :class:`qutip.core.data.Data`.

    Methods
    -------
    set(state, t0, options)
        Prepare the ODE solver.

    step(t)
        Evolve to t, must be `set` before.

    run(state, tlist)
        Yield (t, state(t)) for t in tlist, must be `set` before.

    update_args(args)
        Change the argument of the active system

    get_state()
        Optain the state of the solver.

    set_state(state, t)
        Set the state of an existing ODE solver.

    """
    def __init__(self, system, options, args, feedback_args):
        self.system = SolverQEvo(system, options, args, feedback_args)
        self.options = options
        self.name = "undefined"
        self._error_msg = ("ODE integration error: Try to increase "
                           "the allowed number of substeps by increasing "
                           "the nsteps parameter in the Options class.")
        self._ode_solver = None

    def _set_shape(self, example_state):
        if example_state.shape[1] > 1:
            self._mat_state = True
            self._size = example_state.shape[0]
        else:
            self._mat_state = False

    def set(self, state, t0, set_shape):
        pass

    def update_args(self, args):
        self.system.arguments(args)

    def step(self, t):
        """ Evolve to t, must be `set` before. """
        self._ode_solver.integrate(t)
        if not self._ode_solver.successful():
            raise Exception(self._error_msg)
        state = self.get_state()
        return state

    def run(self, tlist):
        """ Yield (t, state(t)) for t in tlist, must be `set` before. """
        for t in tlist[1:]:
            self._ode_solver.integrate(t)
            if not self._ode_solver.successful():
                raise Exception(self._error_msg)
            state = self.get_state()
            yield t, state

    def e_op_prepare(self, e_ops):
        return e_ops

    def get_state(self):
        pass

    def set_state(self, state0, t):
        pass


class EvolverScipyZvode(Evolver):
    # -------------------------------------------------------------------------
    # Solve an ODE for func.
    # Calculate the required expectation values or invoke callback
    # function at each time step.
    #
    # Use scipy's Ode, with zvode solver
    #
    name = "scipy_zvode"

    def set(self, state0, t0, options):
        self.options = options
        self._set_shape(state0)
        self._t = t0

        r = ode(self.system.mul_np_vec)
        options_keys = ['atol', 'rtol', 'nsteps', 'method', 'order',
                        'first_step', 'max_step', 'min_step']
        options = {key: options[key]
                   for key in options_keys
                   if key in options}
        r.set_integrator('zvode', **options)
        self._ode_solver = r

        self.set_state(state0, t0)

    def get_state(self):
        if self._mat_state:
            return _data.column_unstack_dense(
                _data.dense.fast_from_numpy(self._ode_solver._y),
                self._size,
                inplace=True)
        else:
            return _data.dense.fast_from_numpy(self._ode_solver._y)

    def set_state(self, state0, t):
        self._ode_solver.set_initial_value(
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

    def set(self, state0, t0, options):
        self.options = options
        self._set_shape(state0)
        func = self.system.mul_np_vec
        r = ode(self.funcwithfloat)
        options_keys = ['atol', 'rtol', 'nsteps', 'first_step', 'max_step',
                        'ifactor', 'dfactor', 'beta']
        options = {key: options[key]
                   for key in options_keys
                   if key in options}
        r.set_integrator('dop853', **options)
        self._ode_solver = r
        self.set_state(state0, t0)

    def funcwithfloat(self, t, y):
        y_cplx = y.view(complex)
        dy = self.system.mul_np_vec(t, y_cplx)
        return dy.view(np.float64)

    def get_state(self):
        if self._mat_state:
            return _data.column_unstack_dense(
                _data.dense.fast_from_numpy(self._ode_solver.
                                            _y.view(np.complex)),
                self._size,
                inplace=True)
        else:
            return _data.dense.fast_from_numpy(self._ode_solver.
                                               _y.view(np.complex))

    def set_state(self, state0, t):
        self._ode_solver.set_initial_value(
            _data.column_stack(state0).to_array().ravel().view(np.float64),
            t
        )


class EvolverVern(Evolver):
    # -------------------------------------------------------------------------
    # Solve an ODE for func.
    # Calculate the required expectation values or invoke callback
    # function at each time step.
    #
    # Use verner method implimented in cython
    #
    name = "qutip_"

    def set(self, state0, t0, options):
        self.options = options
        self._set_shape(state0)
        func = QtOdeFuncWrapperSolverQEvo(self.system)
        options_keys = ['atol', 'rtol', 'nsteps', 'first_step', 'max_step',
                        'min_step', 'interpolate']
        options = {key: options[key]
                   for key in options_keys
                   if key in options}
        ode = vern7 if options.method == 'vern7' else vern9
        self.name += options.method
        self._ode_solver = ode(func, **options)
        self.set_state(state0, t0)

    def get_state(self):
        return self._ode_solver.y

    def set_state(self, state, t):
        self._ode_solver.set_initial_value(_data.to(_data.Dense, state).copy(),
                                           t)


class EvolverDiag(Evolver):
    # -------------------------------------------------------------------------
    # Solve an ODE for func.
    # Calculate the required expectation values or invoke callback
    # function at each time step.
    #
    # Diagonalize the Hamiltonian and
    # This should be used for constant Hamiltonian evolution (sesolve, mcsolve)
    #
    name = "qutip_diagonalized"

    def __init__(self, system, options, args, feedback_args):
        if not system.const:
            raise ValueError("Hamiltonian system must be constant to use "
                             "diagonalized method")
        self.system = system
        self.U, self.diag = system(0).eigenstates()
        self._dt
        self.Ud = self.U.T.conj()
        self.options = options

    def set(self, state0, t0, options):
        self.options = options
        self._set_shape(state0)
        self._t = t0
        self._y = self.Ud @ state0.as_array()
        self.set_state(state0, t0)

    def step(self, t):
        """ Evolve to t, must be `set` before. """
        dt = self._t - t
        if self.dt != dt:
            self.expH = np.exp(self.diag * dt)
            self.dt = dt
        self._y *= self.expH
        self._t = t
        return self.get_state()

    def run(self, tlist):
        """ Yield (t, state(t)) for t in tlist, must be `set` before. """
        for t in tlist[1:]:
            state = self.step(t)
            yield t, state

    def get_state(self):
        y = self.U @ self._y
        if self._mat_state:
            return _data.column_unstack_dense(
                _data.dense.fast_from_numpy(y),
                self._size,
                inplace=True)
        else:
            return _data.dense.fast_from_numpy(y)

    def set_state(self, state0, t):
        self._y = self.Ud @ state0.as_array()
