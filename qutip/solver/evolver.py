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
           'EvolverVern', 'EvolverDiag', 'get_evolver']


import numpy as np
from numpy.linalg import norm as la_norm
from scipy.integrate import ode
from scipy.integrate._ode import zvode
from ..core import data as _data
from ._solverqevo import SolverQEvo
from .ode.verner7efficient import vern7
from .ode.verner9efficient import vern9
from .ode.wrapper import QtOdeFuncWrapperSolverQEvo
import warnings

all_ode_method = ['adams', 'bdf', 'dop853', 'vern7', 'vern9']

class qutip_zvode(zvode):
    def step(self, *args):
        itask = self.call_args[2]
        self.rwork[0] = args[4]
        self.call_args[2] = 5
        r = self.run(*args)
        self.call_args[2] = itask
        return r

def get_evolver(system, options, args, feedback_args):
    if options.ode['method'] in ['adams','bdf']:
        return EvolverScipyZvode(system, options, args, feedback_args)
    elif options.ode['method'] in ['dop853']:
        return EvolverScipyDop853(system, options, args, feedback_args)
    elif options.ode['method'] in ['vern7', 'vern9']:
        return EvolverVern(system, options, args, feedback_args)
    elif options.ode['method'] in ['diagonalized', 'diag']:
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
    name = "undefined"

    def __init__(self, system, options, args, feedback_args):
        self.system = SolverQEvo(system, options.rhs, args, feedback_args)
        self.options = options.ode
        self._error_msg = ("ODE integration error: Try to increase "
                           "the allowed number of substeps by increasing "
                           "the nsteps parameter in the Options class.")
        self._ode_solver = None
        self._previous_call = 0

    def _set_shape(self, example_state):
        if example_state.shape[1] > 1:
            self._mat_state = True
            self._size = example_state.shape[0]
        else:
            self._mat_state = False

    def set(self, state, t0, options=None):
        pass

    def update_args(self, args):
        self.system.arguments(args)

    def step(self, t, step=False):
        """ Evolve to t, must be `set` before. """
        self._ode_solver.integrate(t, step=step)
        if not self._ode_solver.successful():
            raise Exception(self._error_msg)
        return self.get_state()

    def run(self, tlist):
        """ Yield (t, state(t)) for t in tlist, must be `set` before. """
        for t in tlist[1:]:
            self._ode_solver.integrate(t)
            if not self._ode_solver.successful():
                raise Exception(self._error_msg)
            state = self.get_state()
            yield t, state

    def get_state(self):
        pass

    def set_state(self, state0, t):
        pass

    @property
    def t(self):
        return self._ode_solver.t

    @property
    def solver_call(self):
        return self.system.func_call - self._previous_call


class EvolverScipyZvode(Evolver):
    # -------------------------------------------------------------------------
    # Solve an ODE for func.
    # Calculate the required expectation values or invoke callback
    # function at each time step.
    #
    # Use scipy's Ode, with zvode solver
    #
    name = "scipy_zvode"

    def set(self, state0, t0, options=None):
        if options is not None:
            self.options = options.ode
        self._set_shape(state0)
        self._t = t0
        self._y = state0.copy()

        r = ode(self.system.mul_np_vec)
        options_keys = ['atol', 'rtol', 'nsteps', 'method', 'order',
                        'first_step', 'max_step', 'min_step']
        opt = {key: self.options[key]
               for key in options_keys
               if key in self.options}
        r.set_integrator('zvode', **opt)
        self._ode_solver = r
        self.set_state(state0, t0)
        self.name = "scipy zvode " + opt["method"]
        self._previous_call = self.system.func_call

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

    def step(self, t, step=None):
        """ Evolve to t, must be `set` before. """
        if step:
            # integrate(t, step=True) ignore the time and advance one step.
            # Here we want to advance to t doing maximum one step.
            # So we check if a new step is really needed.
            t_front = self._ode_solver._integrator.rwork[12]
            t_ode = self._ode_solver.t
            # print(t, t_front, t_ode)
            if t > t_front and t_front <= t_ode:
                self._ode_solver.integrate(t, step=True)
                t_front = self._ode_solver._integrator.rwork[12]
            elif t > t_front:
                t = t_front
            if t_front >= t:
                self._ode_solver.integrate(t)
            if not self._ode_solver.successful():
                raise Exception(self._error_msg)
            return self.get_state()

        if self._ode_solver.t != t:
            self._ode_solver.integrate(t)
        if not self._ode_solver.successful():
            raise Exception(self._error_msg)
        return self.get_state()

    def backstep(self, t, t_old, y_old):
        """ Evolve to t, must be `set` before. """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self._ode_solver.integrate(t)
        if not self._ode_solver.successful():
            # print("caught", self._ode_solver.t, t,
            #       self._ode_solver.y, self._ode_solver._integrator.call_args)
            self.set_state(y_old, t_old)
            self._ode_solver.integrate(t)
        return self.get_state()


class EvolverScipyDop853(Evolver):
    # -------------------------------------------------------------------------
    # Solve an ODE for func.
    # Calculate the required expectation values or invoke callback
    # function at each time step.
    #
    # Use scipy's Ode, with dop853 solver
    #
    name = "scipy_dop853"

    def set(self, state0, t0, options=None):
        if options is not None:
            self.options = options.ode
        self._t = t0
        self._y = state0.copy()

        self._set_shape(state0)
        r = ode(self.funcwithfloat)
        options_keys = ['atol', 'rtol', 'nsteps', 'first_step', 'max_step',
                        'ifactor', 'dfactor', 'beta']
        opt = {key: self.options[key]
               for key in options_keys
               if key in self.options}
        r.set_integrator('dop853', **opt)
        self._ode_solver = r
        self.set_state(state0, t0)
        self.name = "scipy dop853"
        self._previous_call = self.system.func_call

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

    def step(self, t, step=None):
        """ Evolve to t, must be `set` before. """
        dt_max = self._ode_solver._integrator.work[6] # allowed max timestep
        dt = t - self._ode_solver.t
        if dt_max * dt < 0: # chande in direction
            self._ode_solver._integrator.reset(len(self._ode_solver._y), False)
            dt_max = -dt_max
        elif dt_max == 0:
            dt_max = 0.01 * dt
        if step:
            # Will probably do more work than strickly one step if cought in
            # one of the previous conditions, making collapse finding for
            # mcsolve not ideal.
            t = self._ode_solver.t + min(dt_max, dt) if dt > 0 else max(dt_max, dt)
        self._ode_solver.integrate(t)
        return self.get_state()

    def backstep(self, t, t_old, y_old):
        """ Evolve to t, must be `set` before. """
        self._ode_solver._integrator.reset(len(self._ode_solver._y), False)
        self._ode_solver.integrate(t)
        return self.get_state()


class EvolverVern(Evolver):
    # -------------------------------------------------------------------------
    # Solve an ODE for func.
    # Calculate the required expectation values or invoke callback
    # function at each time step.
    #
    # Use verner method implimented in cython
    #
    name = "qutip "

    def set(self, state0, t0, options=None):
        if options is not None:
            self.options = options.ode
        self._set_shape(state0)
        self._t = t0
        self._y = state0.copy()
        func = QtOdeFuncWrapperSolverQEvo(self.system)
        options_keys = ['atol', 'rtol', 'nsteps', 'first_step', 'max_step',
                        'min_step', 'interpolate']
        opt = {key: self.options[key]
               for key in options_keys
               if key in self.options}
        ode = vern7 if self.options['method'] == 'vern7' else vern9
        self._ode_solver = ode(func, **opt)
        self.set_state(state0, t0)
        self.name = "qutip " + self.options['method']
        self._previous_call = self.system.func_call

    def get_state(self):
        return self._ode_solver.y

    def set_state(self, state, t):
        self._ode_solver.set_initial_value(
            _data.to(_data.Dense, state),
            t
        )

    def backstep(self, t, t_old, y_old):
        self._ode_solver.integrate(t)
        return self.get_state()


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
        self.diag, self.U = system(0).eigenstates()
        self.U = np.hstack([eket.full() for eket in self.U])
        self.diag = self.diag.reshape((-1,1))
        self._dt = 0.
        self._expH = None
        self.Uinv = np.linalg.inv(self.U)
        self.options = options
        self.name = "qutip diagonalized"

    def set(self, state0, t0, options=None):
        if options is not None:
            self.options = options.ode
        self._set_shape(state0)
        self.set_state(state0, t0)

    def step(self, t, args=None, step=False):
        """ Evolve to t, must be `set` before. """
        dt = t - self._t
        if dt == 0:
            return self.get_state()
        elif self._dt != dt:
            self._expH = np.exp(self.diag * dt)
            self._dt = dt
        self._y *= self._expH
        self._t = t
        return self.get_state()

    def run(self, tlist):
        """ Yield (t, state(t)) for t in tlist, must be `set` before. """
        for t in tlist[1:]:
            state = self.step(t)
            yield t, state

    def get_state(self):
        y = self.U @ self._y
        return _data.dense.fast_from_numpy(y)

    def set_state(self, state0, t):
        self._t = t
        self._y = (self.Uinv @ state0.to_array())

    @property
    def t(self):
        return self._t

    def backstep(self, t, t_old, y_old):
        return self.step(t)

    @property
    def solver_call(self):
        return 1
