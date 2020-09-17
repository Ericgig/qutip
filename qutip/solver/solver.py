# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson,
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
from __future__ import print_function

__all__ = ['SolverOptions',
           'SolverResultsOptions',
           'McOptions']

import os
import sys
import warnings
import datetime
import numpy as np
from collections import OrderedDict
from types import FunctionType, BuiltinFunctionType

from .. import __version__, Qobj, QobjEvo
from ..optionsclass import optionsclass
from ..core import data as _data
from .run import Run
from .evolver import *
from ..ui.progressbar import get_progess_bar

class Result:
    pass

class config:
    pass

def _solver_safety_check(*args, **kwargs):
    return None

class ExpectOps:
    pass

class SolverSystem:
    pass

class solver_safe:
    pass


class Solver:
    def __init__(self):
        self.system = None
        self._safe_mode = False
        self.evolver = None
        self.options = None
        self.e_ops = []
        self.super = False
        self.state = None
        self.t = 0

    def safety_check(self, state):
        pass

    def prepare_state(self, state):
        self.state_dims = state.dims
        self.state_type = state.type
        self.state_qobj = state
        return state.data

    def run(self, state0, tlist, args={}):
        if self._safe_mode:
            self.safety_check(state0)
        state0 = self.prepare_state(state0)
        if args:
            self.evolver.update_args(args)
        result = self._driver_step(tlist, state0)
        return result

    def start(self, state0, t0):
        self.state = self.prepare_state(state0)
        self.t = t0
        self.evolver.set(self.state, self.t)

    def step(self, t, args={}):
        if args:
            self.evolver.update_args(args)
            self.evolver.set(self.state, self.t)
        self.state = self.evolver.step(t)
        self.t = t
        return Qobj(self.state,
                    dims=self.state_dims,
                    type=self.state_type)

    def _driver_step(self, tlist, state0):
        """
        Internal function for solving ODEs.
        """
        progress_bar = get_progess_bar(self.options['progress_bar'])

        self.evolver.set(state0, tlist[0])
        e_ops = self.evolver.e_op_prepare(self.e_ops)
        res = Run(self.e_ops, e_ops, self.options.results,
                  self.state_qobj, self.super)
        res.add(tlist[0], state0)

        progress_bar.start(len(tlist)-1, **self.options['progress_kwargs'])
        for t, state in self.evolver.run(tlist):
            progress_bar.update()
            res.add(t, state)
        progress_bar.finished()

        return res

    def driver(evolver, tlist, state0):
        pass

    def _driver_evolution(self, tlist, state0):
        """ Internal function for solving ODEs. """
        progress_bar = get_progess_bar(options['progress_bar'])

        res = Run(e_ops, options.results, state0, super)

        progress_bar.start(len(tlist)-1, **options['progress_kwargs'])
        states = evolver.evolve(state0, tlist, progress_bar)
        progress_bar.finished()

        for t, state in zip(tlist, states):
            res.add(t, state)

        return res

    def get_evolver(self, options, args, feedback_args):
        if options['method'] in ['adams','bdf']:
            return EvolverScipyZvode(self.system, options, args, feedback_args)
        elif options['method'] in ['dop853']:
            return EvolverScipyDop853(self.system, options, args, feedback_args)
        elif options['method'] in ['vern7']:
            return EvolverVern7(self.system, options, args, feedback_args)


@optionsclass("solver")
class SolverOptions:
    """
    Class of options for evolution solvers such as :func:`qutip.mesolve` and
    :func:`qutip.mcsolve`. Options can be specified either as arguments to the
    constructor::

        opts = SolverOptions(order=10, ...)

    or by changing the class attributes after creation::

        opts = SolverOptions()
        opts.order = 10

    Returns options class to be used as options in evolution solvers.

    The default can be changed by::

        qutip.settings.solver['order'] = 10

    Options
    -------

    atol : float {1e-8}
        Absolute tolerance.
    rtol : float {1e-6}
        Relative tolerance.
    method : str {'adams','bdf'}
        Integration method.
    order : int {12}
        Order of integrator (<=12 'adams', <=5 'bdf')
    nsteps : int {2500}
        Max. number of internal steps/call.
    first_step : float {0}
        Size of initial step (0 = automatic).
    min_step : float {0}
        Minimum step size (0 = automatic).
    max_step : float {0}
        Maximum step size (0 = automatic)
    tidy : bool {True,False}
        Tidyup Hamiltonian and initial state by removing small terms.
    average_states : bool {False}
        Average states values over trajectories in stochastic solvers.
    average_expect : bool {True}
        Average expectation values over trajectories for stochastic solvers.
    ntraj : int {500}
        Number of trajectories in stochastic solvers.
    store_final_state : bool {False, True}
        Whether or not to store the final state of the evolution in the
        result class.
    store_states : bool {False, True}
        Whether or not to store the state vectors or density matrices in the
        result class, even if expectation values operators are given. If no
        expectation are provided, then states are stored by default and this
        option has no effect.
    """
    options = {
        # Absolute tolerance (default = 1e-8)
        "atol": 1e-8,
        # Relative tolerance (default = 1e-6)
        "rtol": 1e-6,
        # Integration method (default = 'adams', for stiff 'bdf')
        "method": 'adams',
        # Maximum order used by integrator (<=12 for 'adams', <=5 for 'bdf')
        "order": 12,
        # Max. number of internal steps/call
        "nsteps": 1000,
        # Size of initial step (0 = determined by solver)
        "first_step": 0,
        # Max step size (0 = determined by solver)
        "max_step": 0,
        # Minimal step size (0 = determined by solver)
        "min_step": 0,
        # tidyup Hamiltonian before calculation (default = True)
        "tidy": True,
        # Number of trajectories (default = 500)
        "ntraj": 500,
        "gui": False,
        # Normalize output of solvers
        # (turned off for batch unitary propagator mode)
        "normalize_output": True,
        # Normalize output of solvers
        # (turned off for batch unitary propagator mode)
        "progress_bar": "text",
        # Normalize output of solvers
        # (turned off for batch unitary propagator mode)
        "progress_kwargs": {"chunk_size":10},
    }

@optionsclass("results", SolverOptions)
class SolverResultsOptions:
    """
    Class of options for evolution solvers such as :func:`qutip.mesolve` and
    :func:`qutip.mcsolve`. Options can be specified either as arguments to the
    constructor::

        opts = SolverOptions(order=10, ...)

    or by changing the class attributes after creation::

        opts = SolverOptions()
        opts.order = 10

    Returns options class to be used as options in evolution solvers.

    The default can be changed by::

        qutip.settings.solver['order'] = 10

    Options
    -------

    atol : float {1e-8}
        Absolute tolerance.
    rtol : float {1e-6}
        Relative tolerance.
    method : str {'adams','bdf'}
        Integration method.
    order : int {12}
        Order of integrator (<=12 'adams', <=5 'bdf')
    nsteps : int {2500}
        Max. number of internal steps/call.
    first_step : float {0}
        Size of initial step (0 = automatic).
    min_step : float {0}
        Minimum step size (0 = automatic).
    max_step : float {0}
        Maximum step size (0 = automatic)
    tidy : bool {True,False}
        Tidyup Hamiltonian and initial state by removing small terms.
    average_states : bool {False}
        Average states values over trajectories in stochastic solvers.
    average_expect : bool {True}
        Average expectation values over trajectories for stochastic solvers.
    ntraj : int {500}
        Number of trajectories in stochastic solvers.
    store_final_state : bool {False, True}
        Whether or not to store the final state of the evolution in the
        result class.
    store_states : bool {False, True}
        Whether or not to store the state vectors or density matrices in the
        result class, even if expectation values operators are given. If no
        expectation are provided, then states are stored by default and this
        option has no effect.
    """
    options = {
        # Average expectation values over trajectories (default = True)
        "average_expect": True,
        # average expectation values
        "average_states": False,
        # store final state?
        "store_final_state": False,
        # store states even if expectation operators are given?
        "store_states": False,
        # average mcsolver density matricies assuming steady state evolution
        "steady_state_average": False,
    }

@optionsclass("mcsolve", SolverOptions)
class McOptions:
    """
    Class of options for evolution solvers such as :func:`qutip.mesolve` and
    :func:`qutip.mcsolve`. Options can be specified either as arguments to the
    constructor::

        opts = SolverOptions(norm_tol=1e-3, ...)

    or by changing the class attributes after creation::

        opts = SolverOptions()
        opts.['norm_tol'] = 1e-3

    Returns options class to be used as options in evolution solvers.

    The default can be changed by::

        qutip.settings.options.montecarlo['norm_tol'] = 1e-3

    Options
    -------

    norm_tol : float {1e-4}
        Tolerance used when finding wavefunction norm in mcsolve.
    norm_t_tol : float {1e-6}
        Tolerance used when finding wavefunction time in mcsolve.
    norm_steps : int {5}
        Max. number of steps used to find wavefunction norm to within norm_tol
        in mcsolve.
    mc_corr_eps : float {1e-10}
        Arbitrarily small value for eliminating any divide-by-zero errors in
        correlation calculations when using mcsolve.
    """
    options = {
        # Tolerance for wavefunction norm (mcsolve only)
        "norm_tol": 1e-4,
        # Tolerance for collapse time precision (mcsolve only)
        "norm_t_tol": 1e-6,
        # Max. number of steps taken to find wavefunction norm to within
        # norm_tol (mcsolve only)
        "norm_steps": 5,
        # small value in mc solver for computing correlations
        "mc_corr_eps": 1e-10,
    }
