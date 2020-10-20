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

# import numpy as np
# from ..core import data as _data

from .. import Qobj
from .result import Result
from .evolver import *
from .AHS.evolver import AHSEvolver
from ..ui.progressbar import get_progess_bar
from ..core.data import to

class Solver:
    """
    Main class of the solvers.
    Do the loop over each times in tlist and does the interface between the
    evolver which deal in data and the Result which use Qobj.
    It's children (SeSolver, McSolver) are responsible with building the system
    (-1j*H).

    methods
    -------
    run(state0, tlist, args={}):
        Do an evolution starting with `state0` at `tlist[0]` and obtain a
        result for each time in `tlist`.
        The system's arguments can be changed with `args`.

    start(state0, t0):
        Set the initial values for an evolution by steps

    step(t, args={}):
        Do a step to `t`. The system arguments for this step can be updated
        with `args`

    attributes
    ----------
    options : SolverOptions
        Options for the solver

    e_ops : list
        list of Qobj or QobjEvo to compute the expectation values.
        Alternatively, function[s] with the signature f(t, state) -> expect
        can be used.

    """
    def __init__(self):
        self._system = None
        self._safe_mode = False
        self._evolver = None
        self._super = False
        self._state = None
        self._t = 0

        self.options = None
        self.e_ops = []

    def _safety_check(self, state):
        pass

    def _prepare_state(self, state):
        self._state_dims = state.dims
        self._state_type = state.type
        self._state_qobj = state
        str_to_type = {layer.__name__.lower(): layer for layer in to.dtypes}
        if self.options["State_data_type"].lower() in str_to_type:
            state = state.to(str_to_type[self.options["State_data_type"].lower()])
        return state.data

    def _restore_state(self, state):
        return Qobj(state,
                    dims=self._state_dims,
                    type=self._state_type,
                    copy=False)

    def run(self, state0, tlist, args={}):
        if self._safe_mode:
            self._safety_check(state0)
        state0 = self._prepare_state(state0)
        if args:
            self._evolver.update_args(args)
        result = self._driver_step(tlist, state0)
        return result

    def start(self, state0, t0):
        self._state = self._prepare_state(state0)
        self._t = t0
        self._evolver.set(self._state, self._t, self.options)

    def step(self, t, args={}):
        if args:
            self._evolver.update_args(args)
            self._evolver.set(self._state, self._t, self.options)
        self._state = self._evolver.step(t)
        self._t = t
        return self._restore_state(self._state.copy())

    def _driver_step(self, tlist, state0):
        """
        Internal function for solving ODEs.
        """
        progress_bar = get_progess_bar(self.options['progress_bar'])

        self._evolver.set(state0, tlist[0], self.options)
        res = Result(self.e_ops, self.options.results, self._super)
        res.add(tlist[0], self._state_qobj)

        progress_bar.start(len(tlist)-1, **self.options['progress_kwargs'])
        for t, state in self._evolver.run(tlist):
            progress_bar.update()
            res.add(t, self._restore_state(state))
        progress_bar.finished()

        return res

    def _get_evolver(self, options, args, feedback_args):
        str_to_type = {layer.__name__.lower(): layer for layer in to.dtypes}
        if self.options["Operator_data_type"].lower() in str_to_type:
            self._system = self._system.to(str_to_type[self.options["Operator_data_type"].lower()])
        if options["ahs"]:
            return AHSEvolver(self._system, options, args, feedback_args)
        return get_evolver(self._system, options, args, feedback_args)
