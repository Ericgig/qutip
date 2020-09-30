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
from ..ui.progressbar import get_progess_bar

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

    def restore_state(self, state):
        return Qobj(state,
                    dims=self.state_dims,
                    type=self.state_type,
                    copy=False)

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
        return self.restore_state(self.state)

    def _driver_step(self, tlist, state0):
        """
        Internal function for solving ODEs.
        """
        progress_bar = get_progess_bar(self.options['progress_bar'])

        self.evolver.set(state0, tlist[0])
        e_ops = self.evolver.e_op_prepare(self.e_ops)
        res = Result(self.e_ops, e_ops, self.options.results,
                     self.state_qobj, self.super)
        res.add(tlist[0], self.state_qobj)

        progress_bar.start(len(tlist)-1, **self.options['progress_kwargs'])
        for t, state in self.evolver.run(tlist):
            progress_bar.update()
            res.add(t, self.restore_state(state))
        progress_bar.finished()

        return res

    def _driver_evolution(self, tlist, state0):
        """ Internal function for solving ODEs. """
        progress_bar = get_progess_bar(options['progress_bar'])

        res = Result(e_ops, options.results, state0, super)

        progress_bar.start(len(tlist)-1, **options['progress_kwargs'])
        states = evolver.evolve(state0, tlist, progress_bar)
        progress_bar.finished()

        for t, state in zip(tlist, states):
            res.add(t, self.restore_state(state))

        return res

    def get_evolver(self, options, args, feedback_args):
        return get_evolver(self.system, options, args, feedback_args)
