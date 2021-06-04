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

__all__ = ['Solver']

# import numpy as np
# from ..core import data as _data

from .. import Qobj, QobjEvo, QobjEvoFunc
from qutip.core.qobjevo import QobjEvoBase
from .result import Result
from .integrator import integrator_collection
from ..ui.progressbar import get_progess_bar
from ..core.data import to
from time import time


class Solver:
    """
    Main class of the solvers.
    Do the loop over each times in tlist and does the interface between the
    evolver which deal in data and the Result which use Qobj.
    It's children (SeSolver, MeSolver) are responsible with building the system
    (-1j*H).

    methods
    -------
    run(state0, tlist, args=None):
        Do an evolution starting with `state0` at `tlist[0]` and obtain a
        result for each time in `tlist`.
        The system's arguments can be changed with `args`.

    start(state0, t0):
        Set the initial values for an evolution by steps

    step(t, args=None):
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
    _super = None
    name = ""
    _t = 0
    _state = None
    _integrator = False
    optionsclass = SolverOptions

    def __init__(self):
        raise NotImplementedError

    def run(self, state0, tlist, args=None):
        state0 = self._prepare_state(state0)
        _integrator = self._get_integrator()
        if args:
            _integrator.update_args(args)
        _time_start = time()
        _integrator.set_state(tlist[0], state0)
        self.stats["preparation time"] += time() - _time_start
        res = Result(self.e_ops, self.options.results, self._super)
        res.add(tlist[0], self._state_qobj)

        progress_bar = get_progess_bar(self.options['progress_bar'])
        progress_bar.start(len(tlist)-1, **self.options['progress_kwargs'])
        for t, state in _integrator.run(tlist):
            progress_bar.update()
            res.add(t, self._restore_state(state, False))
        progress_bar.finished()

        self.stats['run time'] = progress_bar.total_time()
        self.stats.update(_integrator.stats)
        self.stats["method"] = _integrator.name
        res.stats = self.stats.copy()
        return res

    def start(self, state0, t0):
        _time_start = time()
        self._state = self._prepare_state(state0)
        self._t = t0
        self._integrator = self._get_integrator()
        self._integrator.set_state(self._t, self._state)
        self.stats["preparation time"] += time() - _time_start

    def step(self, t, args=None):
        if not self._integrator:
            raise RuntimeError("The `start` method must called first")
        if args:
            self._integrator.update_args(args)
            self._integrator.set_state(self._t, self._state)
        self._t, self._state = self._integrator.step(t, copy=False)
        return self._restore_state(self._state)

    def _get_integrator(self):
        method = self.options.ode["method"]
        rhs = self.options.ode["rhs"]
        td_system = not self._system.isconstant or None
        op_type = self.options.ode["Operator_data_type"]
        # TODO: with #1420, it should be changed to `in to._str2type`
        if op_type in to.dtypes:
            self._system = self._system.to(op_type)
        integrator = integrator_collection[method, rhs]
        # Check if the solver is supported by the integrator
        if not integrator_collection.check_condition(
            method, "", solver=self.name, time_dependent=td_system
        ):
            raise ValueError(f"ODE integrator method {method} not supported "
                f"by {self.name}" +
                ("for time dependent system" if td_system else "")
            )
        if not integrator_collection.check_condition(
            "", rhs, solver=self.name, time_dependent=td_system
        ):
            raise ValueError(f"ODE integrator rhs {rhs} not supported by " +
                f"{self.name}" +
                ("for time dependent system" if td_system else "")
            )
        return integrator(self._system, self.options)

    @property
    def option(self):
        return self._options

    @option.setter
    def option(self, new):
        if new is None:
            new = self.optionsclass()
        if not isinstance(new, self.optionsclass):
            raise TypeError("options must be an instance of",
                            self.optionsclass)
        self._options = new


class MultiTrajSolver:
    """
    ... TODO
    """
    _traj_solver_class = None
    _super = None
    name = ""
    optionsclass = None

    def __init__(self):
        self.seed_sequence = SeedSequence()
        self.traj_solver = False
        self.result = None
        raise NotImplementedError

    def _read_seed(self, seed, ntraj):
        """
        Read user provided seed(s) and produce one for each trajectories.
        Let numpy raise error for input that cannot be seeds.
        """
        if seed is None:
            seeds = self.seed_sequence.spawn(ntraj)
        elif isinstance(seed, SeedSequence):
            seeds = seed.spawn(ntraj)
        elif not isinstance(seed, list):
            seeds = SeedSequence(seed).spawn(ntraj)
        elif isinstance(seed, list) and len(seed) >= ntraj:
            seeds = [SeedSequence(seed_) for seed_ in seed[:ntraj]]
        else:
            raise ValueError("A seed list must be longer than ntraj")
        return seeds

    def start(self, state0, t0, *, ntraj=1, seed=None):
        """Prepare the Solver for stepping."""
        seed = self._read_seed(seed, 1)[0]
        self.traj_solvers = []
        for _ in range(ntraj):
            traj_solver = self._traj_solver_class(self)
            traj_solver.start(state0, t0, Generator(self.bit_gen(seed)))
            self.traj_solvers.append(traj_solver)

    def step(self, t, args=None):
        """Get the state at `t`"""
        if not self.traj_solvers:
            raise RuntimeError("The `start` method must called first")
        out = [traj_solver.step(t, args) for traj_solver in self.traj_solvers]
        return out if len(out) > 1 else out[0]

    def run(self, state0, tlist, args=None,
            ntraj=1, timeout=0, target_tol=None, seed=None):
        """
        Compute ntraj trajectories starting from `state0`.
        """
        self._check_state_dims(state0)
        if self.options.mcsolve['keep_runs_results']:
            self.result = MultiTrajResult(len(self.c_ops), len(self.e_ops))
        else:
            self.result = MultiTrajResultAveraged(len(self.c_ops),
                                                  len(self.e_ops))
        self._run_solver = self._traj_solver_class(self)
        self._run_args = state0, tlist, args
        self.result._to_dm = not self._super
        self.result.stats['run time'] = 0
        self.add_trajectories(ntraj, timeout, target_tol, seed)
        return self.result

    def add_trajectories(self, ntraj=1, timeout=0, target_tol=None, seed=None):
        """
        Add ntraj more trajectories.
        """
        if self.result is None:
            raise RuntimeError("No previous computation, use `run` first.")
        start_time = time()
        seeds = sel._read_seed(seed, ntraj)
        map_func = get_map(self.options.mcsolve)
        map_func(self._run_solver.run, seeds, self._run_args, {},
                 reduce_func=self.result.add,
                 map_kw=self.options.mcsolve['map_options'],
                 progress_bar=self.options["progress_bar"],
                 progress_bar_kwargs=self.options["progress_kwargs"]
                )
        self.result.stats['run time'] += time() - start_time
        self.result.stats.update(self.stats)
        return self.result

    @property
    def option(self):
        return self._options

    @option.setter
    def option(self, new):
        if new is None:
            new = self.optionsclass()
        if not isinstance(new, self.optionsclass):
            raise TypeError("options must be an instance of",
                            self.optionsclass)
        self._options = new
