__all__ = []

import numpy as np
# from ..core import data as _data

from .. import Qobj, QobjEvo, QobjEvoFunc
from qutip.core.qobjevo import QobjEvoBase
from .result import Result
from .integrator import integrator_collection
from ..ui.progressbar import get_progess_bar
from ..core.data import to
from time import time
from .parallel import get_map


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
        self.seed_sequence = np.random.SeedSequence()
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
        elif isinstance(seed, np.random.SeedSequence):
            seeds = seed.spawn(ntraj)
        elif not isinstance(seed, list):
            seeds = np.random.SeedSequence(seed).spawn(ntraj)
        elif isinstance(seed, list) and len(seed) >= ntraj:
            seeds = [np.random.SeedSequence(seed_) for seed_ in seed[:ntraj]]
        else:
            raise ValueError("A seed list must be longer than ntraj")
        return seeds

    def start(self, state0, t0, *, ntraj=1, seed=None):
        """Prepare the Solver for stepping"""
        seeds = self._read_seed(seed, ntraj)
        self.traj_solvers = []

        for seed in seeds:
            traj_solver = self._traj_solver_class(self)
            traj_solver.start(state0, t0, seed)
            self.traj_solvers.append(traj_solver)

    def step(self, t, args=None):
        """Get the state at `t`"""
        if not self.traj_solvers:
            raise RuntimeError("The `start` method must called first.")
        multi = len(self.traj_solvers) = 1
        out = [traj_solver.step(t, args, safe=multi)
               for traj_solver in self.traj_solvers]
        return out if len(out) > 1 else out[0]

    def run(self, state0, tlist, args=None,
            ntraj=1, timeout=0, target_tol=None, seed=None):
        """
        Compute ntraj trajectories starting from `state0`.
        """
        self._check_state_dims(state0)
        result_cls = (MultiTrajResult
                      if self.options.mcsolve['keep_runs_results']
                      else MultiTrajResultAveraged)
        self.result = result_cls(len(self.c_ops), len(self.e_ops), state0)
        self._run_solver = self._traj_solver_class(self)
        self._run_args = state0, tlist, args
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
        seeds = self._read_seed(seed, ntraj)
        map_func = get_map[self.options.mcsolve]
        map_kw = self.options.mcsolve['map_options']
        if timeout:
            map_kw['job_timeout'] = timeout
        if target_tol:
            self.result._set_check_expect_tol(target_tol)
        map_func(
            self._run_solver.run, seeds, self._run_args, {},
            reduce_func=self.result.add, map_kw=map_kw,
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


class _OneTraj(Solver):
    """
    ... TODO
    """
    name = ""

    def start(self, state0, t0, seed=None):
        """Prepare the Solver for stepping."""
        self.generator = Generator(self.bit_gen(seed))
        self._state = self._prepare_state(state0)
        self._t = t0
        self._integrator.set_state(self._t, self._state)

    def step(self, t, args={}):
        """Get the state at `t`."""
        if args:
            self._integrator.arguments(args)
            [op.arguments(args) for op in self.c_ops]
            [op.arguments(args) for op in self.n_ops]
            self._integrator.reset()
        self._t, self._state = self._step(t, copy=False)
        return self._restore_state(self._state)

    def run(self, state0, tlist, args=None, seed=None):
        _time_start = time()
        if args:
            self._integrator.arguments(args)
            [op.arguments(args) for op in self.c_ops]
            [op.arguments(args) for op in self.n_ops]
        if not isinstance(seed, SeedSequence):
            seed = SeedSequence(seed)
        self.generator = Generator(self.bit_gen(seed))

        _state = self._prepare_state(state0)
        self._integrator.set_state(tlist[0], _state)

        result = Result(self.e_ops, self.options.results, state0)
        result.add(tlist[0], state0)
        for t in tlist[1:]:
            t, state = self._step(t)
            state_qobj = self._restore_state(state, False)
            result.add(t, state_qobj)

        result.seed = seed
        result.stats = {}
        result.stats['run time'] = time() - _time_start
        # result.stats.update(self._integrator.stats)
        result.stats.update(self.stats)
        return result

    def _step(self, t, copy=True):
        """Evolve to t, including jumps."""
        raise NotImplementedError
