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

__all__ = ['mcsolve', "McSolver", "MeMcSolver"]

import warnings

import numpy as np
from numpy.random import RandomState, randint
from scipy.integrate import ode
from scipy.integrate._ode import zvode
from ..core import (Qobj, QobjEvo, spre, spost, liouvillian, isket, ket2dm,
                    stack_columns, unstack_columns)
from ..core.data import column_stack, column_unstack, to
from ..core import data as _data
from .options import SolverOptions
from .result import Result, MultiTrajResult, MultiTrajResultAveraged
from .solver import Solver
from .sesolve import sesolve
from .mesolve import mesolve
from .parallel import get_map
from .evolver import *
from time import time


def mcsolve(H, psi0, tlist, c_ops=None, e_ops=None, ntraj=1,
            feedback_args=None, args=None, options=None, seeds=None,
            _safe_mode=True):
    """Monte Carlo evolution of a state vector :math:`|\psi \\rangle` for a
    given Hamiltonian and sets of collapse operators, and possibly, operators
    for calculating expectation values. Options for the underlying ODE solver
    are given by the Options class.

    mcsolve supports time-dependent Hamiltonians and collapse operators using
    either Python functions of strings to represent time-dependent
    coefficients. Note that, the system Hamiltonian MUST have at least one
    constant term.

    As an example of a time-dependent problem, consider a Hamiltonian with two
    terms ``H0`` and ``H1``, where ``H1`` is time-dependent with coefficient
    ``sin(w*t)``, and collapse operators ``C0`` and ``C1``, where ``C1`` is
    time-dependent with coeffcient ``exp(-a*t)``.  Here, w and a are constant
    arguments with values ``W`` and ``A``.

    Using the Python function time-dependent format requires two Python
    functions, one for each collapse coefficient. Therefore, this problem could
    be expressed as::

        def H1_coeff(t,args):
            return sin(args['w']*t)

        def C1_coeff(t,args):
            return exp(-args['a']*t)

        H = [H0, [H1, H1_coeff]]

        c_ops = [C0, [C1, C1_coeff]]

        args={'a': A, 'w': W}

    or in String (Cython) format we could write::

        H = [H0, [H1, 'sin(w*t)']]

        c_ops = [C0, [C1, 'exp(-a*t)']]

        args={'a': A, 'w': W}

    Constant terms are preferably placed first in the Hamiltonian and collapse
    operator lists.

    Parameters
    ----------
    H : :class:`qutip.Qobj`, ``list``
        System Hamiltonian.

    psi0 : :class:`qutip.Qobj`
        Initial state vector

    tlist : array_like
        Times at which results are recorded.

    ntraj : int
        Number of trajectories to run.

    c_ops : :class:`qutip.Qobj`, ``list``
        single collapse operator or a ``list`` of collapse operators.

    e_ops : :class:`qutip.Qobj`, ``list``
        single operator as Qobj or ``list`` or equivalent of Qobj operators
        for calculating expectation values.

    args : dict
        Arguments for time-dependent Hamiltonian and collapse operator terms.

    options : SolverOptions
        Instance of ODE solver options.

    Returns
    -------
    results : :class:`qutip.solver.Result`
        Object storing all results from the simulation.

    .. note::

        It is possible to reuse the random number seeds from a previous run
        of the mcsolver by passing the output Result object seeds via the
        Options class, i.e. SolverOptions(seeds=prev_result.seeds).
    """
    args = args or {}
    feedback_args = feedback_args or {}
    options = options if options is not None else SolverOptions()

    # set the physics
    if not psi0.isket:
        raise ValueError("Initial state must be a state vector.")

    if c_ops is None:
        c_ops = []
    if len(c_ops) == 0:
        warnings.warn("No c_ops, using sesolve")
        return sesolve(H, psi0, tlist, e_ops=e_ops, args=args,
                       options=options, _safe_mode=_safe_mode)

    # load monte carlo class
    mc = McSolver(H, c_ops, e_ops, options, tlist,
                  args, feedback_args, _safe_mode)
    if seeds is not None:
        mc.seed(ntraj, seeds)

    # Run the simulation
    return mc.run(psi0, tlist=tlist, ntraj=ntraj)


def _prob_mcsolve(state):
    return _data.norm.l2(state)**2


def _prob_memcsolve(state):
    return _data.norm.trace(unstack_columns(state))


# -----------------------------------------------------------------------------
# MONTE CARLO CLASS
# -----------------------------------------------------------------------------
class McSolver(Solver):
    """
    Private class for solving Monte Carlo evolution from mcsolve
    """
    def __init__(self, H, c_ops, e_ops=None, options=None,
                 times=None, args=None, feedback_args=None,
                 _safe_mode=False):
        _time_start = time()
        self.stats = {}
        e_ops = e_ops or []
        options = options or SolverOptions()
        args = args or {}
        feedback_args = feedback_args or {}
        if not isinstance(options, SolverOptions):
            raise ValueError("options must be an instance of "
                             "qutip.solver.SolverOptions")

        self._safe_mode = _safe_mode
        self._super = False
        self._state = None
        self._state0 = None
        self._t = 0
        self._seeds = []

        self.e_ops = e_ops
        self.options = options

        if isinstance(H, QobjEvo):
            pass
        elif isinstance(H, (list, Qobj)):
            H = QobjEvo(H, args=args, tlist=times)
        elif callable(H):
            H = QobjEvoFunc(H, args=args)
        else:
            raise ValueError("Invalid Hamiltonian")

        c_evos = []
        for op in c_ops:
            if isinstance(op, QobjEvo):
                c_evos.append(op)
            elif isinstance(op, (list, Qobj)):
                c_evos.append(QobjEvo(op, args=args, tlist=times))
            elif callable(op):
                c_evos.append(QobjEvoFunc(op, args=args))
            else:
                raise ValueError("Invalid Hamiltonian")

        n_evos = [c_evo._cdc() for c_evo in c_evos]
        self._system = -1j* H
        for n_evo in n_evos:
            self._system -= 0.5 * n_evo
        self.c_ops = c_evos
        self._evolver = McEvolver(self, self.c_ops, n_evos,
                                  options, args, feedback_args)
        self.stats["preparation time"] = time() - _time_start
        self.stats['solver'] = "MonteCarlo Schrodinger Equation Evolution"

    def seed(self, ntraj, seeds=[]):
        # setup seeds array
        try:
            seed = int(seeds)
            np.random.seed(seed)
            seeds = []
        except TypeError:
            np.random.seed()

        if len(seeds) < ntraj:
            self._seeds = seeds + list(randint(0, 2**32,
                                               size=ntraj-len(seeds),
                                               dtype=np.uint32))
        else:
            self._seeds = seeds[:ntraj]

    def start(self, state0, t0, seed=None):
        self._state = self._prepare_state(state0)
        self._t = t0
        self._evolver.set_state(self._t, self._state, seed)

    def _safety_check(self, state):
        return None

    def run(self, state0, tlist, ntraj=1, args={}):
        start_time = time()
        if args:
            self._evolver.update_args(args)
        # todo, use parallel map
        if len(self._seeds) != ntraj:
            self.seed(ntraj)
        if self.options.mcsolve['keep_runs_results']:
            self.res = MultiTrajResult(len(self.c_ops))
        else:
            self.res = MultiTrajResultAveraged(len(self.c_ops))
        self.res.seeds = list(self._seeds)
        self.res._to_dm = not self._super
        self._prepare_state(state0)

        map_func = get_map(self.options.mcsolve)
        map_func(self._add_traj, self._seeds, (tlist,), {},
                 reduce_func=self.res.add,
                 map_kw=self.options.mcsolve['map_options'],
                 progress_bar=self.options["progress_bar"],
                 progress_bar_kwargs=self.options["progress_kwargs"]
                 )

        self.stats['run time'] = time() - start_time
        self.stats["method"] = self._evolver.name
        self.res.stats = self.stats.copy()
        return self.res

    def add_traj(self, ntraj, seeds=[]):
        start_time = time()
        self._seeds += seeds
        n_needed = self.res.num_traj + ntraj
        n_seed = len(self._seeds)
        if n_seed < n_needed:
            self._seeds += list(randint(0, 2**32, size=n_needed-n_seed,
                                        dtype=np.uint32))
        elif n_seed >= n_needed:
            self._seeds = self._seeds[:n_needed]

        map_func = get_map(self.options.mcsolve)
        map_func(self._add_traj, self._seeds[self.res.num_traj:],
                 (self.res.times,), {},
                 reduce_func=self.res.add,
                 map_kw=self.options.mcsolve['map_options'],
                 progress_bar=self.options["progress_bar"],
                 progress_bar_kwargs=self.options["progress_kwargs"]
                )
        self.stats['run time'] += time() - start_time
        self.res.stats.update(self.stats)
        return self.res

    def _add_traj(self, seed, tlist):
        _time_start = time()
        self._evolver.set_state(tlist[0], self._state0, seed)
        self.options.results['normalize_output'] = False # Done here
        res_1 = Result(self.e_ops, self.options.results, False)
        res_1.add(tlist[0], self._state_qobj)
        for t, state in self._evolver.run(tlist):
            state_qobj = self._restore_state(state)
            res_1.add(t, state_qobj)
        res_1.collapse = list(self._evolver.collapses)
        res_1.stats = {}
        res_1.stats['run time'] = time() - _time_start
        res_1.stats.update(self._evolver.stats)
        res_1.stats.update(self.stats)
        return res_1

    def _restore_state(self, state):
        norm = 1/_data.norm.l2(state)
        state = _data.mul(state, norm)
        qobj = Qobj(state,
                    dims=self._state_dims,
                    type=self._state_type,
                    copy=False)
        return qobj


class MeMcSolver(McSolver):
    def __init__(self, H, c_ops, sc_ops, e_ops=None, options=None,
                 times=None, args=None, feedback_args=None,
                 _safe_mode=False):
        _time_start = time()
        self.stats = {}
        e_ops = e_ops or []
        options = options or SolverOptions()
        args = args or {}
        feedback_args = feedback_args or {}
        if not isinstance(options, SolverOptions):
            raise ValueError("options must be an instance of "
                             "qutip.solver.SolverOptions")

        self._safe_mode = _safe_mode
        self._super = True
        self._state = None
        self._state0 = None
        self._t = 0
        self._seeds = []

        self.e_ops = e_ops
        self.options = options

        if isinstance(H, QobjEvo):
            pass
        elif isinstance(H, (list, Qobj)):
            H = QobjEvo(H, args=args, tlist=times)
        elif callable(H):
            H = QobjEvoFunc(H, args=args)
        else:
            raise ValueError("Invalid Hamiltonian")

        c_evos = []
        for op in c_ops:
            if isinstance(op, QobjEvo):
                c_evos.append(op)
            elif isinstance(op, (list, Qobj)):
                c_evos.append(QobjEvo(op, args=args, tlist=times))
            elif callable(op):
                c_evos.append(QobjEvoFunc(op, args=args))
            else:
                raise ValueError("Invalid Hamiltonian")

        sc_evos = []
        for op in sc_ops:
            if isinstance(op, QobjEvo):
                sc_evos.append(op)
            elif isinstance(op, (list, Qobj)):
                sc_evos.append(QobjEvo(op, args=args, tlist=times))
            elif callable(op):
                sc_evos.append(QobjEvoFunc(op, args=args))
            else:
                raise ValueError("Invalid Hamiltonian")

        ns_evos = [spre(op._cdc()) + spost(op._cdc()) for op in sc_evos]
        n_evos = [spre(op) * spost(op.dag()) for op in sc_evos]
        self._system = liouvillian(H, c_evos)
        for n_evo in ns_evos:
            self._system -= 0.5 * n_evo
        self.c_ops = n_evos
        self._evolver = McEvolver(self, self.c_ops, n_evos,
                                  options, args, feedback_args)
        self._evolver.norm_func = _prob_memcsolve
        self._evolver.prob_func = _prob_memcsolve
        self.stats["preparation time"] = time() - _time_start
        self.stats['solver'] = "MonteCarlo Master Equation Evolution"

    def _prepare_state(self, state):
        if isket(state):
            state = ket2dm(state)
        self._state_dims = state.dims
        self._state_shape = state.shape
        self._state_type = state.type
        self._state_qobj = state
        str_to_type = {layer.__name__.lower(): layer for layer in to.dtypes}
        if self.options.ode["State_data_type"].lower() in str_to_type:
            state = state.to(str_to_type[self.options.ode["State_data_type"].lower()])
        self._state0 = stack_columns(state.data)
        return self._state0

    def _restore_state(self, state):
        return Qobj(unstack_columns(state),
                    dims=self._state_dims,
                    type=self._state_type,
                    copy=True)


class McEvolver(Evolver):
    def __init__(self, solver, c_ops, n_ops, options,
                 args=None, feedback_args=None):
        args = args or {}
        feedback_args = feedback_args or {}
        self.options = options
        self.collapses = []
        self._evolver = solver._get_evolver(options, args, feedback_args)
        self.c_ops = c_ops
        self.n_ops = n_ops

        self.norm_steps = options.mcsolve['norm_steps']
        self.norm_t_tol = options.mcsolve['norm_t_tol']
        self.norm_tol = options.mcsolve['norm_tol']

        self.norm_func = _data.norm.l2
        self.prob_func = _prob_mcsolve
        self.name = self._evolver.name
        self._stats = self._evolver._stats

    def set_state(self, t, state, seed):
        np.random.seed(seed)
        self.target_norm = np.random.rand()
        self._evolver.set_state(t, state)
        self.collapses = []
        self._evolver.update_feedback(self.collapses)

    def update_args(self, args):
        self.system.arguments(args)
        [op.arguments(args) for op in self.c_ops]
        [op.arguments(args) for op in self.n_ops]

    def run(self, tlist):
        """ Yield (t, state(t)) for t in tlist, must be `set` before. """
        for t in tlist[1:]:
            yield t, self.step(t, False)

    def step(self, t, step=None):
        """ Evolve to t, must be `set` before. """
        tries = 0
        t_old, y_old = self.get_state(copy=True)
        norm_old = self.prob_func(y_old)
        while t_old < t:
            t_step, state = self._evolver.one_step(t, copy=True)
            norm = self.prob_func(state)
            if norm <= self.target_norm:
                self.do_collapse(norm_old, norm, t_old, y_old)
                t_old, y_old = self.get_state(copy=True)
                norm_old = self.prob_func(y_old)
            else:
                t_old = t_step
                norm_old = norm
                y_old = state

        return _data.mul(y_old, 1 / self.norm_func(y_old))

    def get_state(self, copy=False):
        return self._evolver.get_state(copy=copy)

    def do_collapse(self, norm_old, norm, t_prev, y_prev):
        t_final = self._evolver.get_state()[0]
        tries = 0
        while tries < self.norm_steps:
            tries += 1
            if (t_final - t_prev) < self.norm_t_tol:
                t_guess = t_final
                state = self._evolver.get_state()[1]
                break
            t_guess = (
                t_prev
                + ((t_final - t_prev)
                   * np.log(norm_old / self.target_norm)
                   / np.log(norm_old / norm))
            )
            if (t_guess - t_prev) < self.norm_t_tol:
                t_guess = t_prev + self.norm_t_tol
            _, state = self._evolver.backstep(t_guess)
            norm2_guess = self.prob_func(state)
            if (
                np.abs(self.target_norm - norm2_guess) <
                self.norm_tol * self.target_norm
            ):
                break
            elif (norm2_guess < self.target_norm):
                # t_guess is still > t_jump
                t_final = t_guess
                norm = norm2_guess
            else:
                # t_guess < t_jump
                t_prev = t_guess
                y_prev = state.copy()
                norm_old = norm2_guess

        if tries >= self.norm_steps:
            raise Exception("Norm tolerance not reached. " +
                            "Increase accuracy of ODE solver or " +
                            "SolverOptions.mcsolve['norm_steps'].")

        # t_guess, state is at the collapse
        probs = np.zeros(len(self.n_ops))
        for i, n_op in enumerate(self.n_ops):
            probs[i] = n_op.expect(t_guess, state, 1)
        probs = np.cumsum(probs)
        which = np.searchsorted(probs, probs[-1] * np.random.rand())

        state_new = self.c_ops[which].mul(t_guess, state)
        new_norm = self.norm_func(state_new)
        if new_norm < 1e-12:
            # This happen when the collapse is caused by numerical error
            state_new = _data.mul(state, 1 / self.norm_func(state))
        else:
            state_new = _data.mul(state_new, 1 / new_norm)
            self.collapses.append((t_guess, which))
            self.target_norm = np.random.rand()
        self._evolver.set_state(t_guess, state_new)
