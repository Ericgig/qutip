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

__all__ = ['mcsolve']

import warnings

import numpy as np
from numpy.random import RandomState, randint
from scipy.integrate import ode
from scipy.integrate._ode import zvode

from ..core import Qobj, QobjEvo
from ..core import data as _data

"""from ..parallel import parallel_map, serial_map
from ._mcsolve import CyMcOde, CyMcOdeDiag
from .sesolve import sesolve"""

from .options import SolverOptions
from .result import Result
from .solver import Solver
# from ..ui.progressbar import TextProgressBar, BaseProgressBar

#
# Internal, global variables for storing references to dynamically loaded
# cython functions


def mcsolve(H, psi0, tlist, c_ops=None, e_ops=None, ntraj=0,
            feedback_args=None, args=None, options=None, _safe_mode=True):
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

    progress_bar: BaseProgressBar
        Optional instance of BaseProgressBar, or a subclass thereof, for
        showing the progress of the simulation. Set to None to disable the
        progress bar.

    map_func: function
        A map function for managing the calls to the single-trajactory solver.

    map_kwargs: dictionary
        Optional keyword arguments to the map_func function.

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

    if len(c_ops) == 0:
        warnings.warn("No c_ops, using sesolve")
        return sesolve(H, psi0, tlist, e_ops=e_ops, args=args,
                       options=options, progress_bar=progress_bar,
                       _safe_mode=_safe_mode)

    # load monte carlo class
    mc = McSolver(H, c_ops, e_ops, options, tlist,
                  args, feedback_args, _safe_mode)

    # Run the simulation
    return mc.run(psi0, tlist=tlist, num_traj=num_traj)


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
        if e_ops is None:
            e_ops = []
        if options is None:
            options = SolverOptions()
        elif not isinstance(options, SolverOptions):
            raise ValueError("options must be an instance of "
                             "qutip.solver.SolverOptions")
        if args is None:
            args = {}
        if feedback_args is None:
            feedback_args = {}

        self._safe_mode = _safe_mode
        self._super = True
        self._state = None
        self._t = 0
        self._seeds = []

        self.e_ops = e_ops
        self.options = options
        self.res = MultiTrajResult()

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
        self.c_ops = n_evos
        self._evolver = McEvolver(self._system, self.c_ops, n_evos, options,
                                  args, feedback_args)

    def seed(self, ntraj, seeds=[]):
        # setup seeds array
        np.random.seed()
        try:
            seed = int(seeds)
            np.random.seed(seed)
            seeds = []
        except TypeError:
            pass

        if len(seeds) < ntraj:
            self._seeds = seeds + list(randint(0, 2**32,
                                               size=ntraj-len(seeds),
                                               dtype=np.uint32))
        else:
            self._seeds = seeds[:ntraj]

    def _safety_check(self, state):
        return None

    def run(self, state0, tlist, num_traj=1, args={}):
        if args:
            self._evolver.update_args(args)
        # todo, use parallel map
        if len(self._seeds) != num_traj:
            self.seed(num_traj)
        for seed in self._seeds:
            self.res.add(self._add_traj(seed))
        return self.res

    def _add_traj(self, seed):
        self._evolver.set(self._prepare_state(state0),
                          tlist[0], seed)
        res_1 = Result(self.e_ops, self.options, False)
        res_1.add(tlist[0], state0)
        for t, state in self._evolver.run(tlist):
            res_1.add(t, self._restore_state(state))
        return res_1

    def add_traj(self, num_traj):
        n_need = len(self.res.trajectories) + num_traj
        n_seed = len(self.seeds)
        if n_seed <= n_need:
            self._seeds = seeds + list(randint(0, 2**32,
                                               size=n_need-n_seed,
                                               dtype=np.uint32))
        elif n_seed >= n_need:
            self._seeds = self._seeds[:n_need]
        for seed in self._seeds[-num_traj:]:
            self.res.add(self._add_traj(seed))
        return self.res

    def continue_runs(self, tlist):
        raise NotImplementedError

    def _restore_state(self, state):
        return Qobj(state,
                    dims=self._state_dims,
                    type=self._state_type,
                    copy=True) / _data.norm.l2(state)

from .evolver import Evolver, get_evolver

class McEvolver(Evolver):
    def __init__(self, system, c_ops, n_ops, options, args, feedback_args):
        self.options = options
        self._evolver = get_evolver(system, options, args, feedback_args)
        self.collapses = []
        self.c_ops = c_ops
        self.n_ops = n_ops

        self.norm_steps = 5 # TODO: take from McOptions
        self.norm_t_tol = 1e-6
        self.norm_tol = 1e-6

    def set(self, state, t0, seed, options=None):
        np.random.seed(seed)
        self.target_norm = np.random.rand()
        self.options = options or self.options
        self._evolver.set(state, t0, self.options)

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
        y_old = self.get_state()
        t_old = self.t
        norm_old = _data.norm.l2(self.get_state())
        while self.t < t:
            state = self._evolver.step(t, step=1)
            norm = _data.norm.l2(state)
            if norm <= self.target_norm:
                self.do_collapse(norm_old, norm, t_old)
                t_old = self.t
                norm_old = _data.norm.l2(self.get_state())
            else:
                t_old = self.t
                norm_old = norm
        return _data.mul(self.get_state(), 1 / norm_old)

    def get_state(self):
        return self._evolver.get_state()

    def set_state(self, state0, t):
        self._evolver.set_state(state0, t)

    @property
    def t(self):
        return self._evolver.t

    def do_collapse(self, norm_old, norm, t_prev):
        t_final = self.t
        tries = 0
        while tries < self.norm_steps:
            tries += 1
            if (t_final - t_prev) < self.norm_t_tol:
                t_prev = t_final
                break
            t_guess = (
                t_prev
                + ((t_final - t_prev)
                   * np.log(norm_old / self.target_norm)
                   / np.log(norm_old / norm))
            )
            if (t_guess - t_prev) < self.norm_t_tol:
                t_guess = t_prev + self.norm_t_tol
            state = self._evolver.step(t_guess)
            norm2_guess = _data.norm.l2(state)
            if (np.abs(self.target_norm - norm2_guess) < self.norm_tol * self.target_norm):
                break
            elif (norm2_guess < target_norm):
                # t_guess is still > t_jump
                t_final = t_guess
                norm = norm2_guess
            else:
                # t_guess < t_jump
                t_prev = t_guess
                norm_old = norm2_guess

        if tries >= self.norm_steps:
            raise Exception("Norm tolerance not reached. " +
                            "Increase accuracy of ODE solver or " +
                            "SolverOptions.mcsolve['norm_steps'].")

        # t_guess, state is at the collapse
        probs = np.zeros(len(self.n_ops)+1)
        for i, n_op in enumerate(self.n_ops):
            probs[i+1] = n_op.expect(t_guess, state, 1)
        probs = np.cumsum(probs)
        which = np.searchsorted(probs, probs[-1] * np.random.rand())-1

        state_new = self.c_ops[which].mul(t_guess, state)
        state_new = _data.mul(state_new, 1 / _data.norm.l2(state_new))

        self.collapses.append((t_guess, which))
        self.set_state(state_new, t_guess)
