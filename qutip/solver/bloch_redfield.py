# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, QuSTaR
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

__all__ = ['brmesolve', 'bloch_redfield_solve']

import numpy as np
import os
import time
import types
import warnings
import scipy.integrate
from ..core import (
    Qobj, spre, unstack_columns, stack_columns, expect, Cubic_Spline,
)
from ..core import data as _data
from .. import settings as qset
from ..core.cy.openmp.utilities import check_use_openmp
from ..ui.progressbar import BaseProgressBar, TextProgressBar
#from .br_codegen import BR_Codegen
#from ._brtensor import bloch_redfield_tensor
from .options import SolverOptions
from .result import Result
from .mesolve import MeSolver
from .solver_base import Solver

# -----------------------------------------------------------------------------
# Solve the Bloch-Redfield master equation
#
def brmesolve(H, psi0, tlist, a_ops=[], e_ops=[], c_ops=[],
              args={}, use_secular=True, sec_cutoff=0.1,
              # Shouldn't this be solver.atol?
              tol=qset.core['atol'],
              spectra_cb=None, options=None,
              progress_bar=None, _safe_mode=True, verbose=False):
    """
    Solves for the dynamics of a system using the Bloch-Redfield master equation,
    given an input Hamiltonian, Hermitian bath-coupling terms and their associated
    spectrum functions, as well as possible Lindblad collapse operators.

    For time-independent systems, the Hamiltonian must be given as a Qobj,
    whereas the bath-coupling terms (a_ops), must be written as a nested list
    of operator - spectrum function pairs, where the frequency is specified by
    the `w` variable.

    *Example*

        a_ops = [[a+a.dag(),lambda w: 0.2*(w>=0)]]

    For time-dependent systems, the Hamiltonian, a_ops, and Lindblad collapse
    operators (c_ops), can be specified in the QuTiP string-based time-dependent
    format.  For the a_op spectra, the frequency variable must be `w`, and the
    string cannot contain any other variables other than the possibility of having
    a time-dependence through the time variable `t`:

    *Example*

        a_ops = [[a+a.dag(), '0.2*exp(-t)*(w>=0)']]

    It is also possible to use Cubic_Spline objects for time-dependence.  In
    the case of a_ops, Cubic_Splines must be passed as a tuple:

    *Example*

        a_ops = [ [a+a.dag(), (f(w), g(t))] ]

    where f(w) and g(t) are strings or Cubic_spline objects for the bath
    spectrum and time-dependence, respectively.

    Finally, if one has bath-couplimg terms of the form
    H = f(t)*a + conj[f(t)]*a.dag(), then the correct input format is

    *Example*

              a_ops = [ [(a,a.dag()), (f(w), g1(t), g2(t))],... ]

    where f(w) is the spectrum of the operators while g1(t) and g2(t)
    are the time-dependence of the operators `a` and `a.dag()`, respectively

    Parameters
    ----------
    H : Qobj / list
        System Hamiltonian given as a Qobj or
        nested list in string-based format.

    psi0: Qobj
        Initial density matrix or state vector (ket).

    tlist : array_like
        List of times for evaluating evolution

    a_ops : list
        Nested list of Hermitian system operators that couple to
        the bath degrees of freedom, along with their associated
        spectra.

    e_ops : list
        List of operators for which to evaluate expectation values.

    c_ops : list
        List of system collapse operators, or nested list in
        string-based format.

    args : dict
        Placeholder for future implementation, kept for API consistency.

    use_secular : bool {True}
        Use secular approximation when evaluating bath-coupling terms.

    sec_cutoff : float {0.1}
        Cutoff for secular approximation.

    tol : float {qutip.setttings.atol}
        Tolerance used for removing small values after
        basis transformation.

    spectra_cb : list
        DEPRECIATED. Do not use.

    options : :class:`qutip.solver.SolverOptions`
        Options for the solver.

    progress_bar : BaseProgressBar
        Optional instance of BaseProgressBar, or a subclass thereof, for
        showing the progress of the simulation.

    Returns
    -------
    result: :class:`qutip.solver.Result`

        An instance of the class :class:`qutip.solver.Result`, which contains
        either an array of expectation values, for operators given in e_ops,
        or a list of states for the times specified by `tlist`.
    """
    _prep_time = time.time()
    # This allows for passing a list of time-independent Qobj
    # as allowed by mesolve
    if isinstance(H, list):
        if np.all([isinstance(h, Qobj) for h in H]):
            H = sum(H)

    if isinstance(c_ops, Qobj):
        c_ops = [c_ops]

    if isinstance(e_ops, Qobj):
        e_ops = [e_ops]

    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    if not (spectra_cb is None):
        warnings.warn("The use of spectra_cb is deprecated.", FutureWarning)
        _a_ops = []
        for kk, a in enumerate(a_ops):
            _a_ops.append([a, spectra_cb[kk]])
        a_ops = _a_ops

    if _safe_mode:
        _solver_safety_check(H, psi0, a_ops+c_ops, e_ops, args)

    # check for type (if any) of time-dependent inputs
    _, n_func, n_str = td_format_check(H, a_ops+c_ops)

    if progress_bar is None:
        progress_bar = BaseProgressBar()
    elif progress_bar is True:
        progress_bar = TextProgressBar()

    if options is None:
        options = SolverOptions()

    #if (not options.rhs_reuse) or (not config.tdfunc):
    #    # reset config collapse and time-dependence flags to default values
    #    config.reset()

    # check if should use OPENMP
    #check_use_openmp(options)

    if n_str == 0:

        R, ekets = bloch_redfield_tensor(
            H, a_ops, spectra_cb=None, c_ops=c_ops, use_secular=use_secular,
            sec_cutoff=sec_cutoff)

        output = Result()
        output.solver = "brmesolve"
        output.times = tlist

        results = bloch_redfield_solve(
            R, ekets, psi0, tlist, e_ops, options, progress_bar=progress_bar)

        if e_ops:
            output.expect = results
        else:
            output.states = results

        return output

    elif n_str != 0 and n_func == 0:
        output = _td_brmesolve(
            H, psi0, tlist, a_ops=a_ops, e_ops=e_ops, c_ops=c_ops, args=args,
            use_secular=use_secular, sec_cutoff=sec_cutoff, tol=tol,
            options=options, progress_bar=progress_bar, _safe_mode=_safe_mode,
            verbose=verbose, _prep_time=_prep_time)

        return output

    else:
        raise Exception('Cannot mix func and str formats.')


def _ode_rhs(t, state, oper):
    state = _data.dense.fast_from_numpy(state)
    return _data.matmul(oper, state, dtype=_data.Dense).as_ndarray()[:, 0]


# -----------------------------------------------------------------------------
# Evolution of the Bloch-Redfield master equation given the Bloch-Redfield
# tensor.
#
def bloch_redfield_solve(R, ekets, rho0, tlist, e_ops=[], options=None):
    progress_bar=None
    """
    Evolve the ODEs defined by Bloch-Redfield master equation. The
    Bloch-Redfield tensor can be calculated by the function
    :func:`bloch_redfield_tensor`.

    Parameters
    ----------

    R : :class:`qutip.qobj`
        Bloch-Redfield tensor.

    ekets : array of :class:`qutip.qobj`
        Array of kets that make up a basis tranformation for the eigenbasis.

    rho0 : :class:`qutip.qobj`
        Initial density matrix.

    tlist : *list* / *array*
        List of times for :math:`t`.

    e_ops : list of :class:`qutip.qobj` / callback function
        List of operators for which to evaluate expectation values.

    options : :class:`qutip.SolverOptions`
        Options for the ODE solver.

    Returns
    -------

    output: :class:`qutip.solver`

        An instance of the class :class:`qutip.solver`, which contains either
        an *array* of expectation values for the times specified by `tlist`.

    """

    if options is None:
        options = SolverOptions()

    if options['tidy']:
        R.tidyup()

    if progress_bar is None:
        progress_bar = BaseProgressBar()
    elif progress_bar is True:
        progress_bar = TextProgressBar()

    #
    # check initial state
    #
    if rho0.isket:
        # Got a wave function as initial state: convert to density matrix.
        rho0 = rho0.proj()

    #
    # prepare output array
    #
    n_tsteps = len(tlist)
    dt = tlist[1] - tlist[0]
    result_list = []

    #
    # transform the initial density matrix and the e_ops opterators to the
    # eigenbasis
    #
    rho_eb = rho0.transform(ekets).data
    e_eb_ops = [e.transform(ekets) for e in e_ops]

    for e_eb in e_eb_ops:
        if e_eb.isherm:
            result_list.append(np.zeros(n_tsteps, dtype=float))
        else:
            result_list.append(np.zeros(n_tsteps, dtype=complex))

    #
    # setup integrator
    #
    initial_vector = stack_columns(rho_eb).to_array()[:, 0]
    r = scipy.integrate.ode(_ode_rhs)
    r.set_f_params(R.data)
    r.set_integrator('zvode', method=options['method'], order=options['order'],
                     atol=options['atol'], rtol=options['rtol'],
                     nsteps=options['nsteps'],
                     first_step=options['first_step'],
                     min_step=options['min_step'],
                     max_step=options['max_step'])
    r.set_initial_value(initial_vector, tlist[0])

    #
    # start evolution
    #
    dt = np.diff(tlist)
    progress_bar.start(n_tsteps)
    for t_idx, _ in enumerate(tlist):
        progress_bar.update(t_idx)
        if not r.successful():
            break

        rho_eb = unstack_columns(_data.dense.fast_from_numpy(r.y), rho0.shape)
        rho_eb = Qobj(rho_eb, dims=rho0.dims)

        # calculate all the expectation values, or output rho_eb if no
        # expectation value operators are given
        if e_ops:
            for m, e in enumerate(e_eb_ops):
                result_list[m][t_idx] = expect(e, rho_eb)
        else:
            result_list.append(rho_eb.transform(ekets, True))

        if t_idx < n_tsteps - 1:
            r.integrate(r.t + dt[t_idx])
    progress_bar.finished()
    return result_list


def to_QEvo(op, args, times):
    if isinstance(op, QobjEvo):
        return op
    elif isinstance(op, (list, Qobj)):
        return QobjEvo(op, args=args, tlist=times)
    elif callable(op):
        return QobjEvoFunc(op, args=args)
    else:
        raise ValueError("Invalid time dependent format")


class BrMeSolver(MeSolver):
    def __init__(self, H, a_ops, c_ops=[], e_ops=None,
                 use_secular=True, sec_cutoff=0.0,
                 options=None, times=None, args=None, feedback_args=None,
                 _safe_mode=False):
        _time_start = time()
        self.stats = {}
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

        self.e_ops = e_ops
        self.options = options
        atol = options["atol"]

        constant_system = True

        H = to_QEvo(H, args, times)
        constant_system = constant_system and H.const

        c_evos = []
        for op in c_ops:
            c_evos.append(to_QEvo(op, args, times))
            constant_system = constant_system and c_evos[-1].const

        a_evos = []
        spectrum = []
        for (op, spec) in a_ops:
            if isinstance(spec, (tuple, list)):
                raise ValueError("a_ops format changed from v5, use: "
                                 "a_ops=[([op, g(t)], f(w))]")
            if isinstance(op, tuple):
                a_evos.append(to_QEvo(op[0]))
                constant_system = constant_system and a_evos[-1].const
                a_evos.append(to_QEvo(op[1]))
                constant_system = constant_system and a_evos[-1].const
                spectrum.append(build_spectrum(spec))
                spectrum.append(build_spectrum(spec))
            else:
                spectrum.append(build_spectrum(spec))
                a_evos.append(to_QEvo(op))
                constant_system = constant_system and a_evos[-1].const
                if not a_evos[-1].isherm:
                    spectrum.append(build_spectrum(spec))
                    a_evos.append(a_evos[-1].dag())

        if constant_system:
            R = bloch_redfield_tensor(H(0),
                [(a(0), spec) for a, spec in zip(a_evos, spectrum)],
                [c(0) for c in c_evos], use_secular, sec_cutoff, atol,
                evecs=False, basis='original')
            self._system = QobjEvo(R)
        else:
            self._system = BR_RHS(H, a_ops, spectrum, c_ops,
                                  use_secular, sec_cutoff, atol)

        self._evolver = self._get_evolver(options, args, feedback_args)
        self.stats["preparation time"] = time() - _time_start
        self.stats['solver'] = "Bloch Redfield Equation Evolution"
