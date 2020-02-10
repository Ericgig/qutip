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
This module provides solvers for the unitary Schrodinger equation.
"""

__all__ = ['sesolve']

import os
import types
import numpy as np
import scipy.integrate
import qutip.settings as qset
from qutip.qobj import Qobj
from qutip.operators import qeye
from qutip.qobjevo import QobjEvo
from scipy.linalg import norm as la_norm
from qutip.parallel import parallel_map, serial_map
from qutip.qobjevo_maker import qobjevo_maker
from qutip.cy.spconvert import dense1D_to_fastcsr_ket, dense2D_to_fastcsr_fmode
from qutip.cy.spmatfuncs import (cy_expect_psi, cy_ode_psi_func_td,
                                cy_ode_psi_func_td_with_state, normalize_inplace,
                                normalize_op_inplace, normalize_mixed)
from qutip.solver import (Result, Options, config, solver_safe,
                          SolverSystem, Solver, ExpectOps)
from qutip.superoperator import vec2mat
from qutip.ui.progressbar import (BaseProgressBar, TextProgressBar)
from qutip.cy.openmp.utilities import check_use_openmp, openmp_components
from itertools import product

def warn(text):
    print(text)


def stack_ket(kets):
    # TODO: speedup, Qobj to dense to Qobj, probably slow
    out = np.zeros((kets[0].shape[0], len(kets)), dtype=complex)
    for i, ket in enumerate(kets):
        out[:,i] = ket.full().ravel()
    return [Qobj(out)]

def _islistof(obj, type_, default, errmsg):
    if isinstance(obj, type_):
        obj = [obj]
        n = 0
    elif isinstance(obj, list):
        if any((not isinstance(ele, type_) for ele in obj)):
            raise TypeError(errmsg)
        n = len(obj)
    elif not obj and default is not None:
        obj = default
        n = len(obj)
    else:
        raise TypeError(errmsg)
    return obj, n


class SeSolver:
    def __init__(self, H, psi0=None, t0=0, tlist=[], e_ops=None, args=None,
                 options=None, progress_bar=None):
        e_ops = e_ops if e_ops is not None else []
        self.e_ops_dict = None
        if isinstance(e_ops, Qobj):
            self.e_ops = [e_ops]
        elif isinstance(e_ops, dict):
            self.e_ops_dict = e_ops
            self.e_ops = [e for e in e_ops.values()]
        self.progress_bar = progress_bar
        if progress_bar is None:
            progress_bar = BaseProgressBar()
        if progress_bar is True:
            progress_bar = TextProgressBar()
        self.options = options
        check_use_openmp(options)
        self.args = args if args is not None else {}

        self.H = -1j* qobjevo_maker(H, args, tlist=tlist,
                                    e_ops=e_ops, state=psi0)
        nthread = opt.openmp_threads if opt.use_openmp else 0
        self.H.compile(omp=nthread)
        self.with_state = bool(self.H.dynamics_args)
        self.cte = self.H.const
        self.shape = self.H.cte.shape
        self.dims = self.H.cte.dims
        if psi0 is not None:
            self._set_psi(psi0)
        else:
            self.psi0 = None
            self.psi = None
            self.func = None
        return self

    def _set_psi(self, psi0):
        self.psi0 = psi0
        self.psi = psi0.full().ravel("F")
        self.func = self.H.get_mul(self.psi)
        self.solver = self._get_solver()

    def _check(self, psi0):
        if not (psi0.isket or psi0.isunitary):
            raise TypeError("The unitary solver requires psi0 to be"
                            " a ket as initial state"
                            " or a unitary as initial operator.")
        if psi0.dims[0] != self.H.dims[1]:
            raise ValueError("The dimension of psi0 does not "
                             "fit the Hamiltonian")
        try:
            v = self.psi0.full().ravel('F')
            self.func(0., v) + v
        except Exception as e:
            raise ValueError("The Hamiltonian is invalid "
                             "or does not fit psi0") from e

    def run(self, tlist, psi0=None, args=None, outtype=Qobj, _safe_mode=True):
        psi0 = psi0 if psi0 is not None else self.psi0
        if args is not None:
            self.H.arguments(args)
        self.set(psi0, tlist[0])
        if _safe_mode:
            self._check()

        output = Result()
        output.solver = "sesolve"
        output.times = tlist

        states, expect = self.solver.run(psi0, tlist, e_ops, dims=psi0.dims)

        output.expect = expect
        output.num_expect = len(self.e_ops)
        if opt.store_final_state:
            output.final_state = self.transform(states[-1],
                                                self.psi0.dims,
                                                self.solver.statetype,
                                                outtype)
        if opt.store_states:
            output.states = [self.transform(psi, self.psi0.dims,
                                            self.solver.statetype, outtype)
                             for psi in states]

        if e_ops_dict:
            output.expect = {e: output.expect[n]
                             for n, e in enumerate(self.e_ops_dict.keys())}
        return res

    def step(self, t, args=None, outtype=Qobj, e_ops=[]):
        if args is not None:
            self.H.arguments(args)
        state = self.solver.step(self.psi, [self.t, t])
        self.t = t
        self.psi = state
        if e_ops:
            return [expect(op, state) for op in e_ops]
        return self.transform(states, self.psi0.dims,
                              self.solver.statetype, outtype)

    def set(self, psi0=None, t0=0):
        self.t0 = t0
        self.t = t0
        psi0 = psi0 if psi0 is not None else self.psi0
        self._set_psi(psi0)


def sesolve(H, psi0, tlist, e_ops=None, args=None, options=None,
            progress_bar=None, _safe_mode=True):
    """
    Schrodinger equation evolution of a state vector or unitary matrix
    for a given Hamiltonian.

    Evolve the state vector (`psi0`) using a given
    Hamiltonian (`H`), by integrating the set of ordinary differential
    equations that define the system. Alternatively evolve a unitary matrix in
    solving the Schrodinger operator equation.

    The output is either the state vector or unitary matrix at arbitrary points
    in time (`tlist`), or the expectation values of the supplied operators
    (`e_ops`). If e_ops is a callback function, it is invoked for each
    time in `tlist` with time and the state as arguments, and the function
    does not use any return values. e_ops cannot be used in conjunction
    with solving the Schrodinger operator equation

    Parameters
    ----------

    H : :class:`qutip.qobj`, :class:`qutip.qobjevo`, *list*, *callable*
        system Hamiltonian as a Qobj, list of Qobj and coefficient, QobjEvo,
        or a callback function for time-dependent Hamiltonians.
        list format and options can be found in QobjEvo's description.

    psi0 : :class:`qutip.qobj`
        initial state vector (ket)
        or initial unitary operator `psi0 = U`

    tlist : *list* / *array*
        list of times for :math:`t`.

    e_ops : None / list of :class:`qutip.qobj` / callback function
        single operator or list of operators for which to evaluate
        expectation values.
        For list operator evolution, the overlapse is computed:
            tr(e_ops[i].dag()*op(t))

    args : None / *dictionary*
        dictionary of parameters for time-dependent Hamiltonians

    options : None / :class:`qutip.Qdeoptions`
        with options for the ODE solver.

    progress_bar : None / BaseProgressBar
        Optional instance of BaseProgressBar, or a subclass thereof, for
        showing the progress of the simulation.

    Returns
    -------

    output: :class:`qutip.solver`

        An instance of the class :class:`qutip.solver`, which contains either
        an *array* of expectation values for the times specified by `tlist`, or
        an *array* or state vectors corresponding to the
        times in `tlist` [if `e_ops` is an empty list], or
        nothing if a callback function was given inplace of operators for
        which to calculate the expectation values.

    """
    if options is not None and options.rhs_reuse:
        warn("'rhs_reuse' of Options will be deprecated. "
             "Use the object interface of instead: 'SeSolver'")
        if "sesolve" in solver_safe:
            solver = solver_safe["sesolve"]
        else:
            raise Exception("Could not find sesolve Hamiltonian to reuse")
    else:
        solver = SeSolver(H, psi0, tlist[0], tlist, e_ops, args,
                          options, progress_bar)
        solver_safe["sesolve"] = solver
    return solver.run(tlist, psi0, args, _safe_mode=_safe_mode)


# -----------------------------------------------------------------------------
# Solve an ODE for func.
# Calculate the required expectation values or invoke callback
# function at each time step.
def _se_ode_solve(func, ode_args, psi0, tlist, e_ops,
                  opt, progress_bar, dims=None):
    """
    Internal function for solving ODEs.
    """
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # This function is made similar to mesolve's one for futur merging in a
    # solver class
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # prepare output array
    n_tsteps = len(tlist)
    output = Result()
    output.solver = "sesolve"
    output.times = tlist

    if psi0.isunitary:
        initial_vector = psi0.full().ravel('F')
        oper_evo = True
        size = psi0.shape[0]
        # oper_n = dims[0][0]
        # norm_dim_factor = np.sqrt(oper_n)
    elif psi0.isket:
        initial_vector = psi0.full().ravel()
        oper_evo = False
        # norm_dim_factor = 1.0

    r = scipy.integrate.ode(func)
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                     atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                     first_step=opt.first_step, min_step=opt.min_step,
                     max_step=opt.max_step)
    if ode_args:
        r.set_f_params(*ode_args)
    r.set_initial_value(initial_vector, tlist[0])

    e_ops_data = []
    output.expect = []
    if callable(e_ops):
        n_expt_op = 0
        expt_callback = True
        output.num_expect = 1
    elif isinstance(e_ops, list):
        n_expt_op = len(e_ops)
        expt_callback = False
        output.num_expect = n_expt_op
        if n_expt_op == 0:
            # fallback on storing states
            opt.store_states = True
        else:
            for op in e_ops:
                if op.isherm:
                    output.expect.append(np.zeros(n_tsteps))
                else:
                    output.expect.append(np.zeros(n_tsteps, dtype=complex))
        if oper_evo:
            for e in e_ops:
                e_ops_data.append(e.dag().data)
        else:
            for e in e_ops:
                e_ops_data.append(e.data)
    else:
        raise TypeError("Expectation parameter must be a list or a function")

    if opt.store_states:
        output.states = []

    if oper_evo:
        def get_curr_state_data(r):
            return vec2mat(r.y)
    else:
        def get_curr_state_data(r):
            return r.y

    #
    # start evolution
    #
    dt = np.diff(tlist)
    cdata = None
    progress_bar.start(n_tsteps)
    for t_idx, t in enumerate(tlist):
        progress_bar.update(t_idx)
        if not r.successful():
            raise Exception("ODE integration error: Try to increase "
                            "the allowed number of substeps by increasing "
                            "the nsteps parameter in the Options class.")
        # get the current state / oper data if needed
        if opt.store_states or opt.normalize_output or n_expt_op > 0 or expt_callback:
            cdata = get_curr_state_data(r)

        if opt.normalize_output:
            # normalize per column
            if oper_evo:
                cdata /= la_norm(cdata, axis=0)
                #cdata *= norm_dim_factor / la_norm(cdata)
                r.set_initial_value(cdata.ravel('F'), r.t)
            else:
                #cdata /= la_norm(cdata)
                norm = normalize_inplace(cdata)
                if norm > 1e-12:
                    # only reset the solver if state changed
                    r.set_initial_value(cdata, r.t)
                else:
                    r._y = cdata

        if opt.store_states:
            if oper_evo:
                fdata = dense2D_to_fastcsr_fmode(cdata, size, size)
                output.states.append(Qobj(fdata, dims=dims))
            else:
                fdata = dense1D_to_fastcsr_ket(cdata)
                output.states.append(Qobj(fdata, dims=dims, fast='mc'))

        if expt_callback:
            # use callback method
            output.expect.append(e_ops(t, Qobj(cdata, dims=dims)))

        if oper_evo:
            for m in range(n_expt_op):
                output.expect[m][t_idx] = (e_ops_data[m] * cdata).trace()
        else:
            for m in range(n_expt_op):
                output.expect[m][t_idx] = cy_expect_psi(e_ops_data[m], cdata,
                                                        e_ops[m].isherm)

        if t_idx < n_tsteps - 1:
            r.integrate(r.t + dt[t_idx])

    progress_bar.finished()

    if opt.store_final_state:
        cdata = get_curr_state_data(r)
        if opt.normalize_output:
            cdata /= la_norm(cdata, axis=0)
            # cdata *= norm_dim_factor / la_norm(cdata)
        output.final_state = Qobj(cdata, dims=dims)

    return output


def sesolve_old(H, psi0, tlist, e_ops=None, args=None, options=None,
            progress_bar=None, _safe_mode=True):
    """
    Schrodinger equation evolution of a state vector or unitary matrix
    for a given Hamiltonian.

    Evolve the state vector (`psi0`) using a given
    Hamiltonian (`H`), by integrating the set of ordinary differential
    equations that define the system. Alternatively evolve a unitary matrix in
    solving the Schrodinger operator equation.

    The output is either the state vector or unitary matrix at arbitrary points
    in time (`tlist`), or the expectation values of the supplied operators
    (`e_ops`). If e_ops is a callback function, it is invoked for each
    time in `tlist` with time and the state as arguments, and the function
    does not use any return values. e_ops cannot be used in conjunction
    with solving the Schrodinger operator equation

    Parameters
    ----------

    H : :class:`qutip.qobj`, :class:`qutip.qobjevo`, *list*, *callable*
        system Hamiltonian as a Qobj, list of Qobj and coefficient, QobjEvo,
        or a callback function for time-dependent Hamiltonians.
        list format and options can be found in QobjEvo's description.

    psi0 : :class:`qutip.qobj`
        initial state vector (ket)
        or initial unitary operator `psi0 = U`

    tlist : *list* / *array*
        list of times for :math:`t`.

    e_ops : None / list of :class:`qutip.qobj` / callback function
        single operator or list of operators for which to evaluate
        expectation values.
        For list operator evolution, the overlapse is computed:
            tr(e_ops[i].dag()*op(t))

    args : None / *dictionary*
        dictionary of parameters for time-dependent Hamiltonians

    options : None / :class:`qutip.Qdeoptions`
        with options for the ODE solver.

    progress_bar : None / BaseProgressBar
        Optional instance of BaseProgressBar, or a subclass thereof, for
        showing the progress of the simulation.

    Returns
    -------

    output: :class:`qutip.solver`

        An instance of the class :class:`qutip.solver`, which contains either
        an *array* of expectation values for the times specified by `tlist`, or
        an *array* or state vectors corresponding to the
        times in `tlist` [if `e_ops` is an empty list], or
        nothing if a callback function was given inplace of operators for
        which to calculate the expectation values.

    """
    if e_ops is None:
        e_ops = []
    if isinstance(e_ops, Qobj):
        e_ops = [e_ops]
    elif isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    if progress_bar is None:
        progress_bar = BaseProgressBar()
    if progress_bar is True:
        progress_bar = TextProgressBar()

    if not (psi0.isket or psi0.isunitary):
        raise TypeError("The unitary solver requires psi0 to be"
                        " a ket as initial state"
                        " or a unitary as initial operator.")

    if options is None:
        options = Options()
    if options.rhs_reuse and not isinstance(H, SolverSystem):
        warn("'rhs_reuse' of Options will be deprecated. "
             "Use the object interface of instead: 'SeSolver'")
        if "sesolve" in solver_safe:
            H = solver_safe["sesolve"]
        else:
            pass

    if args is None:
        args = {}

    check_use_openmp(options)

    if isinstance(H, SolverSystem):
        ss = H
    else:
        H = qobjevo_maker(H, args, tlist=tlist, e_ops=e_ops, state=psi0)
        ss = _sesolve_QobjEvo(H, tlist, args, options)

    func, ode_args = ss.makefunc(ss, psi0, args, e_ops, options)

    if _safe_mode:
        v = psi0.full().ravel('F')
        func(0., v, *ode_args) + v

    res = _se_ode_solve(func, ode_args, psi0, tlist, e_ops,
                        options, progress_bar, dims=psi0.dims)
    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}
    return res


# -----------------------------------------------------------------------------
# A time-dependent unitary wavefunction equation on the list-function format
#
def _sesolve_QobjEvo(H, tlist, args, opt):
    """
    Prepare the system for the solver, H can be an QobjEvo.
    """
    H_td = -1.0j * qobjevo_maker(H, args, tlist=tlist)
    nthread = opt.openmp_threads if opt.use_openmp else 0
    H_td.compile(omp=nthread)

    ss = SolverSystem()
    ss.type = "QobjEvo"
    ss.H = H_td
    ss.shape = H_td.cte.shape
    ss.dims = H_td.cte.dims
    ss.with_state = bool(H_td.dynamics_args)
    ss.makefunc = _qobjevo_set
    ss.makeoper = _qobjevo_set_oper
    solver_safe["sesolve"] = ss
    return ss

def _qobjevo_set(HS, psi, args, e_ops, opt):
    """
    From the system, get the ode function and args
    """
    H_td = HS.H
    H_td.arguments(args, psi, e_ops)
    if psi.isunitary:
        func = H_td.compiled_qobjevo.ode_mul_mat_f_vec
    elif psi.isket:
        func = H_td.compiled_qobjevo.mul_vec
    elif psi.isunitary:
        func = H_td.compiled_qobjevo.ode_mul_mat_f_oper
    else:
        func = H_td.compiled_qobjevo.ode_mul_mat_f_vec
    # else:
    #    raise TypeError("The unitary solver requires psi0 to be"
    #                    " a ket as initial state"
    #                    " or a unitary as initial operator.")
    return func, ()

def _qobjevo_set_oper(HS, psi, args, opt):
    """
    From the system, get the ode function and args
    """
    H_td = HS.H
    H_td.arguments(args)
    N = psi.shape[0]

    def _oper_evolution(t, mat):
        oper = H_td.compiled_qobjevo.ode_mul_mat_f_oper(t, mat)
        out = mat.reshape((N,N)).T @ H_td.call(t, 1) - oper.reshape((N,N)).T
        return out.ravel("F")

    return _oper_evolution, ()
