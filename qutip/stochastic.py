# -*- coding: utf-8 -*-
#
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
#
#    Significant parts of this code were contributed by Denis Vasilyev.
#
###############################################################################
import numpy as np
import scipy.sparse as sp
from qutip.cy.stochastic_solver import sme, sse, psme, psse, generic
from qutip.qobj import Qobj, isket, isoper, issuper
from qutip.states import ket2dm
from qutip.solver import Result
from qutip.td_qobj import td_Qobj
from qutip.superoperator import (spre, spost, mat2vec, vec2mat,
                                 liouvillian, lindblad_dissipator)
from qutip.solver import Options, _solver_safety_check
from qutip.parallel import serial_map
from qutip.ui.progressbar import TextProgressBar

__all__ = ['ssesolve', 'photocurrentsesolve',
           'smesolve', 'photocurrentmesolve',
           'stochastic_solver_info', 'general_stochastic']

def stochastic_solver_info():
    print( """Available solvers
    euler-maruyama:
        A simple generalization of the Euler method for ordinary
        differential equations to stochastic differential equations.
        Only solver which could take non-commuting sc_ops. *not tested*
        -Order 0.5
        -Code: 'euler-maruyama', 'euler', 0.5

    milstein, Order 1.0 strong Taylor scheme:
        Better approximate numerical solution to stochastic
        differential equations.
        -Order strong 1.0
        -Code: 'milstein', 1.0
        Numerical Solution of Stochastic Differential Equations
        Chapter 10.3 Eq. (3.1), By Peter E. Kloeden, Eckhard Platen

    milstein-imp, Order 1.0 implicit strong Taylor scheme:
        Implicit milstein scheme for the numerical simulation of stiff
        stochastic differential equations.
        -Order strong 1.0
        -Code: 'milstein-imp'
        Numerical Solution of Stochastic Differential Equations
        Chapter 12.2 Eq. (2.9), By Peter E. Kloeden, Eckhard Platen

    predictor-corrector:
        Generalization of the trapezoidal method to stochastic
        differential equations. More stable than explicit methods.
        -Order strong 0.5, weak 1.0
        Only the stochastic part is corrected.
            (alpha = 0, eta = 1/2)
            -Code: 'pred-corr', 'predictor-corrector', 'pc-euler'
        Both the deterministic and stochastic part corrected.
            (alpha = 1/2, eta = 1/2)
            -Code: 'pred-corr-2', 'pc-euler-2'
        Numerical Solution of Stochastic Differential Equations
        Chapter 15.5 Eq. (5.4), By Peter E. Kloeden, Eckhard Platen

    platen:
        Explicit scheme, create the milstein using finite difference instead of
        derivatives. Also contain some higher order terms, thus converge better
        than milstein while staying strong order 1.0.
        Do not require derivatives, therefore usable for
        :func:`qutip.stochastic.general_stochastic`
        -Order strong 1.0, weak 2.0
        -Code: 'platen', 'platen1', 'explicit1'
        The Theory of Open Quantum Systems
        Chapter 7 Eq. (7.47), H.-P Breuer, F. Petruccione

    taylor15, Order 1.5 strong Taylor scheme:
        Solver with more terms of the Ito-Taylor expansion.
        Default solver for smesolve and ssesolve.
        -Order strong 1.5
        -Code: 'taylor15', 1.5
        Numerical Solution of Stochastic Differential Equations
        Chapter 10.4 Eq. (4.6), By Peter E. Kloeden, Eckhard Platen

    taylor15-imp, Order 1.5 implicit strong Taylor scheme:
        implicit Taylor 1.5 (alpha = 1/2, beta = doesn't matter)
        -Order strong 1.5
        -Code: 'taylor15-imp'
        Numerical Solution of Stochastic Differential Equations
        Chapter 12.2 Eq. (2.18), By Peter E. Kloeden, Eckhard Platen

    explicit15, Explicit Order 1.5 Strong Schemes:
        Reproduce the order 1.5 strong Taylor scheme using finite difference
        instead of derivatives. Slower than taylor15 but usable by
        :func:`qutip.stochastic.general_stochastic`
        -Order strong 1.5
        -Code: 'explicit15', 'platen15'
        Numerical Solution of Stochastic Differential Equations
        Chapter 11.2 Eq. (2.13), By Peter E. Kloeden, Eckhard Platen

    ---All solvers are usable in both smesolve and ssesolve and
    for both heterodyne and homodyne.
    The :func:`qutip.stochastic.general_stochastic` only accept derivatives free
    solvers: ['euler', 'platen', 'explicit15'].
    """)

class StochasticSolverOptions:
    """Class of options for stochastic solvers such as
    :func:`qutip.stochastic.ssesolve`, :func:`qutip.stochastic.smesolve`, etc.
    Options can be specified either as arguments to the constructor::

        sso = StochasticSolverOptions(nsubsteps=100, ...)

    or by changing the class attributes after creation::

        sso = StochasticSolverOptions()
        sso.nsubsteps = 1000

    The stochastic solvers :func:`qutip.stochastic.ssesolve`,
    :func:`qutip.stochastic.smesolve`, :func:`qutip.stochastic.ssepdpsolve` and
    :func:`qutip.stochastic.smepdpsolve` all take the same keyword arguments as
    the constructor of these class, and internally they use these arguments to
    construct an instance of this class, so it is rarely needed to explicitly
    create an instance of this class.

    Attributes
    ----------

    H : :class:`qutip.Qobj`
        System Hamiltonian.

    state0 : :class:`qutip.Qobj`
        Initial state vector (ket) or density matrix.

    times : *list* / *array*
        List of times for :math:`t`. Must be uniformly spaced.

    c_ops : list of :class:`qutip.Qobj`
        List of deterministic collapse operators.

    sc_ops : list of :class:`qutip.Qobj`
        List of stochastic collapse operators. Each stochastic collapse
        operator will give a deterministic and stochastic contribution
        to the equation of motion according to how the d1 and d2 functions
        are defined.

    e_ops : list of :class:`qutip.Qobj`
        Single operator or list of operators for which to evaluate
        expectation values.

    m_ops : list of :class:`qutip.Qobj`
        List of operators representing the measurement operators. The expected
        format is a nested list with one measurement operator for each
        stochastic increament, for each stochastic collapse operator.

    args : dict / list
        List of dictionary of additional problem-specific parameters.
        Implicit methods can adjust tolerance via args = {'tol':value}

    ntraj : int
        Number of trajectors.

    nsubsteps : int
        Number of sub steps between each time-spep given in `times`.

    d1 : function
        Function for calculating the operator-valued coefficient to the
        deterministic increment dt.

    d2 : function
        Function for calculating the operator-valued coefficient to the
        stochastic increment(s) dW_n, where n is in [0, d2_len[.

    d2_len : int (default 1)
        The number of stochastic increments in the process.

    dW_factors : array
        Array of length d2_len, containing scaling factors for each
        measurement operator in m_ops.

    rhs : function
        Function for calculating the deterministic and stochastic contributions
        to the right-hand side of the stochastic differential equation. This
        only needs to be specified when implementing a custom SDE solver.

    generate_A_ops : function
        Function that generates a list of pre-computed operators or super-
        operators. These precomputed operators are used in some d1 and d2
        functions.

    generate_noise : function
        Function for generate an array of pre-computed noise signal.

    homogeneous : bool (True)
        Wheter or not the stochastic process is homogenous. Inhomogenous
        processes are only supported for poisson distributions.

    solver : string
        Name of the solver method to use for solving the stochastic
        equations. Valid values are:
        1/2 order algorithms: 'euler-maruyama', 'pc-euler'
        1 order algorithms: 'milstein', 'platen', 'milstein-imp'
        3/2 order algorithms: 'taylor15', 'taylor15-imp', 'explicit15'
        call :func:`qutip.stochastic.stochastic_solver_info` for a description
        of the solvers.

        Implicit methods can adjust tolerance via args = {'tol':value},
        default is {'tol':1e-6}

    method : string ('homodyne', 'heterodyne', 'photocurrent')
        The name of the type of measurement process that give rise to the
        stochastic equation to solve. Specifying a method with this keyword
        argument is a short-hand notation for using pre-defined d1 and d2
        functions for the corresponding stochastic processes.

    distribution : string ('normal', 'poisson')
        The name of the distribution used for the stochastic increments.

    store_measurements : bool (default False)
        Whether or not to store the measurement results in the
        :class:`qutip.solver.SolverResult` instance returned by the solver.

    noise : array
        Vector specifying the noise.

    normalize : bool (default True)
        Whether or not to normalize the wave function during the evolution.

    options : :class:`qutip.solver.Options`
        Generic solver options.

    map_func: function
        A map function or managing the calls to single-trajactory solvers.

    map_kwargs: dictionary
        Optional keyword arguments to the map_func function function.

    progress_bar : :class:`qutip.ui.BaseProgressBar`
        Optional progress bar class instance.

    """
    def __init__(self, H=None, c_ops=[], sc_ops=[], state0=None,
                 e_ops=[], m_ops=None, store_measurement=False, dW_factors=None,
                 solver="euler", method="homodyne", normalize=1,
                 times=None, nsubsteps=1, ntraj=1, tol=None,
                 generate_noise=None, noise=None,
                 progress_bar=None, map_func=None, map_kwargs=None,
                 args={}, options=None):

        if options is None:
            options = Options()

        if progress_bar is None:
            progress_bar = TextProgressBar()

        # System
        # Cast to td_Qobj so the code has only one version for both the
        # constant and time-dependent case.
        if H is not None:
            try:
                self.H = td_Qobj(H, args=args, tlist=times, raw_str=True)
            except:
                raise Exception("The hamiltonian format is not valid")
        else:
            self.H = H

        if sc_ops:
            try:
                self.sc_ops = [td_Qobj(op, args=args, tlist=times,
                                 raw_str=True) for op in sc_ops]
            except:
                raise Exception("The sc_ops format is not valid.\n"+
                                "[ Qobj / td_Qobj / [Qobj,coeff]]")
        else:
            self.sc_ops = sc_ops

        if c_ops:
            try:
                self.c_ops = [td_Qobj(op, args=args, tlist=times,
                              raw_str=True) for op in c_ops]
            except:
                raise Exception("The c_ops format is not valid.\n"+
                                "[ Qobj / td_Qobj / [Qobj,coeff]]")
        else:
            self.c_ops = c_ops

        self.state0 = state0
        self.rho0 = mat2vec(state0.full()).ravel()
        #print(self.rho0.shape)

        # Observation
        self.e_ops = e_ops
        self.m_ops = m_ops
        self.store_measurement = store_measurement
        self.store_states = options.store_states
        self.dW_factors = dW_factors

        # Solver
        self.solver = solver
        self.method = method
        self.normalize = normalize
        self.times = times
        self.nsubsteps = nsubsteps
        self.dt = (times[1] - times[0]) / self.nsubsteps
        self.ntraj = ntraj
        if tol is not None:
            self.tol = tol
        elif "tol" in args:
            self.tol = args["tol"]
        else:
            self.tol = 1e-7

        # Noise
        if noise is not None:
            if isinstance(noise, int):
                # noise contain a seed
                np.random.seed(noise)
                noise = np.random.randint(0, 2**32, ntraj)
            noise = np.array(noise)
            if len(noise.shape) == 1:
                if noise.shape[0] < ntraj:
                    raise Exception("'noise' does not have enought seeds" +
                                    "len(noise) >= ntraj")
                # numpy seed must be between 0 and 2**32-1
                # 'u4': unsigned 32bit int
                self.noise = noise.astype("u4")
                self.noise_type = 0

            elif len(noise.shape) == 4:
                # taylor case not included
                dw_len = (2 if method == "heterodyne" else 1)
                dw_len_str = (" * 2" if method == "heterodyne" else "")
                if noise.shape[0] < ntraj:
                    raise Exception("'noise' does not have the right shape" +
                                    "shape[0] >= ntraj")
                if noise.shape[1] < len(times):
                    raise Exception("'noise' does not have the right shape" +
                                    "shape[1] >= len(times)")
                if noise.shape[2] < nsubsteps:
                    raise Exception("'noise' does not have the right shape" +
                                    "shape[2] >= nsubsteps")
                if noise.shape[3] < len(self.sc_ops) * dw_len:
                    raise Exception("'noise' does not have the right shape" +
                                    "shape[3] >= len(self.sc_ops)" + dw_len_str)
                self.noise_type = 1
                self.noise = noise

        else:
            self.noise = np.random.randint(0, 2**32, ntraj).astype("u4")
            self.noise_type = 0

        # Map
        self.progress_bar = progress_bar
        if self.ntraj > 1 and map_func:
            self.map_func = map_func
        else:
            self.map_func = serial_map
        self.map_kwargs = map_kwargs if map_kwargs is not None else {}

        # Other
        self.options = options
        self.args = args

        if self.solver in ['euler-maruyama', 'euler', 50, 0.5]:
            self.solver_code = 50
            self.solver = 'euler-maruyama'

        elif self.solver in ['platen', 'platen1', 'explicit1', 100]:
            self.solver_code = 100
            self.solver = 'platen'
        elif self.solver in ['pred-corr', 'predictor-corrector', 'pc-euler', 101]:
            self.solver_code = 101
            self.solver = 'pred-corr'
        elif self.solver in ['milstein', 102, 1.0]:
            self.solver_code = 102
            self.solver = 'milstein'
        elif self.solver in ['milstein-imp', 103]:
            self.solver_code = 103
            self.solver = 'milstein-imp'
        elif self.solver in ['pred-corr-2', 'pc-euler-2', 104]:
            self.solver_code = 104
            self.solver = 'pred-corr-2'

        elif self.solver in ['platen15', 'explicit15', 150]:
            self.solver_code = 150
            self.solver = 'explicit15'
        elif self.solver in ['taylor15', None, 1.5, 152]:
            self.solver_code = 152
            self.solver = 'taylor15'
        elif self.solver in ['taylor15-imp', 153]:
            self.solver_code = 153
            self.solver = 'taylor15-imp'
        else:
            raise Exception("The solver should be one of "+\
                            "[None, 'euler-maruyama', 'platen', 'pc-euler', "+\
                            "'pc-euler-2', 'milstein', 'milstein-imp', "+\
                            "'taylor15', 'taylor15-imp', 'explicit15']")


def smesolve(H, rho0, times, c_ops=[], sc_ops=[], e_ops=[],
                 _safe_mode=True, args={}, **kwargs):
    """
    Solve stochastic master equation. Dispatch to specific solvers
    depending on the value of the `solver` keyword argument.

    Parameters
    ----------

    H : :class:`qutip.Qobj`
        System Hamiltonian.

    rho0 : :class:`qutip.Qobj`
        Initial density matrix or state vector (ket).

    times : *list* / *array*
        List of times for :math:`t`. Must be uniformly spaced.

    c_ops : list of :class:`qutip.Qobj`
        Deterministic collapse operator which will contribute with a standard
        Lindblad type of dissipation.

    sc_ops : list of :class:`qutip.Qobj`
        List of stochastic collapse operators. Each stochastic collapse
        operator will give a deterministic and stochastic contribution
        to the eqaution of motion according to how the d1 and d2 functions
        are defined.

    e_ops : list of :class:`qutip.Qobj`
        single operator or list of operators for which to evaluate
        expectation values.

    kwargs : *dictionary*
        Optional keyword arguments. See
        :class:`qutip.stochastic.StochasticSolverOptions`.

    Returns
    -------

    output: :class:`qutip.solver.SolverResult`

        An instance of the class :class:`qutip.solver.SolverResult`.

    """

    if isket(rho0):
        rho0 = ket2dm(rho0)

    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    sso = StochasticSolverOptions(H=H, state0=rho0, times=times, c_ops=c_ops,
                                  sc_ops=sc_ops, e_ops=e_ops, args= args,
                                  **kwargs)

    sso.me = True
    if _safe_mode:
        _safety_checks(sso)

    sso.LH = liouvillian(sso.H, c_ops = sso.sc_ops + sso.c_ops) * sso.dt
    #sso.d1 = 1 + sso.LH * sso.dt
    if sso.method == 'homodyne' or sso.method is None:
        if sso.m_ops is None:
            sso.m_ops = [op + op.dag() for op in sso.sc_ops]
        sso.sops = [spre(op) + spost(op.dag()) for op in sso.sc_ops]
        if not isinstance(sso.dW_factors, list):
            sso.dW_factors = [1] * len(sso.sops)
        elif len(sso.dW_factors) != len(sso.sops):
            raise Exception("The len of dW_factors is not the same as sc_ops")

    elif sso.method == 'heterodyne':
        if sso.m_ops is None:
            m_ops = []
        sso.sops = []
        for c in sso.sc_ops:
            if sso.m_ops is None:
                m_ops += [c + c.dag(), -1j * c - c.dag() ]
            sso.sops += [(spre(c) + spost(c.dag())) / np.sqrt(2),
                         (spre(c) - spost(c.dag())) * -1j / np.sqrt(2)]
        sso.m_ops = m_ops
        if not isinstance(sso.dW_factors, list):
            sso.dW_factors = [np.sqrt(2)] * len(sso.sops)
        elif len(sso.dW_factors) == len(sso.sc_ops):
            dW_factors = []
            for fact in sso.dW_factors:
                dW_factors += [np.sqrt(2) * fact, np.sqrt(2) * fact]
            sso.dW_factors = dW_factors
        elif len(sso.dW_factors) != len(sso.sops):
            raise Exception("The len of dW_factors is not the same as sc_ops")

    elif sso.method == "photocurrent":
        raise NotImplementedError("Not yet")

    else:
        raise Exception("The method must be one of None, homodyne, heterodyne")

    sso.ce_ops = [td_Qobj(spre(op)) for op in sso.e_ops]
    sso.cm_ops = [td_Qobj(spre(op)) for op in sso.m_ops]

    sso.LH.compile()
    [op.compile() for op in sso.sops]
    [op.compile() for op in sso.cm_ops]
    [op.compile() for op in sso.ce_ops]

    if sso.solver_code in [103, 153]:
        sso.imp = 1 - sso.LH * 0.5
        sso.imp.compile()

    sso.solver_obj = sme
    sso.solver_name = "smesolve_" + sso.solver

    res = _sesolve_generic(sso, sso.options, sso.progress_bar)

    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}

    return res

def ssesolve(H, rho0, times, sc_ops=[], e_ops=[],
                 _safe_mode=True, args={}, **kwargs):
    """
    Solve stochastic master equation. Dispatch to specific solvers
    depending on the value of the `solver` keyword argument.

    Parameters
    ----------

    H : :class:`qutip.Qobj`
        System Hamiltonian.

    rho0 : :class:`qutip.Qobj`
        State vector (ket).

    times : *list* / *array*
        List of times for :math:`t`. Must be uniformly spaced.

    sc_ops : list of :class:`qutip.Qobj`
        List of stochastic collapse operators. Each stochastic collapse
        operator will give a deterministic and stochastic contribution
        to the eqaution of motion according to how the d1 and d2 functions
        are defined.

    e_ops : list of :class:`qutip.Qobj`
        single operator or list of operators for which to evaluate
        expectation values.

    kwargs : *dictionary*
        Optional keyword arguments. See
        :class:`qutip.stochastic.StochasticSolverOptions`.

    Returns
    -------

    output: :class:`qutip.solver.SolverResult`

        An instance of the class :class:`qutip.solver.SolverResult`.
    """

    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    sso = StochasticSolverOptions(H=H, state0=rho0, times=times, sc_ops=sc_ops,
                                  e_ops=e_ops, args= args, **kwargs)

    sso.me = False
    if _safe_mode:
        _safety_checks(sso)

    if sso.method == 'homodyne' or sso.method is None:
        if sso.m_ops is None:
            sso.m_ops = [op + op.dag() for op in sso.sc_ops]
        #sso.sops = [[op, -op.norm() *sso.dt/2, op + op.dag()] for op in sso.sc_ops]
        sso.sops = [[op, op + op.dag()] for op in sso.sc_ops]
        if not isinstance(sso.dW_factors, list):
            sso.dW_factors = [1] * len(sso.sops)
        elif len(sso.dW_factors) != len(sso.sops):
            raise Exception("The len of dW_factors is not the same as sc_ops")

    elif sso.method == 'heterodyne':
        if sso.m_ops is None:
            m_ops = []
        sso.sops = []
        for c in sso.sc_ops:
            if sso.m_ops is None:
                m_ops += [c + c.dag(), -1j * c - c.dag() ]
            if c.const:
                c1 = (c + c.dag()) / np.sqrt(2)*0.5
                c2 = (c - c.dag()) * (-1j / np.sqrt(2))*0.5
            else:
                # Not clean, should have a way to compress for a common coeff
                op = c.to_list()[0][0]
                f = c.to_list()[0][1]
                op1 = (op + op.dag()) / np.sqrt(2)*0.5
                c1 = td_Qobj([op1,f], args=args, tlist=times, raw_str=True)
                op2 = (op - op.dag()) * (-1j / np.sqrt(2))*0.5
                c2 = td_Qobj([op2,f], args=args, tlist=times, raw_str=True)
            sso.sops += [[c1, c1 + c1.dag()],
                         [c2, c2 + c2.dag()]]
        sso.m_ops = m_ops
        if not isinstance(sso.dW_factors, list):
            sso.dW_factors = [np.sqrt(2)] * len(sso.sops)
        elif len(sso.dW_factors) == len(sso.sc_ops):
            dW_factors = []
            for fact in sso.dW_factors:
                dW_factors += [np.sqrt(2) * fact, np.sqrt(2) * fact]
            sso.dW_factors = dW_factors
        elif len(sso.dW_factors) != len(sso.sops):
            raise Exception("The len of dW_factors is not the same as sc_ops")

    elif sso.method == "photocurrent":
        raise NotImplementedError("Not yet")

    else:
        raise Exception("The method must be one of None, homodyne, heterodyne")

    sso.LH = sso.H * (-1j*sso.dt )
    for ops in sso.sops:
        sso.LH -= ops[0].norm()*0.5*sso.dt

    sso.ce_ops = [td_Qobj(op) for op in sso.e_ops]
    sso.cm_ops = [td_Qobj(op) for op in sso.m_ops]

    sso.LH.compile()
    [[op.compile() for op in ops] for ops in sso.sops]
    [op.compile() for op in sso.cm_ops]
    [op.compile() for op in sso.ce_ops]

    sso.solver_obj = sse
    sso.solver_name = "ssesolve_" + sso.solver

    res = _sesolve_generic(sso, sso.options, sso.progress_bar)

    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}

    return res


def photocurrentmesolve(H, rho0, times, c_ops=[], sc_ops=[], e_ops=[],
                 _safe_mode=True, args={}, **kwargs):
    """
    Solve stochastic master equation using the photocurrent method.

    Parameters
    ----------

    H : :class:`qutip.Qobj`
        System Hamiltonian.

    rho/psi : :class:`qutip.Qobj`
        Initial density matrix or state vector (ket).

    times : *list* / *array*
        List of times for :math:`t`. Must be uniformly spaced.

    c_ops : list of :class:`qutip.Qobj`
        If master equation,
        Deterministic collapse operator which will contribute with a standard
        Lindblad type of dissipation.

    sc_ops : list of :class:`qutip.Qobj`
        List of stochastic collapse operators. Each stochastic collapse
        operator will give a deterministic and stochastic contribution
        to the eqaution of motion according to how the d1 and d2 functions
        are defined.

    e_ops : list of :class:`qutip.Qobj` / callback function single
        single operator or list of operators for which to evaluate
        expectation values.

    kwargs : *dictionary*
        Optional keyword arguments. See
        :class:`qutip.stochastic.StochasticSolverOptions`.

    Returns
    -------

    output: :class:`qutip.solver.SolverResult`

        An instance of the class :class:`qutip.solver.SolverResult`.
    """
    if isket(rho0):
        rho0 = ket2dm(rho0)

    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    sso = StochasticSolverOptions(H=H, state0=rho0, times=times,
                                  c_ops=c_ops, sc_ops=sc_ops, e_ops=e_ops,
                                  args=args, **kwargs)
    sso.solver_code = 60
    sso.me = True
    if _safe_mode:
        _safety_checks(sso)

    if sso.m_ops is None:
        sso.m_ops = [op + op.dag() for op in sso.sc_ops]
    if not isinstance(sso.dW_factors, list):
        sso.dW_factors = [1] * len(sso.sc_ops)
    elif len(sso.dW_factors) != len(sso.sc_ops):
        raise Exception("The len of dW_factors is not the same as sc_ops")

    sso.solver_obj = psme
    sso.solver_name = "photocurrent_mesolve_" + sso.solver
    sso.LH = liouvillian(sso.H, c_ops = sso.c_ops) * sso.dt
    def _prespostdag(op):
        return spre(op) * spost(op.dag())
    sso.sops = [[spre(op.norm()) + spost(op.norm()),
                 spre(op.norm()),
                 op.apply(_prespostdag)._f_norm2()] for op in sso.sc_ops]
    sso.ce_ops = [td_Qobj(spre(op)) for op in sso.e_ops]
    sso.cm_ops = [td_Qobj(spre(op)) for op in sso.m_ops]

    sso.LH.compile()
    [[op.compile() for op in ops] for ops in sso.sops]
    [op.compile() for op in sso.cm_ops]
    [op.compile() for op in sso.ce_ops]

    res = _sesolve_generic(sso, sso.options, sso.progress_bar)

    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}

    return res

def photocurrentsesolve(H, rho0, times, sc_ops=[], e_ops=[],
                 _safe_mode=True, args={}, **kwargs):
    """
    Solve stochastic schrodinger equation using the photocurrent method.

    Parameters
    ----------

    H : :class:`qutip.Qobj`
        System Hamiltonian.

    rho/psi : :class:`qutip.Qobj`
        Initial density matrix or state vector (ket).

    times : *list* / *array*
        List of times for :math:`t`. Must be uniformly spaced.

    sc_ops : list of :class:`qutip.Qobj`
        List of stochastic collapse operators. Each stochastic collapse
        operator will give a deterministic and stochastic contribution
        to the eqaution of motion according to how the d1 and d2 functions
        are defined.

    e_ops : list of :class:`qutip.Qobj` / callback function single
        single operator or list of operators for which to evaluate
        expectation values.

    kwargs : *dictionary*
        Optional keyword arguments. See
        :class:`qutip.stochastic.StochasticSolverOptions`.

    Returns
    -------

    output: :class:`qutip.solver.SolverResult`

        An instance of the class :class:`qutip.solver.SolverResult`.
    """
    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    sso = StochasticSolverOptions(H=H, state0=rho0, times=times,
                                  c_ops=c_ops, sc_ops=sc_ops, e_ops=e_ops,
                                  args=args, **kwargs)
    sso.solver_code = 60
    sso.me = False
    if _safe_mode:
        _safety_checks(sso)

    if sso.m_ops is None:
        sso.m_ops = [op + op.dag() for op in sso.sc_ops]
    if not isinstance(sso.dW_factors, list):
        sso.dW_factors = [1] * len(sso.sc_ops)
    elif len(sso.dW_factors) != len(sso.sc_ops):
        raise Exception("The len of dW_factors is not the same as sc_ops")

    sso.solver_obj = psse
    sso.solver_name = "photocurrent_sesolve_" + sso.solver
    sso.sops = [[op, op.norm()] for op in sso.sc_ops]
    sso.LH = sso.H * (-1j*sso.dt )
    for ops in sso.sops:
        sso.LH -= ops[0].norm()*0.5*sso.dt
    sso.ce_ops = [td_Qobj(op) for op in sso.e_ops]
    sso.cm_ops = [td_Qobj(op) for op in sso.m_ops]

    sso.LH.compile()
    [[op.compile() for op in ops] for ops in sso.sops]
    [op.compile() for op in sso.cm_ops]
    [op.compile() for op in sso.ce_ops]

    res = _sesolve_generic(sso, sso.options, sso.progress_bar)

    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}

    return res


def general_stochastic(state0, times, d1, d2, e_ops=[],
               _safe_mode=True, len_d2=1, args={}, **kwargs):
    """
    Solve stochastic general equation. Dispatch to specific solvers
    depending on the value of the `solver` keyword argument.

    Parameters
    ----------

    state0 : :class:`qutip.Qobj`
        Initial state vector (ket) or density matrix as a vector.

    times : *list* / *array*
        List of times for :math:`t`. Must be uniformly spaced.

    d1 : function, callable class
        Function representing the deterministic evolution of the system.

        def d1(time (double), state (as a np.array vector)):
            return 1d np.array

    d2 : function, callable class
        Function representing the stochastic evolution of the system.

        def d2(time (double), state (as a np.array vector)):
            return 2d np.array (N_sc_ops, len(state0))

    len_d2 : int
        Number of output vector produced by d2

    e_ops : list of :class:`qutip.Qobj`
        single operator or list of operators for which to evaluate
        expectation values.
        Must be a superoperator if the state vector is a density matrix.

    kwargs : *dictionary*
        Optional keyword arguments. See
        :class:`qutip.stochastic.StochasticSolverOptions`.

    Returns
    -------

    output: :class:`qutip.solver.SolverResult`
        An instance of the class :class:`qutip.solver.SolverResult`.
    """

    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    if not "solver" in kwargs:
        kwargs["solver"] = 50

    sso = StochasticSolverOptions(H=None, state0=state0, times=times,
                                  e_ops=e_ops, args=args, **kwargs)
    if sso.solver_code not in [50, 100, 150]:
        raise Exception("Only Euler, platen, platen15 can be " +
                        "used for the general stochastic solver")

    sso.me = False
    sso.d1 = d1
    sso.d2 = d2
    if _safe_mode:
        l_vec = sso.rho0.shape[0]
        try:
            out_d1 = d1(0., sso.rho0)
        except Exception as e:
            raise Exception("d1(0., mat2vec(state0.full()).ravel()) failed:\n"+\
                            e)
        except:
            raise Exception("d1(0., mat2vec(state0.full()).ravel()) failed")
        try:
            out_d2 = d2(0., sso.rho0)
        except Exception as e:
            raise Exception("d2(0., mat2vec(state0.full()).ravel()) failed:\n"+\
                            e)
        except:
            raise Exception("d2(0., mat2vec(state0.full()).ravel()) failed")
        if out_d1.shape[0] != l_vec or len(out_d1.shape) != 1:
            raise Exception("d1 must return an 1d numpy array with "+\
                            "the same number of element than the " +\
                            "initial state as a vector")
        if len(out_d2.shape) != 2 and out_d2.shape[1] != l_vec and \
            out_d2.shape[0] != len_d2:
            raise Exception("d2 must return an 2d numpy array with the shape "+\
                            "(l2_len, len(mat2vec(state0.full()).ravel()) )")
        if out_d1.dtype != np.dtype('complex128') or \
           out_d2.dtype != np.dtype('complex128') :
            raise Exception("d1 and d2 must return complex numpy array")
        for op in sso.e_ops:
            shape_op = op.shape
            if sso.me:
                if shape_op[0]**2 != l_vec or shape_op[1]**2 != l_vec:
                    raise Exception("The size of the e_ops does not fit the intial state")
            else:
                if shape_op[0] != l_vec or shape_op[1] != l_vec:
                    raise Exception("The size of the e_ops does not fit the intial state")


    if sso.store_measurement:
        raise Exception("General stochastic solver cannot store measurement")
    sso.m_ops = []
    sso.cm_ops = []
    sso.dW_factors = [1.] * len_d2
    sso.sops = [None] * len_d2
    sso.ce_ops = [td_Qobj(op) for op in sso.e_ops]
    [op.compile() for op in sso.ce_ops]

    sso.solver_obj = generic
    sso.solver_name = "general_stochastic_solver_" + sso.solver

    ssolver = generic()
    ssolver.set_data(sso)
    ssolver.set_solver(sso)

    res = _sesolve_generic(sso, sso.options, sso.progress_bar)

    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}

    return res


def _safety_checks(sso):
    l_vec = sso.rho0.shape[0]
    if sso.H.cte.issuper:
        if not sso.me:
            raise
        shape_op = sso.H.cte.shape
        if shape_op[0] != l_vec or shape_op[1] != l_vec:
            raise Exception("The size of the hamiltonian does not fit the intial state")
    else:
        shape_op = sso.H.cte.shape
        if sso.me:
            if shape_op[0]**2 != l_vec or shape_op[1]**2 != l_vec:
                raise Exception("The size of the hamiltonian does not fit the intial state")
        else:
            if shape_op[0] != l_vec or shape_op[1] != l_vec:
                raise Exception("The size of the hamiltonian does not fit the intial state")

    for op in sso.sc_ops:
        if op.cte.issuper:
            if not sso.me:
                raise
            shape_op = op.cte.shape
            if shape_op[0] != l_vec or shape_op[1] != l_vec:
                raise Exception("The size of the sc_ops does not fit the intial state")
        else:
            shape_op = op.cte.shape
            if sso.me:
                if shape_op[0]**2 != l_vec or shape_op[1]**2 != l_vec:
                    raise Exception("The size of the sc_ops does not fit the intial state")
            else:
                if shape_op[0] != l_vec or shape_op[1] != l_vec:
                    raise Exception("The size of the sc_ops does not fit the intial state")

    for op in sso.c_ops:
        if op.cte.issuper:
            if not sso.me:
                raise
            shape_op = op.cte.shape
            if shape_op[0] != l_vec or shape_op[1] != l_vec:
                raise Exception("The size of the c_ops does not fit the intial state")
        else:
            shape_op = op.cte.shape
            if sso.me:
                if shape_op[0]**2 != l_vec or shape_op[1]**2 != l_vec:
                    raise Exception("The size of the c_ops does not fit the intial state")
            else:
                if shape_op[0] != l_vec or shape_op[1] != l_vec:
                    raise Exception("The size of the c_ops does not fit the intial state")

    for op in sso.e_ops:
        shape_op = op.shape
        if sso.me:
            if shape_op[0]**2 != l_vec or shape_op[1]**2 != l_vec:
                raise Exception("The size of the e_ops does not fit the intial state")
        else:
            if shape_op[0] != l_vec or shape_op[1] != l_vec:
                raise Exception("The size of the e_ops does not fit the intial state")

    if sso.m_ops is not None:
        for op in sso.m_ops:
            shape_op = op.shape
            if sso.me:
                if shape_op[0]**2 != l_vec or shape_op[1]**2 != l_vec:
                    raise Exception("The size of the m_ops does not fit the intial state")
            else:
                if shape_op[0] != l_vec or shape_op[1] != l_vec:
                    raise Exception("The size of the m_ops does not fit the intial state")

def _sesolve_generic(sso, options, progress_bar):
    """
    Internal function. See smesolve.   Not Good yet ------------------------------------------
    """
    data = Result()
    data.times = sso.times
    data.expect = np.zeros((len(sso.e_ops), len(sso.times)), dtype=complex)
    data.ss = np.zeros((len(sso.e_ops), len(sso.times)), dtype=complex)
    data.noise = []
    data.measurement = []
    data.solver = sso.solver_name

    nt = sso.ntraj
    task = _single_trajectory
    map_kwargs = {'progress_bar': sso.progress_bar}
    map_kwargs.update(sso.map_kwargs)
    task_args = (sso,)
    task_kwargs = {}

    results = sso.map_func(task, list(range(sso.ntraj)),
                           task_args, task_kwargs, **map_kwargs)

    for result in results:
        states_list, dW, m, expect, ss = result
        data.states.append(states_list)
        data.noise.append(dW)
        data.measurement.append(m)
        data.expect += expect
        data.ss += ss

    # average density matrices
    if options.average_states and np.any(data.states):
        data.states = [sum([data.states[mm][n] for mm in range(nt)]).unit()
                       for n in range(len(data.times))]

    # average
    data.expect = data.expect / nt

    # standard error
    if nt > 1:
        data.se = (data.ss - nt * (data.expect ** 2)) / (nt * (nt - 1))
    else:
        data.se = None

    # convert complex data to real if hermitian
    data.expect = [np.real(data.expect[n, :])
                   if e.isherm else data.expect[n, :]
                   for n, e in enumerate(sso.e_ops)]

    return data


def _single_trajectory(i, sso):
    ssolver = sso.solver_obj()
    ssolver.set_data(sso)
    ssolver.set_solver(sso)
    result = ssolver.cy_sesolve_single_trajectory(i, sso)
    return result
