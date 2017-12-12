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
from qutip.cy.stochastic_solver import sme
from qutip.qobj import Qobj, isket, isoper, issuper
from qutip.states import ket2dm
from qutip.solver import Result
from qutip.td_qobj import td_Qobj
from qutip.superoperator import (spre, spost, mat2vec, vec2mat,
                                 liouvillian, lindblad_dissipator)
from qutip.solver import Options, _solver_safety_check
from qutip.parallel import serial_map
from qutip.ui.progressbar import TextProgressBar
from qutip.settings import debug

#from scipy.linalg.blas import get_blas_funcs
#try:
#    norm = get_blas_funcs("znrm2", dtype=np.float64)
#except:
#    from scipy.linalg import norm

#from numpy.random import RandomState
#
#from scipy.sparse.linalg import LinearOperator
#from scipy.linalg.blas import zaxpy

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
        1/2 order algorithms: 'euler-maruyama', 'fast-euler-maruyama',
        'pc-euler' is a predictor-corrector method which is more
        stable than explicit methods,
        1 order algorithms: 'milstein', 'fast-milstein', 'platen',
        'milstein-imp' is semi-implicit Milstein method,
        3/2 order algorithms: 'taylor15',
        'taylor15-imp' is semi-implicit Taylor 1.5 method.
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
        self.H = td_Qobj(H, args=args, tlist=times, raw_str=True)
        self.sc_ops = [td_Qobj(op, args=args, tlist=times,
                                 raw_str=True) for op in sc_ops]
        self.c_ops = [td_Qobj(op, args=args, tlist=times,
                                raw_str=True) for op in c_ops]
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
        if generate_noise is not None:
            self.noise_type = 2
            self.generate_noise = generate_noise

        elif noise is not None:
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
                dw_len = (" * 2" if method == "heterodyne" else "")
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


def another_smesolve(H, rho0, times, c_ops=[], sc_ops=[], e_ops=[],
                 _safe_mode=True, debug=False, args={}, **kwargs):
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

    TODO
    ----
        Add checks for commuting jump operators in Milstein method.
    """

    if debug:
        logger.debug(inspect.stack()[0][3])

    if isket(rho0):
        rho0 = ket2dm(rho0)

    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None

    if _safe_mode:
        pass ###----------------------------------------------------------------------------------------------------------

    sso = StochasticSolverOptions(H=H, state0=rho0, times=times, c_ops=c_ops,
                                  sc_ops=sc_ops, e_ops=e_ops, args= args,
                                  **kwargs)

    sso.me = True

    sso.LH = liouvillian(sso.H, c_ops = sso.sc_ops + sso.c_ops) * sso.dt
    #sso.d1 = 1 + sso.LH * sso.dt
    if sso.method == 'homodyne' or sso.method is None:
        if sso.m_ops is None:
            sso.m_ops = [op + op.dag() for op in sc_ops]
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

    if sso.solver in ['euler-maruyama', 'euler', None, 50, 0.5]:
        sso.solver_code = 50

    elif sso.solver in ['platen', 'platen1', 'platen1.0', 100, 1.0]:
        sso.solver_code = 100

    elif sso.solver in ['pred-corr', 'predictor-corrector', 'pc-euler', 101]:
        sso.solver_code = 101

    elif sso.solver in ['milstein', 102]:
        sso.solver_code = 102

    elif sso.solver in ['milstein-imp', 103]:
        sso.solver_code = 103

    if sso.solver_code is None:
        raise Exception("The solver should be one of "+\
                        "[None, 'euler-maruyama', "+\
                        "'milstein', 'platen', 'taylor15', "+\
                        "'milstein-imp', 'taylor15-imp', 'pc-euler']")

    res = _smesolve_generic(sso, sso.options, sso.progress_bar)

    if e_ops_dict:
        res.expect = {e: res.expect[n]
                      for n, e in enumerate(e_ops_dict.keys())}

    return res

def _smesolve_generic(sso, options, progress_bar):
    """
    Internal function. See smesolve.   Not Good yet ------------------------------------------
    """
    if debug:
        logger.debug(inspect.stack()[0][3])

    data = Result()
    data.solver = "smesolve"
    data.times = sso.times
    data.expect = np.zeros((len(sso.e_ops), len(sso.times)), dtype=complex)
    data.ss = np.zeros((len(sso.e_ops), len(sso.times)), dtype=complex)
    data.noise = []
    data.measurement = []

    ssolver = sme()
    #print(sso.d1.rhs(0,sso.rho0))
    ssolver.set_data(sso.LH, sso.sops)
    ssolver.set_solver(sso)

    nt = sso.ntraj

    task = ssolver.cy_sesolve_single_trajectory

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
