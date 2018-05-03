# -*- coding: utf-8 -*-
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2014 and later, Alexander J G Pitchford
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

# @author: Alexander Pitchford
# @author: Eric Giguère


import os
import warnings
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
# QuTiP
from qutip import Qobj
from qutip.sparse import sp_eigs, _dense_eigs
import qutip.settings as settings
# QuTiP logging
import qutip.logging_utils as logging
logger = logging.get_logger()
# QuTiP control modules
import qutip.control.errors as errors
import qutip.control.tslotcomp as tslotcomp
import qutip.control.fidcomp as fidcomp
import qutip.control.propcomp as propcomp
import qutip.control.symplectic as sympl
import qutip.control.dump as qtrldump

DEF_NUM_TSLOTS = 10
DEF_EVO_TIME = 1.0





class dynamics:
    """
    This class compute the error and gradient for the GRAPE systems.*

    * Other object do the actual computation to cover the multiple situations.

    methods:
        cost(x):
            x: np.ndarray, state of the pulse
            return 1-fidelity

        gradient(x):
            x: np.ndarray, state of the pulse
            return the error gradient

        control_system():


    """

    def __init__(self, initial, target, H, ctrl,
                 times=None, tau=None, T=0, t_step=0, _num_x=0, _filter=None,
                 phase,
                 prop_method = None,
                 **kwarg):
        self.initial = initial      # Initial state/rho/operator as Qobj
        self.target = target        # Target state/rho/operator as Qobj
        self.drift_dyn_gen = H      # Hamiltonians or Liouvillian as a Qobj
        self.ctrl_dyn_gen = ctrl    # Control operator [Qobj]
        self._num_ctrls = len(ctrl)

        if _filter in None:
            self.filter = filters.pass_througth()
        else:
            self.filter = _filter

        self._x_shape, self.time = self.filter.init_timeslots(times, tau, T,
                                                    t_step, _num_x, _num_ctrls)
        self._num_tslots = len(self.time)-1
        self._evo_time = self.time[-1]
        if np.allclose(np.diff(self.time), self.time[1]-self.time[0]):
            self._tau = self.time[1]-self.time[0]
        else:
            self._tau = np.diff(self.time)
        # state and gradient before filter
        self._x = np.zeros(self._x_shape)
        self.gradient_x = np.zeros(self._x_shape)
        # state and gradient after filter
        self._ctrl_amps = np.zeros((self._num_tslots, self._num_ctrls))
        self._gradient_u = np.zeros((self._num_tslots, self._num_ctrls))

        self._set_memory_optimizations(**kwarg)

        if isinstance(initial, np.ndarray):
            self._initial = initial
            self._target = target
        elif isinstance(initial, Qobj) and initial.isoper:
            self._initial = matrice(initial, dense=self.oper_dense)
            self._target = matrice(target, dense=self.oper_dense)
        else:
            self._initial = np.ndarray(initial.data)
            self._target = np.ndarray(target.data)

        if not self.cache_drift_at_T:
            self._drift_dyn_gen = H # ----- Not implemented yet ----- ?
        else:
            if not H.const:
                self._drift_dyn_gen = [matrice(H(t), dense=self.oper_dense)
                                       for t in self.time]
            else:
                self._drift_dyn_gen = np.ndarray(
                                    [matrice(H, dense=self.oper_dense)])
            for mat in self._drift_dyn_gen:
                mat.method = prop_method
                mat.fact_mat_round_prec = fact_mat_round_prec
                mat._mem_eigen_adj = self.cache_dyn_gen_eigenvectors_adj
                mat._mem_prop = self.cache_prop
                mat.epsilon = epsilon
                if self.cache_phased_dyn_gen:
                    mat = self._apply_phase(mat)

        if not self.cache_ctrl_at_T:
            self._ctrl_dyn_gen = ctrl # ----- Not implemented yet ----- ?
        elif all((ctr.const for ctr in ctrl)):
            self._ctrl_dyn_gen = np.ndarray(
                                    [[matrice(ctr(0), dense=self.oper_dense)]
                                    for ctr in ctrl])
        else:
            self._ctrl_dyn_gen = np.ndarray(
                                    [[matrice(ctr(t), dense=self.oper_dense)
                                   for t in self.time] for ctr in ctrl]

        if _tslotcomp is None:
            self.tslotcomp =  tslotcomp.TSlotCompUpdateAll(self)
        else:
            self.tslotcomp = _tslotcomp
        self.tslotcomp.set(self)

        if _fidcomp is None:
            self.costcomp =  fidcomp.FidCompTraceDiff(self)
        elif isinstance(_fidcomp, list):
            self.costcomp = _fidcomp
        else:
            self.costcomp = [_fidcomp]
        for cost in self.costcomp:
            cost.set(self)

        # computation objects
        self._dyn_gen = []         # S[t] = H_0[t] + u_i[t]*H_i
        self._prop = []            # U[t] exp(-i*S[t])
        self._prop_grad = [[]]     # d U[t] / du_i
        self._fwd_evo = []         # U[t]U[t-dt]...U[dt]U[0] /initial
        self._onwd_evo = []        # /target U[T]U[T-dt]...U[t+dt]U[t]


        # These internal attributes will be of the internal operator data type
        # used to compute the evolution
        # Note this maybe ndarray, Qobj or some other depending on oper_dtype

        # self._phased_ctrl_dyn_gen = None
        # self._dyn_gen_phase = None
        # self._phase_application = None
        # self._phased_dyn_gen = None
        # self._onto_evo_target = None
        # self._onto_evo = None

    def _set_memory_optimizations(self, memory_optimization=0,
                                  cache_dyn_gen_eigenvectors_adj=None,
                                  cache_phased_dyn_gen=None,
                                  sparse_eigen_decomp=None,
                                  cache_drift_at_T=None,
                                  cache_prop_grad=None,
                                  cache_ctrl_at_T=None,
                                  cache_prop=None,
                                  oper_dtype=None,
                                  **kwarg):
        """
        Set various memory optimisation attributes based on the
        memory_optimization attribute.
        """

        if oper_dtype is None:
            self._choose_oper_dtype()
        else self.oper_dtype = oper_dtype

        if cache_phased_dyn_gen is None:
            self.cache_phased_dyn_gen = memory_optimization == 0
        else:
            self.cache_phased_dyn_gen = cache_phased_dyn_gen

        if cache_prop_grad is None:
            iself.cache_prop_grad = memory_optimization == 0
        else:
            self.cache_prop_grad = cache_prop_grad

        if cache_dyn_gen_eigenvectors_adj is None:
            self.cache_dyn_gen_eigenvectors_adj = memory_optimization == 0
        else:
            self.cache_dyn_gen_eigenvectors_adj = cache_dyn_gen_eigenvectors_adj

        if sparse_eigen_decomp is None:
            self.sparse_eigen_decomp = memory_optimization > 1
        else:
            self.sparse_eigen_decomp = sparse_eigen_decomp

        # If the drift operator depends on time, precompute for each t
        if cache_drift_at_T is None:
            self.cache_drift_at_T = True # memory_optimization > 1
        else:
            self.cache_drift_at_T = cache_drift_at_T

        # If one of the ctrl operators depend on time, precompute for each t
        if cache_ctrl_at_T is None:
            self.cache_ctrl_at_T = True # memory_optimization > 1
        else:
            self.cache_ctrl_at_T = cache_drift_at_T

        if cache_prop is None:
            self.cache_prop = memory_optimization == 0
        else:
            self.cache_prop = cache_prop


    def clean(self):
        """Remove object saved but not used during computation."""
        pass


    ### -------------------------- Computation part ---------------------------
    def error(self, x):
        if not np.allclose(self.x_ == x):
            self._compute_state(x)
        return self.error

    def gradient(self, x):
        if not np.allclose(self.x_ == x):
            self._compute_state(x)
        return self.gradient_x

    def _compute_state(self, x):
        """For a state x compute the cost and grandient"""
        self.x_ = x
        self._ctrl_amps = self.filter(x)
        self.tslotcomp.set(self._ctrl_amps)
        for costs in self.costcomp:
            error, gradient_u_cost = costs()
            self.error += error
            gradient_u += gradient_u_cost
        self.gradient_x = self.filter.reverse(gradient_u)

    def _apply_phase(self, dg):
        return self._prephase * dg * self._postphase






def _is_unitary(self, A):
        """
        Checks whether operator A is unitary
        A can be either Qobj or ndarray
        """
        if isinstance(A, Qobj):
            unitary = np.allclose(np.eye(A.shape[0]), A*A.dag().full(),
                        atol=self.unitarity_tol)
        else:
            unitary = np.allclose(np.eye(len(A)), A.dot(A.T.conj()),
                        atol=self.unitarity_tol)

        return unitary

def _calc_unitary_err(self, A):
        if isinstance(A, Qobj):
            err = np.sum(abs(np.eye(A.shape[0]) - A*A.dag().full()))
        else:
            err = np.sum(abs(np.eye(len(A)) - A.dot(A.T.conj())))

        return err

def unitarity_check(self):
        """
        Checks whether all propagators are unitary
        """
        for k in range(self.num_tslots):
            if not self._is_unitary(self._prop[k]):
