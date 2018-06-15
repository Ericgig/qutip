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




from qutip.qobj import Qobj
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp


#import qutip.control.tslotcomp as tslotcomp
#import qutip.control.fidcomp as fidcomp
#import qutip.control.filters as filters
#import qutip.control.matrix as matrix
#import qutip.control.optimize as optimize


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

    def __init__(self):
        self.ready = False
        self.costs = []
        self.other_cost = []

    def set_physic(self, H, ctrls, initial, target):
        self.drift_dyn_gen = H      # Hamiltonians or Liouvillian as a Qobj or td_Qobj
        if isinstance(H, Qobj):
            self.issuper = H.issuper
            self.dims = H.dims
            self.shape = H.shape
        elif isinstance(H, td_Qobj):
            self.issuper = H.cte.issuper
            self.dims = H.cte.dims
            self.shape = H.cte.shape
        else:
            raise Exception("The drift operator is expected to be a Qobj or td_Qobj.")
        if len(self.shape) != 2 or self.shape[0] != self.shape[1]:
            raise Exception("Invalid drift operator.")

        if isinstance(ctrls, (Qobj, td_Qobj)):
            ctrls = [ctrls]
        self.ctrl_dyn_gen = ctrls    # Control operator [Qobj] or [td_Qobj]
        self._num_ctrls = len(ctrls)
        for ctrl in ctrls:
            if not isinstance(ctrl, (Qobj, td_Qobj)):
                raise Exception("The ctrls operators are expected to be a list of Qobj or td_Qobj.")
            if isinstance(ctrl, td_Qobj):
                c = c.cte
            else:
                c = ctrl
            if not self.dims == c.dims:
                raise Exception("Drift and ctrls operators dimensions do not match.")

        if isinstance(initial, Qobj):
            if initial.dims == self.dims[1]:
                self.initial = initial
                self.state_evolution = True
            elif initial.dims == self.dims:
                self.initial = initial
                self.state_evolution = False
            else:
                raise Exception("Dims of the initial state and Drift operator do not match.")
        else:
            raise Exception("The initial state is expected to be a Qobj.")

        if not isinstance(target, Qobj):
            raise Exception("The target state is expected to be a Qobj.")
        else:
            if not initial.dims == target.dims:
                raise Exception("Dims of the target state and initial state do not match.")
            self.target = target

    def set_filter(self, _filter, **kwargs):
        if isinstance(_filter, str):
            if _filter == "fourrier":
                self.filter = filters.fourrier(**kwargs)
            if _filter == "spline":
                self.filter = filters.spline(**kwargs)
            if _filter == "convolution":
                self.filter = filters.convolution(**kwargs)
        elif isinstance(_filter, filters.filter):
            self.filter = _filter

    def set_times(self, times=None, tau=None, T=0, t_step=0, _num_x=0):
        self.t_data = (times, tau, T, t_step, _num_x)

    def optimization(self, mode="auto", mem=0, _tslotcomp=None):
        if mode in ["auto","mem","speed","precision"]:
            self.mode = mode

        if mem and isinstance(mem, (int, float)):
            self.mem = mem
        else:
            #Not sure to want to get here yet
            import psutil
            self.mem = psutil.virtual_memory().free/1024/1024

        if _tslotcomp is not None:
            self.tslotcomp = _tslotcomp

    def set_cost(self, mode=None, early=False, weight=None):
        """
        From the dimension of the state and drift operator, the
        family of fidelity computer is determined.

        mode :: str
            Tag of the fidelity computation
            Unitary-state:
                "SU":   real(<target,final>)
                "PSU":  abs(<target,final>)
                "PSU2": abs(<target,final>)**2
        ...

        early :: bool, [int]
            Solve for solution that converge early (slower)

            False: Only compute the fidelity of the resulting state.
            True: Sum the fidelity at every timestep. -- set by the filter
            list: Sum the fidelity at desired timestep.

        weight :: number, [number]
            Weight of the fidelity or list of weights for each timestep if early
        """


        if not self.issuper and self.state_evolution:
            # Unitary state evolution
            if mode is None:
                mode = "PSU"
            if not mode in ["SU","PSU","PSU2"]:
                raise Exception("mode must be one of 'SU','PSU','PSU2' for unitary state evolution")
            self.mode = mode

        if early:
            self.early = True
            self.early_times = []
            if isinstance(early, (list, np.array)):
                self.early_times = early

        if isinstance(weight, (int,float)):
            self.weight = weight
        elif isinstance(weight, (list, np.array)):
            if not len(self.early_time)==len(weight):
                self.weight = weight
            else:
                raise Exception("The number of weight do not match the numbers of times")
        else:
            raise Exception("weight should be a real or list of real.")

    def _add_cost(self, cost_comp=None):
        self.other_cost.append(cost_comp)

    def option(self, **kwargs):
        for k,v in kwargs.items():
            if k in self.option:
                self.option[k] = v

    def prepare(self):
        if _filter in None:
            self.filter = filters.pass_througth()
        else:
            self.filter = _filter

        self._x_shape, self.time = self.filter.init_timeslots(*self.t_data)

        self._num_tslots = len(self.time)-1
        self._evo_time = self.time[-1]
        if np.allclose(np.diff(self.time), self.time[1]-self.time[0]):
            self._tau = matrix.falselist_cte(self.time[1]-self.time[0])
        else:
            self._tau = np.diff(self.time)

        # state and gradient before filter
        self._x = np.zeros(self._x_shape)
        self.gradient_x = np.zeros(self._x_shape)
        # state and gradient after filter
        self._ctrl_amps = np.zeros((self._num_tslots, self._num_ctrls))
        self._gradient_u = np.zeros((self._num_tslots, self._num_ctrls))

        if self.tslotcomp is None:
            matrix
            drift = self.drift_dyn_gen
            ctrls = self.ctrl_dyn_gen
            initial = self.initial
            target = self.target
            self.tslotcomp = tslotcomp.TSlotCompUpdateAll(drift, ctrls, initial, target,
                                                          self._tau, self._num_tslots,
                                                          self._num_ctrls)

        if self.costcomp is None:
            if self.state_evolution and not self.issuper:
                if self.early:
                    self.costcomp = [fidcomp.FidCompUnitaryEarly(self.tslotcomp, self.target, self.mode,
                                                                self.early_times, self.weight)]
                else:
                    self.costcomp = [fidcomp.FidCompUnitary(self.tslotcomp, self.target, self.mode)]
            else:
                raise NotImplementedError()

        if self.other_cost:
            self.costcomp += self.other_cost

        if not self.x0:
            self.x0 = np.random.rand(self._x_shape)

        self.solver = optimize.Optimizer(self._error, self._gradient, self.x0)

    def run(self):

        result = ...
        self.solver.run(result)


    ### -------------------------- Computation part ---------------------------
    def _error(self, x):
        if not np.allclose(self.x_ == x):
            self._compute_state(x)
        return self.error

    def _gradient(self, x):
        if not np.allclose(self.x_ == x):
            if not self.gradient_x:
                self._compute_grad()
        return self.gradient_x

    def _compute_state(self, x):
        """For a state x compute the cost"""
        self.x_ = x
        self._ctrl_amps = self.filter(x)
        self.tslotcomp.set(self._ctrl_amps)
        self.error = 0.
        for costs in self.costcomp:
            error = costs.costs()
            self.error += error

    def _compute_grad(self):
        """For a state x compute the grandient"""
        gradient_u = np.zeros((self._num_tslots, self._num_ctrls))
        for costs in self.costcomp:
            gradient_u_cost = costs.grad()
            gradient_u += gradient_u_cost
        self.gradient_x = self.filter.reverse(gradient_u)

    """def _apply_phase(self, dg):
        return self._prephase * dg * self._postphase"""













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

    def __init__(self):
        pass


    def set_physic(self, H, ctrl, initial, target=None):
        self.drift_dyn_gen = H      # Hamiltonians or Liouvillian as a Qobj
        self.ctrl_dyn_gen = ctrl    # Control operator [Qobj]
        self._num_ctrls = len(ctrl)

        self.initial = initial      # Initial state/rho/operator as Qobj
        if target is not None:
            self.target = target    # Target state/rho/operator as Qobj





    def __init__(self, initial, target, H, ctrl, phase=None,
                 times=None, tau=None, T=0, t_step=0, _num_x=0, _filter=None,
                 _tslotcomp=None,
                 ):


        if _filter in None:
            self.filter = filters.pass_througth()
        else:
            self.filter = _filter

        self._x_shape, self.time = self.filter.init_timeslots(times, tau, T,
                                                    t_step, _num_x, _num_ctrls)
        self._num_tslots = len(self.time)-1
        self._evo_time = self.time[-1]
        if np.allclose(np.diff(self.time), self.time[1]-self.time[0]):
            self._tau = falselist_cte(self.time[1]-self.time[0])
        else:
            self._tau = np.diff(self.time)
        # state and gradient before filter
        self._x = np.zeros(self._x_shape)
        self.gradient_x = np.zeros(self._x_shape)
        # state and gradient after filter
        self._ctrl_amps = np.zeros((self._num_tslots, self._num_ctrls))
        self._gradient_u = np.zeros((self._num_tslots, self._num_ctrls))

        #self._set_memory_optimizations(**kwarg)

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
                                   for t in self.time] for ctr in ctrl])

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
        pass
        """
        Set various memory optimisation attributes based on the
        memory_optimization attribute.
        """

        """if oper_dtype is None:
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
            self.cache_prop = cache_prop"""


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
            if not self.gradient_x:
                self._compute_grad()
        return self.gradient_x

    def _compute_state(self, x):
        """For a state x compute the cost"""
        self.x_ = x
        self._ctrl_amps = self.filter(x)
        self.tslotcomp.set(self._ctrl_amps)
        self.error = 0.
        for costs in self.costcomp:
            error = costs.costs()
            self.error += error

    def _compute_grad(self):
        """For a state x compute the grandient"""
        gradient_u = np.zeros((self._num_tslots, self._num_ctrls))
        for costs in self.costcomp:
            gradient_u_cost = costs.grad()
            gradient_u += gradient_u_cost
        self.gradient_x = self.filter.reverse(gradient_u)

    """def _apply_phase(self, dg):
        return self._prephase * dg * self._postphase"""
