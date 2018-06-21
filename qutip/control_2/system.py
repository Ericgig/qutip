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
from qutip import Qobj, td_Qobj
#from qutip.sparse import sp_eigs, _dense_eigs
import qutip.settings as settings
# QuTiP logging
import qutip.logging_utils as logging
logger = logging.get_logger()
# QuTiP control modules
#import qutip.control.errors as errors
#import qutip.control.tslotcomp as tslotcomp
#import qutip.control.fidcomp as fidcomp
#import qutip.control.propcomp as propcomp
#import qutip.control.symplectic as sympl
#import qutip.control.dump as qtrldump




from qutip.qobj import Qobj
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from qutip.control.optimresult import OptimResult
import qutip.control_2.tslotcomp as tslotcomp
import qutip.control_2.fidcomp as fidcomp
import qutip.control_2.filters as filters
import qutip.control_2.matrix as matrix
import qutip.control_2.optimize as optimize

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
        self.costcomp = None
        self.costs = []
        self.other_cost = []
        self.early = False

        self.filter = None
        self.psi0 = None
        self.mode = "speed"
        self.tslotcomp = None

    def set_initial_state(self, u):
        self.psi0 = u

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
            if initial.dims == self.dims:
                self.initial = initial
                self.state_evolution = False
            elif initial.dims[0] == self.dims[1] and initial.shape[1] == 1:
                self.initial = initial
                self.state_evolution = True
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

    def set_times(self, times=None, tau=None, T=0, t_step=0, num_x=0):
        self.t_data = (times, tau, T, t_step, num_x)

    def optimization(self, mode="speed", mem=0, _tslotcomp=None):
        if mode in ["mem","speed"]:
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

        if weight is None:
            self.weight = 1.
        elif isinstance(weight, (int,float)):
            self.weight = weight
        elif isinstance(weight, (list, np.ndarray)):
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
        if self.filter is None:
            self.filter = filters.pass_througth()

        self._x_shape, self.time = self.filter.init_timeslots(*self.t_data, num_ctrl=self._num_ctrls)

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
            if self.mode == "mem":
                matrix_type = matrix.control_sparse
            else:
                matrix_type = matrix.control_dense
            if isinstance(self.drift_dyn_gen, Qobj):
                drift = matrix.falselist_cte(matrix_type(self.drift_dyn_gen))
            else:
                drift = matrix.falselist_func(self.drift_dyn_gen, self._tau, matrix_type)

            ctrl_td = any([isinstance(ctrl, td_Qobj) for ctrl in self.ctrl_dyn_gen])
            if not ctrl_td:
                ctrls = matrix.falselist2d_cte(
                    [matrix_type(ctrl) for ctrl in self.ctrl_dyn_gen])
            else:
                drift = matrix.falselist2d_func(
                    [td_Qobj(ctrl) for ctrl in self.ctrl_dyn_gen], self._tau, matrix_type)

            initial = matrix_type(self.initial)   # As a vec?
            target = matrix_type(self.target)     # As a vec?
            self.tslotcomp = tslotcomp.TSComp_Save_Power_all(drift, ctrls, initial, target,
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

        if self.psi0 is None:
            self.x0 = np.random.rand(self._x_shape)
        else:
            if isinstance(self.psi0, list):
                self.psi0 = np.array(self.psi0)
            if self.psi0.shape == self._x_shape:
                # pre filter
                self.x0 = self.psi0
            elif self.psi0.shape == (self._num_tslots, self._num_ctrls):
                # post filter u0
                raise NotImplementedError()
            else:
                raise Exception("x0 bad shape")

        self.x_ = self.x0*0.5
        self.gradient_x = False

        self.solver = optimize.Optimizer(self._error, self._gradient, self.x0)

    def run(self):
        result = OptimResult()
        result.initial_amps = self.filter(self.x0)
        result.initial_x = self.x0*1.
        result.initial_fid_err = self._error(self.x0)
        result.evo_full_initial =  self.tslotcomp.state_T(self._num_tslots).data*1

        self.solver.run_optimization(result)

        result.evo_full_final =  self.tslotcomp.state_T(self._num_tslots).data*1
        result.final_amps = self.filter(result.final_x)

        return result



    ### -------------------------- Computation part ---------------------------
    def _error(self, x):
        if not np.allclose(self.x_, x):
            self.gradient_x = False
            self._compute_state(x)
        return self.error

    def _gradient(self, x):
        if self.gradient_x is False:
            if not np.allclose(self.x_, x):
                self._compute_state(x)
            self._compute_grad()
        return self.gradient_x

    def _compute_state(self, x):
        """For a state x compute the cost"""
        self.x_ = x*1.
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
