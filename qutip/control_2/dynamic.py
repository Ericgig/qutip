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
from qutip import Qobj, td_Qobj, mat2vec
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

import importlib
import importlib.util

moduleName = "/home/eric/algo/qutip/qutip/qutip/control_2/matrix.py"
spec = importlib.util.spec_from_file_location("matrix", moduleName)
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)

moduleName = "/home/eric/algo/qutip/qutip/qutip/control_2/tslotcomp.py"
spec = importlib.util.spec_from_file_location("tslotcomp", moduleName)
tslotcomp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tslotcomp)

moduleName = "/home/eric/algo/qutip/qutip/qutip/control_2/fid_comp.py"
spec = importlib.util.spec_from_file_location("fidcomp", moduleName)
fidcomp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fidcomp)

moduleName = "/home/eric/algo/qutip/qutip/qutip/control_2/filters.py"
spec = importlib.util.spec_from_file_location("filters", moduleName)
filters = importlib.util.module_from_spec(spec)
spec.loader.exec_module(filters)

moduleName = "/home/eric/algo/qutip/qutip/qutip/control_2/stats.py"
spec = importlib.util.spec_from_file_location("stats", moduleName)
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)
Stats = stats.Stats

moduleName = "/home/eric/algo/qutip/qutip/qutip/control_2/optimize.py"
spec = importlib.util.spec_from_file_location("optimize", moduleName)
optimize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(optimize)

moduleName = "/home/eric/algo/qutip/qutip/qutip/control_2/pulsegen.py"
spec = importlib.util.spec_from_file_location("pulsegen", moduleName)
pulsegen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pulsegen)


from qutip.control.optimresult import OptimResult
"""
import qutip.control_2.tslotcomp as tslotcomp
import qutip.control_2.fidcomp as fidcomp
import qutip.control_2.filters as filters
import qutip.control_2.matrix as matrix
import qutip.control_2.optimize as optimize
from qutip.control_2.stats import Stats
"""

class dynamics:
    """
    This class compute the error and gradient for the GRAPE systems.*

    * Other object do the actual computation to cover the multiple situations.

    methods:
        _error(x):
            x: np.ndarray, state of the pulse
            return 1-fidelity

        _gradient(x):
            x: np.ndarray, state of the pulse
            return the error gradient
    """

    def __init__(self):
        # Main object
        self.stats = None
        self.filter = None
        self.tslotcomp = None
        self.costcomp = []
        self.solver = None
        # Options
        self.opt_mode = ("", "")
        #self.options = {}
        self.termination_conditions = optimize.termination_conditions
        self.solver_method_options = optimize.method_options
        self.matrix_options = matrix.matrix_opt.copy()
        self.mem = 0
        # Costcomp option
        self.mode = None
        self.early = False
        self.early_times = None
        self.weight = None
        self.state_evolution = None
        self.other_cost = []
        self.options_list = []
        # Physical system
        self.drift_dyn_gen = None
        self.issuper = None
        self.dims = (0,)
        self.shape = (0,0)
        self.ctrl_dyn_gen = None
        self.initial = None
        self.target = None
        # Shape of the controls array
        self.t_data = (None, None, None, None, None)
        self.time = None
        self._tau = None
        self._num_tslots = 0
        self._num_ctrls = 0
        self._x_shape = (0,0)
        # Initial controls array
        self.psi0 = None
        self.x0 = None
        # Running controls array
        self.x_ = None
        self._ctrl_amps = None
        self.gradient_x = None
        self._gradient_u = None
        self.error = np.inf
        self.fidelity_stats = []
        self.u_limits = None
        # Result object
        self.result_phase = 1.

    def set_initial_state(self, u):
        self.psi0 = u

    def set_physic(self, H, ctrls, initial=None, target=None):
        self.drift_dyn_gen = H # Hamiltonians or Liouvillian as a Qobj or td_Qobj
        if isinstance(H, Qobj):
            self.issuper = H.issuper
            self.dims = H.dims
            self.shape = H.shape
        elif isinstance(H, td_Qobj):
            self.issuper = H.cte.issuper
            self.dims = H.cte.dims
            self.shape = H.cte.shape
        else:
            raise Exception("The drift operator is expected"
                            " to be a Qobj or td_Qobj.")
        if len(self.shape) != 2 or self.shape[0] != self.shape[1]:
            raise Exception("Invalid drift operator.")

        if isinstance(ctrls, (Qobj, td_Qobj)):
            ctrls = [ctrls]
        self.ctrl_dyn_gen = ctrls # Control operator [Qobj] or [td_Qobj]
        self._num_ctrls = len(ctrls)
        for ctrl in ctrls:
            if not isinstance(ctrl, (Qobj, td_Qobj)):
                raise Exception("The ctrls operators are expected"
                                " to be a list of Qobj or td_Qobj.")
            if isinstance(ctrl, td_Qobj):
                c = c.cte
            else:
                c = ctrl
            if not self.dims == c.dims:
                raise Exception("Drift and ctrls operators"
                                " dimensions do not match.")

        if isinstance(target, Qobj):
            target = target.full()
        if isinstance(target, np.ndarray):
            if target.shape == self.shape:
                self.target = target
                self.state_evolution = False
            else:
                if len(target.shape) == 2 and \
                    target.shape[0] == target.shape[1]:
                        target = mat2vec(target)
                elif len(target.shape) == 2:
                    target = target.flatten()
                if len(target) == self.shape[1]:
                    self.state_evolution = True
                    self.target = target
                else:
                    raise Exception("Dims of the target state and"
                                    " Drift operator do not match.")
        else:
            raise Exception("The target state is expected to be "
                            "a Qobj or an array.")

        if initial is None:
            pass
        else:
            if isinstance(initial, Qobj):
                initial = initial.full()
            if isinstance(initial, np.ndarray):
                if target.shape == initial.shape:
                    self.initial = initial
                else:
                    if len(initial.shape) == 2 and \
                        initial.shape[0] == initial.shape[1]:
                            initial = mat2vec(initial)
                    elif len(initial.shape) == 2:
                        initial = initial.flatten()
                    if len(initial) == len(target):
                        self.initial = initial
                    else:
                        raise Exception("Dims of the target state and"
                                        " initial state do not match.")
            else:
                raise Exception("The initial state is expected to be a Qobj or "
                                "np.array.")

        """if isinstance(target, Qobj):
            if target.dims == self.dims:
                self.target = target.full()
                self.state_evolution = False
            elif target.dims[0] == self.dims[1] and target.shape[1] == 1:
                self.target = target.full().flatten()
                self.state_evolution = True
            else:
                print(self.issuper)
                print(self.dims)
                print(target.shape)
                print(target.dims)
                raise Exception("Dims of the target state and"
                                " Drift operator do not match.")

        elif isinstance(target, np.ndarray):
            if len(target) == np.prod(target.shape):
                self.state_evolution = True
                self.target = target.flatten()
            if self.issuper and len(target) == np.prod(self.dims[1]):
                self.state_evolution = True
                self.target = target.flatten()
            elif len(target.shape) == 2 and target.shape == self.dims:
                self.target = target
                self.state_evolution = False
            else:
                print(self.issuper)
                print(target.shape)
                print(self.dims)

        f initial is None:
            pass
        elif isinstance(initial, Qobj):
            if not initial.dims == target.dims:
                raise Exception("Dims of the target state and"
                                " initial state do not match.")
            self.initial = initial.full()
            if self.state_evolution:
                self.initial = self.initial.flatten()
        elif isinstance(initial, np.ndarray):
            if not initial.shape == target.shape:
                raise Exception("Dims of the target state and"
                                " initial state do not match.")
            self.initial = initial
        else:
            raise Exception("The initial state is expected to be a Qobj.")"""

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

    def set_ulimit(self, times=None, tau=None, T=0, t_step=0, num_x=0):
        self.t_data = (times, tau, T, t_step, num_x)

    def set_stats(self, timings=1, states=1):
        self.stats = Stats(timings, states)

    def set_cost(self, mode=None, early=False, weight=None):
        """
        From the dimension of the state and drift operator, the
        family of fidelity computer is determined.

        mode :: str
            Tag of the fidelity computation
            State evolution:
                "SU":   real(<target,final>)
                "PSU":  abs(<target,final>)
                "PSU2": abs(<target,final>)**2
                "Diff": abs(target-final)**2

            Operator evolution:
                "TrDiff":   real(<target,final>)
                "TrSq":     abs(tr(target*final))**2
                "TrAbs":    abs(tr(target*final))

        early :: bool, [int]
            Solve for solution that converge early (slower)

            False: Only compute the fidelity of the resulting state.
            True: Sum the fidelity at every timestep. -- set by the filter
            list: Sum the fidelity at desired timestep.

        weight :: number, [number]
            Weight of the fidelity or list of weights for each timestep if early
        """

        if self.state_evolution:
            # State evolution
            if mode is None:
                mode = "PSU"
            if not mode in ["SU","PSU","PSU2","Diff"]:
                raise Exception("mode must be one of 'SU', 'PSU', 'PSU2', "
                                "'Diff' for unitary state evolution")
            self.mode = mode
        else:
            # Operator evolution
            if mode is None:
                mode = "TrDiff"
            if not mode in ["TrDiff","TrSq","TrAbs"]:
                raise Exception("mode must be one of 'TrDiff', 'TrSq', 'TrAbs'"
                                " for unitary state evolution")
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
                raise Exception("The number of weight do not match"
                                " the numbers of times")
        else:
            raise Exception("weight should be a real or list of real.")

    def _add_cost(self, cost_comp=None):
        self.other_cost.append(cost_comp)

    def optimization(self, opt_mode="yet", mat_mode="", tslot_mode="",
                     mem=0, _tslotcomp=None):
        if opt_mode == "mem":
            self.opt_mode[0] = "sparse"
            self.opt_mode[1] = "power"
        elif opt_mode == "speed":
            self.opt_mode[0] = "dense"
            self.opt_mode[1] = "int"

        if mat_mode in ["sparse", "dense", "mixed"]:
            self.opt_mode[0] = mat_mode
        if tslot_mode in ["power", "int", "full"]:
            self.opt_mode[1] = tslot_mode

        self.mem = mem

        if _tslotcomp is not None:
            self.tslotcomp = _tslotcomp

        """
        if "" in self.opt_mode:
            if mem and isinstance(mem, (int, float)):
                self.mem = mem
            else:
                #Not sure to want to get here yet
                import psutil
                self.mem = psutil.virtual_memory().free/1024/1024
        """

    def report(self):
        for line in self.options_list:
            print(line)

    def prepare(self):
        if self.stats is None:
            self.set_stats()

        if self.filter is None:
            self.filter = filters.pass_througth()
        self.set_amp_bound(set_ulimit[0], set_ulimit[1])

        self._x_shape, self.time = \
                self.filter.init_timeslots(*self.t_data,
                                           num_ctrl=self._num_ctrls)
        self._num_tslots = len(self.time)-1
        """if np.allclose(np.diff(self.time), self.time[1]-self.time[0]):
            self._tau = matrix.falselist_cte(self.time[1]-self.time[0],
                                             self._num_tslots)
        else:"""
        self._tau = np.diff(self.time)

        # state and gradient before filter
        self.x_ = np.zeros(self._x_shape)
        self.gradient_x = np.zeros(self._x_shape)
        # state and gradient after filter
        self._ctrl_amps = np.zeros((self._num_tslots, self._num_ctrls))
        self._gradient_u = np.zeros((self._num_tslots, self._num_ctrls))

        matrix.matrix_opt.update(self.matrix_options)

        if self.tslotcomp is None:
            if "" == self.opt_mode:
                pass

            if self.opt_mode[0] == "sparse":
                matrix_type = matrix.control_sparse
                self.options_list += \
                    ["Using sparse matrix"]
            elif self.opt_mode[0] == "mixed":
                matrix_type = matrix.control_sparse
                self.matrix.matrix_opt["sparse2dense"] = True
                self.options_list += \
                    ["Using sparse and dense matrix"]
            else:
                # fall back to dense
                matrix_type = matrix.control_dense
                self.options_list += \
                    ["Using dense matrix"]

            if isinstance(self.drift_dyn_gen, Qobj):
                drift = matrix.falselist_cte(matrix_type(self.drift_dyn_gen),
                                             self._num_tslots)
            else:
                drift = matrix.falselist_func(self.drift_dyn_gen,
                                              self._tau, matrix_type,
                                              self._num_tslots)

            ctrl_td = any([isinstance(ctrl, td_Qobj)
                                for ctrl in self.ctrl_dyn_gen])
            if not ctrl_td:
                ctrls = matrix.falselist2d_cte(
                    [matrix_type(ctrl) for ctrl in self.ctrl_dyn_gen],
                                                   self._num_tslots)
            else:
                ctrls = matrix.falselist2d_func(
                    [td_Qobj(ctrl) for ctrl in self.ctrl_dyn_gen],
                    self._tau, matrix_type, self._num_tslots)

            if self.opt_mode[1] == "power":
                self.tslotcomp = tslotcomp.TSComp_Power(drift, ctrls,
                    self.initial, self._tau, self._num_tslots, self._num_ctrls)
                self.options_list += ["Using power evolution without "
                                     "saving operators for memory"]
            elif self.opt_mode[1] == "int":
                self.tslotcomp = tslotcomp.TSComp_Int(drift, ctrls,
                    self.initial, self._tau, self._num_tslots, self._num_ctrls)
                self.options_list += ["Using differential equations evolution "
                                     "saving operators for memory"]
            else:
                # fall back to full save
                self.tslotcomp = tslotcomp.TSComp_Save_Power_all(drift, ctrls,
                    self.initial, self._tau, self._num_tslots, self._num_ctrls)
                self.options_list += ["Using power evolution "
                                     "saving operators for speed"]

        if not self.costcomp:
            if self.state_evolution:
                if self.early:
                    self.costcomp = [
                        fidcomp.FidCompStateEarly(self.tslotcomp,
                                                  self.target,
                                                  self.mode,
                                                  times=self.early_times,
                                                  weight=self.weight)]
                    self.options_list += ["Cost computer: FidCompStateEarly "
                                          + self.mode]
                else:
                    self.costcomp = [fidcomp.FidCompState(self.tslotcomp,
                                                          self.target,
                                                          self.mode)]
                    self.options_list += ["Cost computer: FidCompState "
                                          + self.mode]
            else:
                if self.early:
                    self.costcomp = [
                        fidcomp.FidCompOperatorEarly(self.tslotcomp,
                                                     self.target,
                                                     self.mode,
                                                     times=self.early_times,
                                                     weight=self.weight)]
                    self.options_list += ["Cost computer: FidCompOperatorEarly "
                                          + self.mode]
                else:
                    self.costcomp = [fidcomp.FidCompOperator(self.tslotcomp,
                                                             self.target,
                                                             self.mode)]
                    self.options_list += ["Cost computer: FidCompOperator "
                                          + self.mode]
        if self.other_cost:
            self.costcomp += self.other_cost
        if self.psi0 is None:
            self.x0 = np.random.rand(self._x_shape)
            self.options_list = ["Setting random starting amplitude"]
        else:
            if isinstance(self.psi0, pulsegen.PulseGen):
                self.psi0 = self.psi0(self._tau, self._num_ctrls)
            if isinstance(self.psi0, list):
                self.psi0 = np.array(self.psi0)
            if self.psi0.shape == self._x_shape:
                # pre filter
                self.x0 = self.psi0
                self.psi0 = self.filter(self.psi0)
            elif self.psi0.shape == (self._num_tslots, self._num_ctrls):
                # post filter
                self.x0 = self.filter.reverse_state(self.psi0)
            else:
                raise Exception("x0 bad shape")
        self.x_ = self.x0 * np.inf
        self.gradient_x = False
        self.solver = optimize.Optimizer(self._error, self._gradient, self.x0,
                                         self.stats, self._compute_stats)
        self.solver.add_bounds(self.filter.get_xlimit())

    def run(self):
        self.prepare()
        self.report()
        self.stats.options_list = self.options_list
        result = OptimResult()
        result.initial_amps = self.filter(self.x0)
        result.initial_x = self.x0*1.
        result.initial_fid_err = self._error(self.x0)
        result.evo_full_initial = self.tslotcomp.state_T(self._num_tslots) *\
                    self.result_phase

        self.solver.run_optimization(result)

        result.evo_full_final = self.tslotcomp.state_T(self._num_tslots) *\
                    self.result_phase
        result.final_amps = self.filter(result.final_x)

        return result


    ### -------------------------- Computation part ---------------------------
    def _compute_stats(self, x):
        if not np.allclose(self.x_, x):
            self.gradient_x = False
            self._compute_state(x)
            self._compute_grad()

        if self.stats.states == 1:
            self.stats.fidelity += [sum(self.fidelity_stats)]
        elif self.stats.states >= 2:
            self.stats.x += [x*1.]
            self.stats.u += [self._ctrl_amps*1.]
            self.stats.fidelity += [self.fidelity_stats]

        if self.stats.states >= 2:
            self.stats.grad_norm += [self.grad_norm]
        if self.stats.states >= 3:
            self.stats.grad_x += [self.gradient_x*1.]
            self.stats.grad_u += [self.gradient_u_cost*1.]

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
        self.stats.num_fidelity_computes += 1

        self.x_ = x*1.
        self._ctrl_amps = self.filter(x)
        self.tslotcomp.set(self._ctrl_amps)
        self.error = 0.

        error = []
        for costs in self.costcomp:
            error += [costs.costs()]
        self.error = sum(error)
        self.fidelity_stats += error

    def _compute_grad(self):
        """For a state x compute the grandient"""
        self.stats.num_grad_computes += 1

        gradient_u_cost = np.zeros((self._num_tslots, self._num_ctrls, \
                               len(self.costcomp)))
        gradient_u = np.zeros((self._num_tslots, self._num_ctrls))
        for i, costs in enumerate(self.costcomp):
            gradient_u_cost[:,:,i] = costs.grad()
            gradient_u += gradient_u_cost[:,:,i]
        self.gradient_x = self.filter.reverse(gradient_u)

        if self.stats.states >= 2:
            self.grad_norm += np.sum(self.gradient_x*\
                                                   self.gradient_x.conj())
        if self.stats.states >= 3:
            self.gradient_u_cost = gradient_u_cost
