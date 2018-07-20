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
# @email1: agp1@aber.ac.uk
# @email2: alex.pitchford@gmail.com
# @organization: Aberystwyth University
# @supervisor: Daniel Burgarth

"""
Timeslot Computer
These classes determine which dynamics generators, propagators and evolutions
are recalculated when there is a control amplitude update.
The timeslot computer processes the lists held by the dynamics object

The default (UpdateAll) updates all of these each amp update, on the
assumption that all amplitudes are changed each iteration. This is typical
when using optimisation methods like BFGS in the GRAPE algorithm

The alternative (DynUpdate) assumes that only a subset of amplitudes
are updated each iteration and attempts to minimise the number of expensive
calculations accordingly. This would be the appropriate class for Krotov type
methods. Note that the Stats_DynTsUpdate class must be used for stats
in conjunction with this class.
NOTE: AJGP 2011-10-2014: This _DynUpdate class currently has some bug,
no pressing need to fix it presently

If all amplitudes change at each update, then the behavior of the classes is
equivalent. _UpdateAll is easier to understand and potentially slightly faster
in this situation.

Note the methods in the _DynUpdate class were inspired by:
DYNAMO - Dynamic Framework for Quantum Optimal Control
See Machnes et.al., arXiv.1011.4874
"""

"""
2018 - Eric Giguere

Timeslot Computer
Will be reconstructed to be generator-like:

1) set for the iteration
2) compute and return the evolution at T
3) return back(t),prop(t),fwd(t) from the end one-per-one

This way, the prop(t) do not *need* to be saved:
    In memory_optimization mode, no matrix/state need to be saved at each time.
    However, the prop must be computed one extra time for each tslot...

    For compute time, timeslot keeps the saved prop/state for each tslot.
"""

import os
import warnings
import numpy as np
import scipy as sc
import timeit
# QuTiP
from qutip import Qobj
# QuTiP control modules
import qutip.control.errors as errors
import qutip.control.dump as qtrldump
# QuTiP logging
import qutip.logging_utils as logging
logger = logging.get_logger()

import importlib
import importlib.util
moduleName = "/home/eric/algo/qutip/qutip/qutip/control_2/matrix.py"
spec = importlib.util.spec_from_file_location("mat", moduleName)
mat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mat)
control_dense = mat.control_dense
control_sparse = mat.control_sparse

def _is_unitary(prop):
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

def _calc_unitary_err(prop):
    if isinstance(A, Qobj):
        err = np.sum(abs(np.eye(A.shape[0]) - A*A.dag().full()))
    else:
        err = np.sum(abs(np.eye(len(A)) - A.dot(A.T.conj())))
    return err

def unitarity_check(props):
    """
    Checks whether all propagators are unitary
    """
    for k in range(self.num_tslots):
        if not self._is_unitary(self._prop[k]):
            pass

shape_c = [(0,0)]
def d2c(d_vec, N=0):
    if N==0:
        N=len(d_vec)//2
    c_vec = d_vec[:N]+1j*d_vec[N:]
    try:
        return c_vec.reshape(shape_c[0])
    except:
        print(N)
        print(d_vec)
        print(c_vec)
        print(shape_c[0])

def c2d(c_vec, N=0):
    shape_c[0] = c_vec.shape
    c_vec = c_vec.flatten()
    if N==0:
        N=len(c_vec)
    d_vec = np.zeros(N*2)
    d_vec[:N] = c_vec.real
    d_vec[N:] = c_vec.imag
    return d_vec


class TimeslotComputer(object):
    """
    Base class for all Timeslot Computers
    """
    def __init__(self, H, ctrl, initial, tau, n_t, num_ctrl):
        self.id_text = 'TS_COMP_BASE'
        self.cache_text = 'Save'
        self.exp_text = 'Power'
        self.drift = H
        self.ctrl = ctrl
        self.initial = initial
        self.tau = tau
        self.n_t = n_t
        self.num_ctrl = num_ctrl
        self.T = 0
        self.size = self.drift[0].data.shape[0]

    def set(self, u):
        self._ctrl_amps = u
        self.status = 0

    def _compute_gen(self):
        self._dyn_gen = [None] * self.n_t
        for t in range(self.n_t):
            self._dyn_gen[t] = self.drift[t].copy()
            for i in range(self.num_ctrl):
                self._dyn_gen[t] += self._ctrl_amps[t,i] * self.ctrl[t, i]

    def state_T(self, T_target):
        pass

    def forward(self, T_targets):
        pass

    def reversed(self, target=False):
        pass

    def reversed_cumulative(self, target=False, times=None, phase=None):
        pass


class TSComp_Save_Power_all(TimeslotComputer):
    """
    Timeslot Computer - Update All
    Updates and keep all dynamics generators, propagators and evolutions when
    ctrl amplitudes are updated
    """
    def __init__(self, H, ctrl, initial, tau, n_t, num_ctrl):
        super().__init__(H, ctrl, initial, tau, n_t, num_ctrl)
        self.id_text = 'ALL'
        self.cache_text = 'Save'
        self.exp_text = 'Power'

    def _compute_gen(self):
        """
        Recalculates the evolution operators.
        Dynamics generators (e.g. Hamiltonian) and
        prop (propagators) are calculated as necessary

        Changed to a function: take propagators and return evolution
        """
        # Compute and cache all dyn_gen
        self._dyn_gen = [None] * self.n_t
        for t in range(self.n_t):
            self._dyn_gen[t] = self.drift[t].copy()
            for i in range(self.num_ctrl):
                self._dyn_gen[t] += self._ctrl_amps[t,i] * self.ctrl[t,i]

        # Compute and cache all prop and dprop
        self._prop = [None] * self.n_t
        self._dU = [None] * self.n_t # np.empty((self.n_t, self.num_ctrl), dtype=object)
        self.fwd = [self.initial]
        for t in range(self.n_t):
            self._prop[t], self._dU[t] = self._dyn_gen[t].dexp(self.ctrl[t,0],
                                             self.tau[t], compute_expm=True)
            # self._dU[t,1:] = [self._dyn_gen[t].dexp(self.ctrl[t,i], self.tau[t])
            #              for i in range(1,self.num_ctrl)]
            self.fwd.append(self._prop[t] * self.fwd[t])

    def state_T(self, T_target):
        self._compute_gen()
        self.T = T_target
        return self.fwd[T_target]

    def forward(self, T_targets):
        self._compute_gen()
        self.T = max(T_targets)
        for t in np.sort(T_targets):
            yield self.fwd[t]

    def reversed(self, target=None):
        if target is None:
            back = np.eye(self.size)
        else:
            if isinstance(target, control_dense):
                back = target.data
            else:
                back = target
        dU = [None] * self.num_ctrl
        for i in range(self.T-1,-1,-1):
            dU[0] =  self._dU[i]
            dU[1:] = [self._dyn_gen[i].dexp(self.ctrl[i,j], self.tau[i])
                          for j in range(1,self.num_ctrl)]
            yield i, back, dU, self._prop[i], self.fwd[i]
            back = self._prop[i].__rmul__(back)

    def reversed_cumulative(self, target=None, times=None, phase=None):
        if times is None:
            times = np.arange(self.T)+1
        else:
            times = np.sort(np.array(times, dtype=int))
        if phase is None:
            phase = np.ones(len(times))
        if target is None:
            target = np.eye(self.drift[0].data.shape[0])
            back = target * phase[-1]
        else:
            if isinstance(target, control_dense):
                back = target.data * phase[-1]
                print("Nope")
            else:
                back = target * phase[-1]
        ind = times.shape[0]-2
        dU = [None] * self.num_ctrl
        for i in range(times[-1]-1,-1,-1):
            dU[0] =  self._dU[i]
            dU[1:] = [self._dyn_gen[i].dexp(self.ctrl[i,j], self.tau[i])
                          for j in range(1,self.num_ctrl)]
            yield i, back, dU, self._prop[i], self.fwd[i]
            back = self._prop[i].__rmul__(back)
            if ind != -1 and i == times[ind]:
                back += target*phase[ind]
                ind -= 1


class TSComp_Power(TimeslotComputer):
    """
    Timeslot Computer - Update All
    Updates and keep all dynamics generators, propagators and evolutions when
    ctrl amplitudes are updated
    """
    def __init__(self, H, ctrl, initial, tau, n_t, num_ctrl):
        super().__init__(H, ctrl, initial, tau, n_t, num_ctrl)
        self.id_text = 'ALL'
        self.cache_text = 'None'
        self.exp_text = 'Power'

    def _compute_gen(self):
        """
        Recalculates the evolution operators.
        Dynamics generators (e.g. Hamiltonian) and
        prop (propagators) are calculated as necessary

        Changed to a function: take propagators and return evolution
        """
        # Compute and cache all dyn_gen
        self._dyn_gen = [None] * self.n_t
        for t in range(self.n_t):
            self._dyn_gen[t] = self.drift[t].copy()
            for i in range(self.num_ctrl):
                self._dyn_gen[t] += self._ctrl_amps[t,i] * self.ctrl[t,i]

        # Compute and cache all prop and dprop
        self._prop = [None] * self.n_t
        self._dU = np.empty((self.n_t, self.num_ctrl), dtype=object)
        self.fwd = [self.initial]
        for t in range(self.n_t):
            self._prop[t], self._dU[t,0] = self._dyn_gen[t].dexp(self.ctrl[t,0],
                                             self.tau[t], compute_expm=True)
            self._dU[t,1:] = [self._dyn_gen[t].dexp(self.ctrl[t,i], self.tau[t])
                         for i in range(1,self.num_ctrl)]
            self.fwd.append(self._prop[t] * self.fwd[t])

    def state_T(self, T_target):
        self.T = T_target
        fwd = self.initial
        for t in range(T_target):
            _dyn_gen = self.drift[t].copy()
            for i in range(self.num_ctrl):
                _dyn_gen += self._ctrl_amps[t,i] * self.ctrl[t,i]
            _prop = _dyn_gen.prop(self.tau[t])
            fwd = (_prop * fwd)
        self.final = fwd
        return fwd

    def forward(self, T_targets):
        T_targets = np.sort(np.array(T_targets, dtype=int))
        self.T = T_targets[-1]
        fwd = self.initial
        self.fwd = []
        if 0 in T_targets:
            yield fwd

        for t in range(T_targets[-1]):
            _dyn_gen = self.drift[t].copy()
            for i in range(self.num_ctrl):
                _dyn_gen += self._ctrl_amps[t,i] * self.ctrl[t,i]
            _prop = _dyn_gen.prop(self.tau[t])
            fwd = (_prop * fwd)
            if t+1 in T_targets:
                yield fwd
        self.final = fwd

    def reversed(self, target=None):
        if target is None:
            back = np.eye(self.size)
        else:
            if isinstance(target, control_dense):
                back = target.data
            else:
                back = target
        _dU = [None] * self.num_ctrl
        fwd = self.final
        for t in range(self.T-1,-1,-1):
            _dyn_gen = self.drift[t].copy()
            for i in range(self.num_ctrl):
                _dyn_gen += self._ctrl_amps[t,i] * self.ctrl[t,i]
            _prop, _dU[0] = _dyn_gen.dexp(self.ctrl[t,0],
                            self.tau[t], compute_expm=True)
            _dU[1:] = [_dyn_gen.dexp(self.ctrl[t,i], self.tau[t])
                                for i in range(1,self.num_ctrl)]
            fwd = _prop.dag() * fwd
            yield t, back, _dU, _prop, fwd
            back = _prop.__rmul__(back)

    def reversed_cumulative(self, target=None, times=None, phase=None):
        if times is None:
            times = np.arange(self.T)+1
        else:
            times = np.sort(np.array(times, dtype=int))
        if phase is None:
            phase = np.ones(len(times))
        if target is None:
            back = np.eye(self.size) * phase[-1]
            target = np.eye(self.size)
        else:
            if isinstance(target, control_dense):
                back = target.data * phase[-1]
            else:
                back = target * phase[-1]

        _dU = [None] * self.num_ctrl
        fwd = self.final
        ind = times.shape[0]-2
        for t in range(times[-1]-1,-1,-1):
            _dyn_gen = self.drift[t].copy()
            for i in range(self.num_ctrl):
                _dyn_gen += self._ctrl_amps[t,i] * self.ctrl[t,i]
            _prop, _dU[0] = _dyn_gen.dexp(self.ctrl[t,0],
                             self.tau[t], compute_expm=True)
            _dU[1:] = [_dyn_gen.dexp(self.ctrl[t,i], self.tau[t])
                                for i in range(1,self.num_ctrl)]
            fwd = _prop.dag() * fwd
            yield t, back, _dU, _prop, fwd
            back = _prop.__rmul__(back)
            if ind != -1 and t == times[ind]:
                #print(target)
                back += target*phase[ind]
                ind -= 1


class TSComp_Int(TimeslotComputer):
    """
    Timeslot Computer - Update All
    Updates and keep all dynamics generators, propagators and evolutions when
    ctrl amplitudes are updated
    """
    def __init__(self, H, ctrl, initial, tau, n_t, num_ctrl):
        super().__init__(H, ctrl, initial, tau, n_t, num_ctrl)
        self.id_text = 'ALL'
        self.cache_text = 'None'
        self.exp_text = 'Integration'
        self.times = np.insert(np.cumsum(tau), 0, 0.)


        self.T = 0
        dt = min(self.tau)/10
        def f(x,t):
            x_complex = d2c(x)
            ti = np.searchsorted(self.times[1:-1],t+dt)
            try:
                return c2d((self._dyn_gen[ti]*x_complex))
            except:
                print(t,ti,self.times)
                print(self._dyn_gen[ti]*x_complex)
        self.int_func = f

    def _compute_gen(self):
        """
        Recalculates the evolution operators.
        Dynamics generators (e.g. Hamiltonian) and
        prop (propagators) are calculated as necessary

        Changed to a function: take propagators and return evolution
        """
        # Compute and cache all dyn_gen
        self._dyn_gen = [None] * self.n_t
        for t in range(self.n_t):
            self._dyn_gen[t] = self.drift[t].copy()
            for i in range(self.num_ctrl):
                self._dyn_gen[t] += self._ctrl_amps[t,i] * self.ctrl[t,i]

    def state_T(self, T_target):
        self._compute_gen()
        self.T = T_target
        res = sc.integrate.odeint(self.int_func,
                                  c2d(self.initial),
                                  self.times)
        self.final = d2c(res[-1])
        self.fwd = [d2c(state) for state in res]
        return self.final

    def forward(self, T_targets):
        self._compute_gen()
        T_targets = np.sort(np.array(T_targets, dtype=int))
        self.T = T_targets[-1]
        times = self.times[:self.T+1]
        res = sc.integrate.odeint(self.int_func,
                                  c2d(self.initial),
                                  times)
        self.final = d2c(res[-1])
        self.fwd = [d2c(state) for state in res]
        for i in T_targets:
            yield d2c(res[i])

    def reversed(self, target=None):
        if target is None:
            back = np.eye(self.drift[0].data.shape[0])
        else:
            if isinstance(target, (int, float, complex)):
                back = np.eye(self.drift[0].data.shape[0]) * target
            elif isinstance(target, (control_dense, control_sparse)):
                back = target.data
            else:
                back = target.conj()

        times = self.times[::-1]
        back = sc.integrate.odeint(self.int_func, c2d(back), times)
        for t in range(self.T-1,-1,-1):
            _dU = [self._dyn_gen[t].dexp(self.ctrl[t,i], self.tau[t])
                                for i in range(self.num_ctrl)]
            yield t, d2c(back[self.T-1-t]).conj(), _dU, None, self.fwd[t]

    def reversed_cumulative(self, target=None, times=None, phase=None):
        if times is None:
            times = np.arange(self.T)+1
        times = np.sort(np.array(times, dtype=int))[::-1]

        if phase is None:
            phase = np.ones(len(times))

        if target is None:
            back = np.eye(self.size) * phase[-1]
            target = np.eye(self.size)
        else:
            if isinstance(target, (int, float, complex)):
                back = np.eye(self.drift[0].data.shape[0]) * target * phase[-1]
                target = np.eye(self.drift[0].data.shape[0]) * target
            if isinstance(target, control_dense):
                back = target.data * phase[-1]
                target = target.data
            else:
                target = target.conj()
                back = target * phase[-1]

        ode_times = self.times[self.T-1::-1]
        i = 0
        ind = times.shape[0]-2
        #back = sc.integrate.odeint(self.int_func, c2d(back), ode_times)
        #back = [d2c(_back) for _back in back][::-1]

        for t in range(self.T-1,-1,-1):
            _dU = [self._dyn_gen[t].dexp(self.ctrl[t,i], self.tau[t])
                                for i in range(self.num_ctrl)]
            yield t, back.conj(), _dU, None, self.fwd[t]
            back = d2c(sc.integrate.odeint(self.int_func, c2d(back),
                                           ode_times[i:i+2])[-1])
            i += 1
            if t in times:
                back += target * phase[ind]
                ind -= 1


        """
        if self.T in times:
            #print("First")
            #T = self.T-1
            #_dU = [self._dyn_gen[T].dexp(self.ctrl[T,i], self.tau[T])
            #                    for i in range(self.num_ctrl)]
            #yield T, back, _dU, None, self.fwd[T]
            ode_times = range(times[0], times[1]-1, -1)
            ii = 1
        else:
            ode_times = range(self.T+1, times[0]-1, -1)
            ii = 0

        ind = 0
        for t in range(times[0]-1,-1,-1):
            _dU = [self._dyn_gen[t].dexp(self.ctrl[t,i], self.tau[t])
                                for i in range(self.num_ctrl)]
            yield t, back, _dU, None, self.fwd[t]
            ind += 1
            if t == times[ii]-1:
                ii += 1
                ind = 0
                back = back + target*phase[ind]
                ode_times = range(times[ii], times[ii]-1, -1)
                back = d2c(sc.integrate.odeint(self.int_func,
                                               c2d(back),
                                               ode_times)[-1])
        """
