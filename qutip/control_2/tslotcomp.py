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
import timeit
# QuTiP
from qutip import Qobj
# QuTiP control modules
import qutip.control.errors as errors
import qutip.control.dump as qtrldump
# QuTiP logging
import qutip.logging_utils as logging
logger = logging.get_logger()


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


class TimeslotComputer(object):
    """
    Base class for all Timeslot Computers
    """
    def __init__(self, H, ctrl, initial, target, tau, n_t):
        self.id_text = 'TS_COMP_BASE'
        self.cache_text = 'Save'
        self.exp_text = 'Power'
        self.drift = H
        self.ctrl = ctrl
        self.initial = initial
        self.target = target
        self.tau = tau
        self.n_t = n_t
        self.num_ctrl = len(ctrl)
        self._ctrl_amps = np.zeros((len(self.tau),self.num_ctrl))

    def set(self, u):
        self._ctrl_amps = u
        self.status = 0

    def _compute_gen(self):
        self._dyn_gen = [None] * self.n_t
        for t in range(self.n_t):
            self._dyn_gen[t] = self.drift[t].copy()
            for i in range(self.num_ctrl):
                self._dyn_gen[t] += self._ctrl_amps[t,i] * self.ctrl[t, i]
        #if not self.cache_phased_dyn_gen:
            #for t in range(self.num_tslots):
                #self._dyn_gen[t] = self._apply_phase(self._dyn_gen[t])


class TSComp_Save_Power_all(TimeslotComputer):
    """
    Timeslot Computer - Update All
    Updates and keep all dynamics generators, propagators and evolutions when
    ctrl amplitudes are updated
    """
    def __init__(self, H, ctrl, initial, target, tau, n_t, num_ctrl):
        self.id_text = 'ALL'
        self.cache_text = 'Save'
        self.exp_text = 'Power'
        self.drift = H
        self.ctrl = ctrl
        self.initial = initial
        self.target = target
        self.tau = tau
        self.n_t = n_t
        self.num_ctrl = num_ctrl

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
        self._compute_gen()
        self.T = T_target
        return self.fwd[T_target]

    def forward(self, T_targets):
        self._compute_gen()
        self.T = max(T_targets)
        for t in np.sort(T_targets):
            yield self.fwd[t]

    def reversed_onwd(self):
        back = 1
        for i in range(self.T-1,-1,-1):
            yield i, back, self._dU[i], self._prop[i], self.fwd[i]
            back = back*self._prop[i]

    def reversed_onto(self, target=False, targetd=False):
        if target:
            back = target.dag()
        elif targetd:
            back = targetd
        else:
            back = self.target.dag()
        for i in range(self.T-1,-1,-1):
            yield i, back, self._dU[i], self._prop[i], self.fwd[i]
            back = back*self._prop[i]

    def reversed_cumulative(self, target=False, targetd=False, times=None,
                            phase=None):
        if times is None:
            times = np.arange(self.T)+1
        else:
            times = np.sort(np.array(times, dtype=int))
        if phase is None:
            phase = np.ones(len(times))
        if target or targetd:
            if target:
                targetd = target.dag()
            back = targetd*phase[-1]
        else:
            targetd = self.target.dag()
            back = targetd*phase[-1]
        ind = times.shape[0]-2
        for i in range(times[-1]-1,-1,-1):
            yield i, back, self._dU[i], self._prop[i], self.fwd[i]
            back = back*self._prop[i]
            if ind != -1 and i == times[ind]:
                back += targetd*phase[ind]
                ind -= 1


class TSComp_Power(TimeslotComputer):
    """
    Timeslot Computer - Update All
    Updates and keep all dynamics generators, propagators and evolutions when
    ctrl amplitudes are updated
    """
    def __init__(self, H, ctrl, initial, target, tau, n_t, num_ctrl):
        self.id_text = 'ALL'
        self.cache_text = 'None'
        self.exp_text = 'Power'
        self.drift = H
        self.ctrl = ctrl
        self.initial = initial
        self.target = target
        self.tau = tau
        self.n_t = n_t
        self.num_ctrl = num_ctrl

        self.T = 0

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

    def reversed_onwd(self):
        back = 1
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
            back = back*_prop

    def reversed_onto(self, target=False, targetd=False):
        if target:
            back = target.dag()
        elif targetd:
            back = targetd
        else:
            back = self.target.dag()
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
            back = back*_prop

    def reversed_cumulative(self, target=False, targetd=False, times=None,
                            phase=None):
        if times is None:
            times = np.arange(self.T)+1
        else:
            times = np.sort(np.array(times, dtype=int))

        if phase is None:
            phase = np.ones(len(times))
        if target or targetd:
            if target:
                targetd = target.dag()
            back = targetd*phase[-1]
        else:
            targetd = self.target.dag()
            back = targetd*phase[-1]

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
            back = back*_prop
            if ind != -1 and t == times[ind]:
                back += targetd*phase[ind]
                ind -= 1
