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
Fidelity Computer

These classes calculate the fidelity error - function to be minimised
and fidelity error gradient, which is used to direct the optimisation

They may calculate the fidelity as an intermediary step, as in some case
e.g. unitary dynamics, this is more efficient

The idea is that different methods for computing the fidelity can be tried
and compared using simple configuration switches.

Note the methods in these classes were inspired by:
DYNAMO - Dynamic Framework for Quantum Optimal Control
See Machnes et.al., arXiv.1011.4874
The unitary dynamics fidelity is taken directly frm DYNAMO
The other fidelity measures are extensions, and the sources are given
in the class descriptions.
"""

import os
import warnings
import numpy as np
import scipy.sparse as sp
import itertools
# import scipy.linalg as la
import timeit
# QuTiP
from qutip import Qobj
# QuTiP logging
import qutip.logging_utils as logging
logger = logging.get_logger()
# QuTiP control modules
import qutip.control.errors as errors

def rhoProdTrace(rho0, rho1, N=None):
    if N is None:
        N = int(np.sqrt(rho0.shape[0]))
    trace = 0.
    for i,j in itertools.product(range(N),range(N)):
        trace += rho0[i*N+j]*rho1[j*N+i]
    return trace

class FidCompState():
    def __init__(self, tslotcomp, target, phase_option):
        """
        Computes fidelity error and gradient assuming unitary dynamics, e.g.
        closed qubit systems
        Note fidelity and gradient calculations were taken from DYNAMO
        (see file header)

        Attributes
        ----------
        phase_option : string
            determines how global phase is treated in fidelity calculations:
                PSU - global phase ignored
                PSU2 - global phase ignored
                SU - global phase included
        """
        self.tslotcomp = tslotcomp
        self.num_ctrls = self.tslotcomp.num_ctrl
        self.num_tslots = self.tslotcomp.n_t
        self.target = target

        self.SU = phase_option
        self.target_d = target.conj()
        if self.SU == "SU":
            self.dimensional_norm = np.real(np.dot(self.target_d,target))
        elif self.SU in ["PSU","PSU2"]:
            self.dimensional_norm = np.abs(np.dot(self.target_d,target))
        elif self.SU in ["Diff"]:
            self.dimensional_norm = target.data.shape[0]
            self.target_d = 1.
        elif self.SU == "SuTr":
            self.dimensional_norm = int(np.sqrt(target.data.shape[0]))
        else:
            raise Exception("Invalid phase_option for FidCompState.")

    def costs(self):
        self.final = self.tslotcomp.state_T(self.num_tslots)
        if self.SU == "SU":
            fidelity_prenorm = np.dot(self.target_d,self.final)
            cost = 1 - np.real(fidelity_prenorm) / self.dimensional_norm
        elif self.SU == "PSU":
            fidelity_prenorm = np.dot(self.target_d,self.final)
            cost = 1 - np.abs(fidelity_prenorm) / self.dimensional_norm
        elif self.SU == "PSU2":
            fidelity_prenorm = np.dot(self.target_d,self.final)
            cost = 1 - np.real(fidelity_prenorm*fidelity_prenorm.conj())
        elif self.SU == "Diff":
            dvec = (self.target - self.final)
            cost = np.real( np.dot(dvec.conj(),dvec)) / self.dimensional_norm
        elif self.SU == "SuTr":
            fidelity_prenorm = 0
            N = self.dimensional_norm
            fidelity_prenorm = rhoProdTrace(self.target, self.final, N)
            cost = 1 - np.real(fidelity_prenorm)
        return cost

    def grad(self):
        n_ctrls = self.num_ctrls
        n_ts = self.num_tslots

        #final = self.tslotcomp.state_T(n_ts)
        fidelity_prenorm = np.dot(self.target_d,self.final)
        if self.SU == "Diff":
            self.target_d = (self.target - self.final).conj()

        # create n_ts x n_ctrls zero array for grad start point
        grad = np.zeros([n_ts, n_ctrls], dtype=complex)

        if self.SU != "SuTr":
            # loop through all ctrl timeslots calculating gradients
            for k, onto_evo, dU, U, fwd_evo in \
                            self.tslotcomp.reversed(target=self.target_d):
                for j in range(n_ctrls):
                    grad[k, j] = -np.dot(onto_evo,dU[j]*fwd_evo)

        if self.SU == "SU":
            grad_normalized = np.real(grad) / self.dimensional_norm
        elif self.SU == "PSU":
            grad_normalized = np.real(grad / self.dimensional_norm *\
                                      np.exp(-1j * np.angle(fidelity_prenorm)))
        elif self.SU == "PSU2":
            grad_normalized = np.real(2 * fidelity_prenorm.conj() * grad)
        elif self.SU == "Diff":
            grad_normalized = np.real(2 * grad / self.dimensional_norm)

        elif self.SU == "SuTr":
            for k, onto_evo, dU, U, fwd_evo in \
                        self.tslotcomp.reversed():
                for j in range(n_ctrls):
                    dfinal = onto_evo @ (dU[j] * fwd_evo)
                    grad[k, j] = -rhoProdTrace(self.target,
                                               dfinal,
                                               self.dimensional_norm)
                # only work on dense state, sparse state ok?
            grad_normalized = np.real(grad)

        return grad_normalized

class FidCompStateEarly():
    def __init__(self, tslotcomp, target, phase_option, times=None, weight=None):
        """
        Computes fidelity error and gradient assuming unitary dynamics, e.g.
        closed qubit systems
        Note fidelity and gradient calculations were taken from DYNAMO
        (see file header)

        Attributes
        ----------
        phase_option : string
            determines how global phase is treated in fidelity calculations:
                PSU - global phase ignored
                PSU2 - global phase ignored
                SU - global phase included
        """
        self.tslotcomp = tslotcomp
        self.num_ctrls = self.tslotcomp.num_ctrl
        self.num_tslots = self.tslotcomp.n_t

        self.SU = phase_option
        self.target = target
        self.target_d =target.conj()

        if times is None:
            times = np.arange(self.num_tslots, dtype=int)[::-1]+1
        self.times = times
        if weight is None:
            weight = np.ones(len(times))/len(times)
        if len(weight) != len(times):
            raise Exception("The number of weight is not the same as times")
        self.weight = np.array(weight)

        if self.SU == "SU":
            self.dimensional_norm = np.real(np.dot(self.target_d,target))
        elif self.SU in ["PSU","PSU2"]:
            self.dimensional_norm = np.abs(np.dot(self.target_d,target))
        elif self.SU in ["Diff"]:
            self.dimensional_norm = target.data.shape[0]
            self.target_d = 1.
        else:
            raise Exception("Invalid phase_option for FidCompStateEarly.")

    def costs(self):
        fidelity = self.costs_t()
        return np.sum(fidelity)

    def costs_t(self):
        self.fidelity_prenorm = np.zeros(len(self.times),dtype=complex)
        self.diff = np.zeros((len(self.times),len(self.target)), dtype=complex)
        fidelity = np.zeros(len(self.times))
        for i, f_state in enumerate(self.tslotcomp.forward(self.times)):
            if self.SU == "SU":
                self.fidelity_prenorm[i] = np.dot(self.target_d,f_state)
                fidelity[i] = 1 - np.real(self.fidelity_prenorm[i]) / self.dimensional_norm
            elif self.SU == "PSU":
                self.fidelity_prenorm[i] = np.dot(self.target_d,f_state)
                fidelity[i] = 1 - np.abs(self.fidelity_prenorm[i]) / self.dimensional_norm
            elif self.SU == "PSU2":
                self.fidelity_prenorm[i] = np.dot(self.target_d,f_state)
                fidelity[i] = 1 - (self.fidelity_prenorm[i]*self.fidelity_prenorm[i].conj()).real
            elif self.SU == "Diff":
                dvec = (self.target - f_state)
                self.diff[i] = dvec.conj()
                fidelity[i] = np.real(np.dot(self.diff[i],dvec)) / self.dimensional_norm
            #elif self.SU == "DMTr":
        return fidelity*self.weight

    def grad(self):
        n_ctrls = self.num_ctrls
        n_ts = self.num_tslots

        # create n_ts x n_ctrls zero array for grad start point
        grad = np.zeros([n_ts, n_ctrls], dtype=complex)

        if self.SU == "SU":
            phase = self.weight / self.dimensional_norm
        elif self.SU == "PSU":
            phase = np.exp(-1j * np.angle(self.fidelity_prenorm) )*self.weight / self.dimensional_norm
        elif self.SU == "PSU2":
            phase = 2 * self.fidelity_prenorm.conj()*self.weight
        elif self.SU == "Diff":
            self.target_d = 1
            phase = []
            for i in range(len(self.times)):
                phase += [2/self.dimensional_norm * self.weight[i] * self.diff[i,:]]

        # loop through all ctrl timeslots calculating gradients
        for k, rev_evo, dU, U, fwd_evo in \
                    self.tslotcomp.reversed_cumulative( \
                        target=self.target_d, times=self.times,
                        phase=phase):
            for j in range(n_ctrls):
                grad[k, j] = -np.dot(rev_evo,dU[j]*fwd_evo)

        return np.real(grad)

class FidCompStateForbidden():
    def __init__(self, tslotcomp, forbidden, phase_option, times=None, weight=None):
        """
        Computes fidelity error and gradient assuming unitary dynamics, e.g.
        closed qubit systems
        Note fidelity and gradient calculations were taken from DYNAMO
        (see file header)

        Attributes
        ----------
        phase_option : string
            determines how global phase is treated in fidelity calculations:
                PSU - global phase ignored
                PSU2 - global phase ignored
                SU - global phase included
        """
        self.tslotcomp = tslotcomp
        self.num_ctrls = self.tslotcomp.num_ctrl
        self.num_tslots = self.tslotcomp.n_t

        self.SU = phase_option

        self.forbidden = forbidden
        self.forbidden_d = forbidden.conj()
        if times is None:
            times = np.arange(self.num_tslots, dtype=int)[::-1]+1
        self.times = times
        if weight is None:
            weight = np.ones(len(times))/len(times)
        if len(weight) != len(times):
            raise Exception("The number of weight is not the same as times")
        self.weight = np.array(weight)

        if self.SU == "SU":
            self.dimensional_norm = np.real(np.dot(self.forbidden_d,forbidden))
        elif self.SU in ["PSU","PSU2"]:
            self.dimensional_norm = np.abs(np.dot(self.forbidden_d,forbidden))
        elif self.SU in ["Diff"]:
            self.dimensional_norm = forbidden.data.shape[0]
            self.forbidden_d = 1.
        else:
            raise Exception("Invalid phase_option for FidCompStateEarly.")

    def costs(self):
        fidelity = self.costs_t()
        return np.sum(fidelity)

    def costs_t(self):
        self.fidelity_prenorm = np.zeros(len(self.times),dtype=complex)
        self.diff = np.zeros((len(self.times),len(self.forbidden)), dtype=complex)
        fidelity = np.zeros(len(self.times))
        for i, f_state in enumerate(self.tslotcomp.forward(self.times)):
            if self.SU == "SU":
                self.fidelity_prenorm[i] = np.dot(self.forbidden_d, f_state)
                fidelity[i] = np.real(self.fidelity_prenorm[i]) / self.dimensional_norm
            elif self.SU == "PSU":
                self.fidelity_prenorm[i] = np.dot(self.forbidden_d, f_state)
                fidelity[i] = np.abs(self.fidelity_prenorm[i]) / self.dimensional_norm
            elif self.SU == "PSU2":
                self.fidelity_prenorm[i] = np.dot(self.forbidden_d, f_state)
                fidelity[i] = (self.fidelity_prenorm[i]*self.fidelity_prenorm[i].conj()).real
            elif self.SU == "Diff":
                dvec = (self.forbidden - f_state)
                self.diff[i] = dvec.conj()
                fidelity[i] = -np.real(np.dot(self.diff[i], dvec)) / self.dimensional_norm
        return fidelity*self.weight

    def grad(self):
        n_ctrls = self.num_ctrls
        n_ts = self.num_tslots

        # create n_ts x n_ctrls zero array for grad start point
        grad = np.zeros([n_ts, n_ctrls], dtype=complex)

        if self.SU == "SU":
            phase = self.weight / self.dimensional_norm
        elif self.SU == "PSU":
            phase = np.exp(-1j * np.angle(self.fidelity_prenorm) ) \
                    * self.weight / self.dimensional_norm
        elif self.SU == "PSU2":
            phase = 2 * self.fidelity_prenorm.conj() * self.weight
        elif self.SU == "Diff":
            self.forbidden_d = 1
            phase = []
            for i in range(len(self.times)):
                phase += [2 / self.dimensional_norm *
                          self.weight[i] * self.diff[i,:]]

        # loop through all ctrl timeslots calculating gradients
        for k, rev_evo, dU, U, fwd_evo in \
                    self.tslotcomp.reversed_cumulative( \
                        target=self.forbidden_d, times=self.times,
                        phase=phase):
            for j in range(n_ctrls):
                grad[k, j] = np.dot(rev_evo,dU[j]*fwd_evo)

        return np.real(grad)


class FidCompOperator():
    def __init__(self, tslotcomp, target, mode="TrDiff", scale_factor=0):
        """
        Computes fidelity error and gradient for general system dynamics
        by calculating the the fidelity error as the trace of the overlap
        of the difference between the target and evolution resulting from
        the pulses with the transpose of the same.
        This should provide a distance measure for dynamics described by matrices
        Note the gradient calculation is taken from:
        'Robust quantum gates for open systems via optimal control:
        Markovian versus non-Markovian dynamics'
        Frederik F Floether, Pierre de Fouquieres, and Sophie G Schirmer

        Attributes
        ----------
        scale_factor : float
            The fidelity error calculated is of some arbitary scale. This
            factor can be used to scale the fidelity error such that it may
            represent some physical measure
            If None is given then it is caculated as 1/2N, where N
            is the dimension of the drift, when the Dynamics are initialised.
        """
        self.tslotcomp = tslotcomp
        self.num_ctrls = self.tslotcomp.num_ctrl
        self.num_tslots = self.tslotcomp.n_t
        self.target = target
        self.mode = mode

        if mode=="TrDiff":
            if not scale_factor:
                self.scale_factor = 1.0 / (2.0*self.target.data.shape[0])
            else:
                self.scale_factor = scale_factor
        elif mode=="TrSq":
            self.target_d = target.T.conj()
            if not scale_factor:
                self.scale_factor = 1.0 / (self.target.data.shape[0])**2
            else:
                self.scale_factor = scale_factor
        elif mode=="TrAbs":
            self.target_d = target.T.conj()
            if not scale_factor:
                self.scale_factor = 1.0 / (self.target.data.shape[0])
            else:
                self.scale_factor = scale_factor
        else:
            raise Exception("mode: 'TrDiff', 'TrSq', 'TrAbs'.")

    def costs(self):
        n_ts = self.num_tslots
        final = self.tslotcomp.state_T(n_ts)
        if self.mode=="TrDiff":
            evo_f_diff = self.target - final
            #fid_err = self.scale_factor*np.real((evo_f_diff.dag()*evo_f_diff).tr())
            fid_err = self.scale_factor*np.real(np.sum(evo_f_diff.conj()*evo_f_diff))
        elif self.mode=="TrSq":
            fid = (self.target_d@final).trace()
            fid_err = 1 - self.scale_factor * np.real(fid * np.conj(fid))
        elif self.mode=="TrAbs":
            fid = (self.target_d@final).trace()
            fid_err = 1-self.scale_factor*np.abs(fid)
        if np.isnan(fid_err):
            # Shouldn't this raise an error?
            fid_err = np.Inf
        return fid_err

    def grad(self):
        n_ctrls = self.num_ctrls
        n_ts = self.num_tslots
        final = self.tslotcomp.state_T(n_ts)
        grad = np.zeros([self.num_tslots, self.num_ctrls])
        if self.mode=="TrDiff":
            evo_f_diff = self.target - final
            for k, onwd_evo, dU, U, fwd_evo in self.tslotcomp.reversed():
                for j in range(n_ctrls):
                    grad[k, j] = -2*self.scale_factor*np.real(
                        (evo_f_diff.T.conj()@onwd_evo@(dU[j]*fwd_evo)).trace())
                    #grad[k, j] = -2*self.scale_factor*np.real(
                    #    (evo_f_diff.T.conj()@onwd_evo@(dU[j]@fwd_evo)).trace())
        elif self.mode=="TrSq":
            trace = np.conj((self.target_d @ final).trace())
            for k, onwd_evo, dU, U, fwd_evo in self.tslotcomp.reversed():
                for j in range(n_ctrls):
                    grad[k, j] = -2*self.scale_factor*\
                        np.real( trace*(self.target_d@onwd_evo@(dU[j]*fwd_evo)).trace() )
                    #grad[k, j] = -2*self.scale_factor*\
                    #    np.real( trace*(self.target_d*onwd_evo*dU[j]*fwd_evo).trace() )
        elif self.mode=="TrAbs":
            fid = (self.target_d@final).trace()
            for k, onwd_evo, dU, U, fwd_evo in self.tslotcomp.reversed():
                for j in range(n_ctrls):
                    grad[k, j] = -self.scale_factor*\
                        np.real( (self.target_d@onwd_evo@(dU[j]*fwd_evo)).trace() \
                                 *np.exp(-1j * np.angle(fid)) )
                    #grad[k, j] = -self.scale_factor*\
                    #    np.real( (self.target_d*onwd_evo*dU[j]*fwd_evo).trace() \
                    #             *np.exp(-1j * np.angle(fid)) )
        grad[np.isnan(grad)] = np.Inf
        return  grad

class FidCompOperatorEarly():
    def __init__(self, tslotcomp, target, mode="TrDiff", scale_factor=0,
                 times=None, weight=None):
        """
        Computes fidelity error and gradient for general system dynamics
        by calculating the the fidelity error as the trace of the overlap
        of the difference between the target and evolution resulting from
        the pulses with the transpose of the same.
        This should provide a distance measure for dynamics described by matrices
        Note the gradient calculation is taken from:
        'Robust quantum gates for open systems via optimal control:
        Markovian versus non-Markovian dynamics'
        Frederik F Floether, Pierre de Fouquieres, and Sophie G Schirmer

        Attributes
        ----------
        scale_factor : float
            The fidelity error calculated is of some arbitary scale. This
            factor can be used to scale the fidelity error such that it may
            represent some physical measure
            If None is given then it is caculated as 1/2N, where N
            is the dimension of the drift, when the Dynamics are initialised.
        """
        self.tslotcomp = tslotcomp
        self.num_ctrls = self.tslotcomp.num_ctrl
        self.num_tslots = self.tslotcomp.n_t
        self.target = target
        self.mode = mode

        if times is None:
            times = np.arange(self.num_tslots, dtype=int)[::-1]+1
        self.times = times
        if weight is None:
            weight = np.ones(len(times))/len(times)
        if len(weight) != len(times):
            raise Exception("The number of weight is not the same as times")
        self.weight = np.array(weight)

        if mode=="TrDiff":
            if not scale_factor:
                self.scale_factor = 1.0 / (2.0*self.target.data.shape[0])
            else:
                self.scale_factor = scale_factor
        elif mode=="TrSq":
            self.target_d = target.T.conj()
            if not scale_factor:
                self.scale_factor = 1.0 / (self.target.data.shape[0])**4
            else:
                self.scale_factor = scale_factor
        elif mode=="TrAbs":
            self.target_d = target.T.conj()
            if not scale_factor:
                self.scale_factor = 1.0 / (self.target.data.shape[0])**2
            else:
                self.scale_factor = scale_factor
        else:
            raise Exception("mode: 'TrDiff', 'TrSq', 'TrAbs'.")

    def costs(self):
        n_ts = self.num_tslots
        fid_err = np.zeros(len(self.times))
        for i, f_state in enumerate(self.tslotcomp.forward(self.times)):
            if self.mode=="TrDiff":
                evo_f_diff = self.target - f_state
                #fid_err[i] = self.scale_factor*np.real((evo_f_diff.T.conj()@evo_f_diff).trace())
                fid_err[i] = self.scale_factor*np.real(np.sum(evo_f_diff.conj()*evo_f_diff))
            elif self.mode=="TrSq":
                fid = (self.target_d@f_state).trace()
                fid_err[i] = 1 - self.scale_factor * np.real(fid * np.conj(fid))
            elif self.mode=="TrAbs":
                fid = (self.target_d@f_state).trace()
                fid_err[i] = 1-self.scale_factor*np.abs(fid)
            if np.isnan(fid_err[i]):
                # Shouldn't this raise an error?
                fid_err[i] = np.Inf
        return np.sum(fid_err*self.weight)

    def costs_t(self):
        n_ts = self.num_tslots
        fid_err = np.zeros(len(self.times))
        for i, f_state in enumerate(self.tslotcomp.forward(self.times)):
            if self.mode=="TrDiff":
                evo_f_diff = self.target - f_state
                #fid_err[i] = self.scale_factor*np.real((evo_f_diff.T.conj()@evo_f_diff).trace())
                fid_err[i] = self.scale_factor*np.real(np.sum(evo_f_diff.conj()*evo_f_diff))
            elif self.mode=="TrSq":
                fid = (self.target_d@f_state).trace()
                fid_err[i] = 1 - self.scale_factor * np.real(fid * np.conj(fid))
            elif self.mode=="TrAbs":
                fid = (self.target_d@f_state).trace()
                fid_err[i] = 1-self.scale_factor*np.abs(fid)
            if np.isnan(fid_err[i]):
                # Shouldn't this raise an error?
                fid_err[i] = np.Inf
        return fid_err*self.weight

    def grad(self):
        n_ctrls = self.num_ctrls
        n_ts = self.num_tslots
        grad = np.zeros([self.num_tslots, self.num_ctrls])
        if self.mode=="TrDiff":
            evo_f_diff = []
            for i, f_state in enumerate(self.tslotcomp.forward(self.times)):
                evo_f_diff.append(-2 * self.scale_factor * self.weight[i] *
                                  (self.target - f_state).T.conj())

            for k, onwd_evo, dU, U, fwd_evo in self.tslotcomp.reversed_cumulative(\
                target=1, times=self.times, phase=evo_f_diff):
                for j in range(n_ctrls):
                    grad[k, j] = np.real((onwd_evo@(dU[j]*fwd_evo)).trace())

        elif self.mode=="TrSq":
            trace = np.zeros(len(self.times),dtype=complex)
            for i, f_state in enumerate(self.tslotcomp.forward(self.times)):
                trace[i] = -2*self.scale_factor*np.conj((self.target_d @ f_state).trace())* self.weight[i]

            for k, onwd_evo, dU, U, fwd_evo in self.tslotcomp.reversed_cumulative(\
                target=self.target_d, times=self.times, phase=trace):
                for j in range(n_ctrls):
                    grad[k, j] = np.real((onwd_evo@(dU[j]*fwd_evo)).trace())

        elif self.mode=="TrAbs":
            phase = np.zeros(len(self.times),dtype=complex)
            for i, f_state in enumerate(self.tslotcomp.forward(self.times)):
                fid = (self.target_d@f_state).trace()
                phase[i] = -self.scale_factor*np.exp(-1j * np.angle(fid))* self.weight[i]

            for k, onwd_evo, dU, U, fwd_evo in self.tslotcomp.reversed_cumulative(\
                target=self.target_d, times=self.times, phase=phase):
                for j in range(n_ctrls):
                    grad[k, j] = np.real((onwd_evo@(dU[j]*fwd_evo)).trace())

        grad[np.isnan(grad)] = np.Inf
        return  grad

class FidCompOperatorForbidden():
    def __init__(self, tslotcomp, forbidden, mode="TrDiff", scale_factor=0,
                 times=None, weight=None):
        """
        Computes fidelity error and gradient for general system dynamics
        by calculating the the fidelity error as the trace of the overlap
        of the difference between the target and evolution resulting from
        the pulses with the transpose of the same.
        This should provide a distance measure for dynamics described by matrices
        Note the gradient calculation is taken from:
        'Robust quantum gates for open systems via optimal control:
        Markovian versus non-Markovian dynamics'
        Frederik F Floether, Pierre de Fouquieres, and Sophie G Schirmer

        Attributes
        ----------
        scale_factor : float
            The fidelity error calculated is of some arbitary scale. This
            factor can be used to scale the fidelity error such that it may
            represent some physical measure
            If None is given then it is caculated as 1/2N, where N
            is the dimension of the drift, when the Dynamics are initialised.
        """
        self.tslotcomp = tslotcomp
        self.num_ctrls = self.tslotcomp.num_ctrl
        self.num_tslots = self.tslotcomp.n_t
        self.target = forbidden
        self.mode = mode

        if times is None:
            times = np.arange(self.num_tslots, dtype=int)[::-1]+1
        self.times = times
        if weight is None:
            weight = np.ones(len(times))/len(times)
        if len(weight) != len(times):
            raise Exception("The number of weight is not the same as times")
        self.weight = np.array(weight)

        if mode=="TrDiff":
            if not scale_factor:
                self.scale_factor = 1.0 / (2.0*self.target.data.shape[0])
            else:
                self.scale_factor = scale_factor
        elif mode=="TrSq":
            self.target_d = forbidden.T.conj()
            if not scale_factor:
                self.scale_factor = 1.0 / (self.target.data.shape[0])**4
            else:
                self.scale_factor = scale_factor
        elif mode=="TrAbs":
            self.target_d = forbidden.T.conj()
            if not scale_factor:
                self.scale_factor = 1.0 / (self.target.data.shape[0])**2
            else:
                self.scale_factor = scale_factor
        else:
            raise Exception("mode: 'TrDiff', 'TrSq', 'TrAbs'.")

    def costs(self):
        n_ts = self.num_tslots
        fid_err = np.zeros(len(self.times))
        for i, f_state in enumerate(self.tslotcomp.forward(self.times)):
            if self.mode=="TrDiff":
                evo_f_diff = self.target - f_state
                #fid_err[i] = -self.scale_factor*np.real((evo_f_diff.T.conj()@evo_f_diff).trace())
                fid_err[i] = -self.scale_factor*np.real(np.sum(evo_f_diff.conj()*evo_f_diff))
            elif self.mode=="TrSq":
                fid = (self.target_d@f_state).trace()
                fid_err[i] = self.scale_factor * np.real(fid * np.conj(fid))
            elif self.mode=="TrAbs":
                fid = (self.target_d@f_state).trace()
                fid_err[i] = self.scale_factor*np.abs(fid)
            if np.isnan(fid_err[i]):
                # Shouldn't this raise an error?
                fid_err[i] = np.Inf
        return np.sum(fid_err*self.weight)

    def costs_t(self):
        n_ts = self.num_tslots
        fid_err = np.zeros(len(self.times))
        for i, f_state in enumerate(self.tslotcomp.forward(self.times)):
            if self.mode=="TrDiff":
                evo_f_diff = self.target - f_state
                #fid_err[i] = -self.scale_factor*np.real((evo_f_diff.T.conj()@evo_f_diff).trace())
                fid_err[i] = -self.scale_factor*np.real(np.sum(evo_f_diff.conj()*evo_f_diff))
            elif self.mode=="TrSq":
                fid = (self.target_d@f_state).trace()
                fid_err[i] = self.scale_factor * np.real(fid * np.conj(fid))
            elif self.mode=="TrAbs":
                fid = (self.target_d@f_state).trace()
                fid_err[i] = self.scale_factor*np.abs(fid)
            if np.isnan(fid_err[i]):
                # Shouldn't this raise an error?
                fid_err[i] = np.Inf
        return fid_err*self.weight

    def grad(self):
        n_ctrls = self.num_ctrls
        n_ts = self.num_tslots
        grad = np.zeros([self.num_tslots, self.num_ctrls])
        if self.mode=="TrDiff":
            evo_f_diff = []
            for i, f_state in enumerate(self.tslotcomp.forward(self.times)):
                evo_f_diff.append(2 * self.scale_factor * self.weight[i] *
                                  (self.target - f_state).T.conj())

            for k, onwd_evo, dU, U, fwd_evo in self.tslotcomp.reversed_cumulative(\
                target=1, times=self.times, phase=evo_f_diff):
                for j in range(n_ctrls):
                    grad[k, j] = np.real((onwd_evo@(dU[j]*fwd_evo)).trace())

        elif self.mode=="TrSq":
            trace = np.zeros(len(self.times),dtype=complex)
            for i, f_state in enumerate(self.tslotcomp.forward(self.times)):
                trace[i] = 2*self.scale_factor*np.conj((self.target_d @ f_state).trace())* self.weight[i]

            for k, onwd_evo, dU, U, fwd_evo in self.tslotcomp.reversed_cumulative(\
                target=self.target_d, times=self.times, phase=trace):
                for j in range(n_ctrls):
                    grad[k, j] = np.real((onwd_evo@(dU[j]*fwd_evo)).trace())

        elif self.mode=="TrAbs":
            phase = np.zeros(len(self.times),dtype=complex)
            for i, f_state in enumerate(self.tslotcomp.forward(self.times)):
                fid = (self.target_d@f_state).trace()
                phase[i] = self.scale_factor*np.exp(-1j * np.angle(fid))* self.weight[i]

            for k, onwd_evo, dU, U, fwd_evo in self.tslotcomp.reversed_cumulative(\
                target=self.target_d, times=self.times, phase=phase):
                for j in range(n_ctrls):
                    grad[k, j] = np.real((onwd_evo@(dU[j]*fwd_evo)).trace())

        grad[np.isnan(grad)] = np.Inf
        return  grad


class FidCompAmp():
    def __init__(self, tslotcomp, weight=0.1, mode=2):
        self.tslotcomp = tslotcomp
        self.num_ctrls = self.tslotcomp.num_ctrl
        self.num_tslots = self.tslotcomp.n_t
        self.mode = mode
        if isinstance(weight, (int, float)):
            weight = weight * np.ones((self.num_tslots,self.num_ctrls))
        elif isinstance(weight, (list, np.ndarray)):
            weight = np.array(weight)
            if len(weight.shape) == 1:
                shape = np.ones((self.num_tslots,self.num_ctrls))
                if weight.shape[0] == self.num_tslots:
                    weight = np.einsum('i,ij->ij', weight, shape)
                elif weight.shape[0] == self.num_ctrls:
                    weight = np.einsum('j,ij->ij', weight, shape)
                else:
                    raise Exception("weight shape not compatible with the amp shape")
            elif weight.shape != (self.num_tslots,self.num_ctrls):
                raise Exception("weight shape not compatible with the amp shape")
        else:
            raise ValueError("weight expected to be one of int, float, list, np.ndarray")
        self.weight = weight

    def costs(self):
        return np.sum(self.tslotcomp._ctrl_amps**self.mode * self.weight)

    def grad(self):
        return self.mode*self.tslotcomp._ctrl_amps**(self.mode-1) * self.weight

class FidCompDAmp():
    def __init__(self, tslotcomp, weight=0.1):
        self.tslotcomp = tslotcomp
        self.num_ctrls = self.tslotcomp.num_ctrl
        self.num_tslots = self.tslotcomp.n_t
        if isinstance(weight, (int, float)):
            weight = weight * np.ones((self.num_tslots-1, self.num_ctrls))
        elif isinstance(weight, (list, np.ndarray)):
            weight = np.array(weight)
            if len(weight.shape) == 1:
                shape = np.ones((self.num_tslots-1, self.num_ctrls))
                if weight.shape[0] == self.num_tslots-1:
                    weight = np.einsum('i,ij->ij', weight, shape)
                elif weight.shape[0] == self.num_ctrls:
                    weight = np.einsum('j,ij->ij', weight, shape)
                else:
                    raise Exception("weight shape not compatible with the amp shape")
            elif weight.shape != (self.num_tslots-1, self.num_ctrls):
                raise Exception("weight shape not compatible with the amp shape")
        else:
            raise ValueError("weight expected to be one of int, float, list, np.ndarray")
        self.weight = weight

    def costs(self):
        return np.sum(np.diff(self.tslotcomp._ctrl_amps, axis=0)**2 * self.weight)

    def grad(self):
        diff = -2 * np.diff(self.tslotcomp._ctrl_amps, axis=0) * self.weight
        out = np.zeros((self.num_tslots, self.num_ctrls))
        out[:-1,:] = diff
        out[1:,:] -= diff
        return out
