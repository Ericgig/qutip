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
# import scipy.linalg as la
import timeit
# QuTiP
from qutip import Qobj
# QuTiP logging
import qutip.logging_utils as logging
logger = logging.get_logger()
# QuTiP control modules
import qutip.control.errors as errors



class FidCompUnitary():
    def __init__(self, parent, phase_option):
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
                SU - global phase included
        """
        self.target_d = target.dag()
        self.num_ctrls = num_ctrls
        self.num_tslots = num_tslots
        self.tslotcomp = parent.tslotcomp
        if not phase_option in ["SU","PSU"]:
            raise Exception("Invalid phase_option for FidCompUnitary.")
        self.SU = phase_option == "SU"
        if self.SU:
            self.dimensional_norm = np.real((self.target_d*target).tr())
        else:
            self.dimensional_norm = np.abs((self.target_d*target).tr())

    def costs(self):
        n_ctrls = self.num_ctrls
        n_ts = self.num_tslots
        final = self.tslotcomp.state_T(n_ts)
        fidelity_prenorm = (self.target_d*final).tr()

        # create n_ts x n_ctrls zero array for grad start point
        grad = np.zeros([n_ts, n_ctrls], dtype=complex)
        # loop through all ctrl timeslots calculating gradients
        for k, onto_evo, dU, U, fwd_evo in self.tslotcomp.reversed_onto:
            for j in range(n_ctrls):
                grad[k, j] = (onto_evo*dU[j]*fwd_evo).tr()

        if self.SU:
            fidelity = np.real(fidelity_prenorm) / self.dimensional_norm
            grad_normalized = np.real(grad) / self.dimensional_norm
        else:
            fidelity = np.abs(fidelity_prenorm) / self.dimensional_norm
            grad_normalized = np.real(grad / self.dimensional_norm *
                                      np.exp(-1j * np.angle(fidelity_prenorm)))

        return np.abs(1 - fidelity), grad_normalized


class FidCompTraceDiff():
    def __init__(self, parent, scale_factor=0):
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
        self.target = parent.target
        self.num_ctrls = parent.num_ctrls
        self.num_tslots = parent.num_tslots
        self.tslotcomp = parent.tslotcomp
        if not scale_factor:
            self.scale_factor = 1.0 / (2.0*self.target.shape[0])
        else:
            self.scale_factor = scale_factor

    def costs(self):
        n_ctrls = self.num_ctrls
        n_ts = self.num_tslots
        final = self.tslotcomp.state_T(n_ts)
        evo_f_diff = self.target - final

        fid_err = self.scale_factor*np.real((evo_f_diff.dag()*evo_f_diff).tr())
        if np.isnan(fid_err):
            fid_err = np.Inf

        # create n_ts x n_ctrls zero array for grad start point
        grad = np.zeros([n_ts, n_ctrls])
        # loop through all ctrl timeslots calculating gradients
        for k, onwd_evo, dU, U, fwd_evo in self.tslotcomp.reversed_onwd:
            for j in range(n_ctrls):
                g = -2*self.scale_factor*np.real(
                    (evo_f_diff.dag()*onwd_evo*dU[j]*fwd_evo).tr())
                if np.isnan(g):
                    g = np.Inf
                grad[k, j] = g
        return fid_err, grad
