# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, QuSTaR.
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
import os
import numpy as np
from qutip.settings import settings as qset

from ..core import QobjEvoFunc, Qobj, QobjEvo, sprepost, liouvillian, Cubic_Spline, coefficient
from ..core.qobjevo import QobjEvoBase
from ._brtools import (bloch_redfield_tensor, CBR_RHS, Spectrum, Spectrum_Str,
                       Spectrum_array, Spectrum_func_t)
from ..core import data as _data


__all__ = ['bloch_redfield']

def make_spectra(f):
    if isinstance(f, Spectrum):
        return f
    elif isinstance(f, str):
        coeff = coefficient(f, args={"w":0})
        return Spectrum_Str(coeff)
    elif isinstance(f, (np.ndarray, Cubic_Spline)):
        coeff = coefficient(f)
        return Spectrum_array(coeff)
    elif callable(f):
        try:
            f(0, 0)
            return Spectrum_func_t(f)
        except Exception:
            return Spectrum(f)

def _legacy_read_a_op(a_ops):
    parsed_a_ops = []
    for op, spec in a_ops:
        if isinstance(op, Qobj):
            if isinstance(spec, str):
                parsed_a_ops.append((op, coefficient(spec, args={'w':0})))
            elif isinstance(spec, tuple):
                if isinstance(spec[0], str):
                    freq_responce = coefficient(spec[0], args={'w':0})
                elif isinstance(spec[0], Cubic_Spline):
                    freq_responce = Coeff_t2w(coefficient(spec[0]))
                else:
                    raise Exception('Invalid bath-coupling specification.')
                if isinstance(spec[0], str):
                    time_responce = coefficient(spec[0], args={'w':0})
                elif isinstance(spec[0], Cubic_Spline):
                    time_responce = coefficient(spec[0])
                else:
                    raise Exception('Invalid bath-coupling specification.')
                parsed_a_ops.append((op, freq_responce * time_responce))
        elif isinstance(op, tuple):
            qobj1, qobj2 = op

            



def bloch_redfield(H, a_ops, c_ops=[],
                   use_secular=True, sec_cutoff=0.1, atol=qset.core['atol'],
                   legacy_a_ops=False):
    ops = [H] + [op for op, spec in a_ops] + c_ops
    Qobj_out = all(isinstance(op, Qobj) for op in ops)

    H = QobjEvo(H)
    if legacy_a_ops:
        FuturWarning
        for op, spec in a_ops:
    else:
        for op, spec in a_ops:


    if not all(isinstance(op, (Qobj, QobjEvo)) for op in ops):
        raise TypeError("Operators must be Qobj of QobjEvo")
    if not all(op.dims == H.dims for op, spec in a_ops):
        raise TypeError("dimension of a_ops and H do not match")
    any_QEvo = any(isinstance(op, QobjEvoBase) for op in ops)
    any_td = any(isinstance(op, QobjEvoBase) and not op.const for op in ops)

    if not any_td:
        H = H(0) if isinstance(H, QobjEvoBase) else H
        cops = [(cop(0) if isinstance(cop, QobjEvoBase) else cop)
                 for cop in c_ops]
        aops = [(aop(0) if isinstance(aop, QobjEvoBase) else aop,
                 make_spectra(spec))
                for aop, spec in a_ops]
        R, ekets = bloch_redfield_tensor(H, aops, cops, use_secular,
                                         sec_cutoff, atol)
        base = np.hstack([psi.full() for psi in ekets])
        S = Qobj(_data.adjoint(_data.create(base)), dims=H.dims)
        R = sprepost(S.dag(), S) @ R @ sprepost(S, S.dag())
        if any_QEvo:
            R = QobjEvo(R)
        return R

    #elif isinstance(H, Qobj) or H.const:
        # H cte, can precompute the eigenvectors
        # pass

    else:
        return BR_QobjEvoFunc(H, a_ops, c_ops,
                              use_secular, sec_cutoff, atol)


class BR_QobjEvoFunc(QobjEvoFunc):
    def __init__(self, H, a_ops, c_ops, use_secular, sec_cutoff, atol):
        if isinstance(H, Qobj):
            H = QobjEvo(H)
        self.H = H.to(_data.Dense)
        self.args = {}
        if c_ops:
            self.dissipator = liouvillian(H=None, c_ops=c_ops)
        else:
            self.dissipator = None
        if isinstance(self.dissipator, Qobj):
            self.dissipator = QobjEvo(self.dissipator)
        self.a_ops = []
        self.spectra = []
        for aop, spec in a_ops:
            self.a_ops.append(QobjEvo(aop).to(_data.Dense)
                              if isinstance(aop, Qobj) else aop.to(_data.Dense))
            self.spectra.append(make_spectra(spec))

        self.opt = (use_secular, sec_cutoff, atol)
        self.compiled_qobjevo = CBR_RHS(self.H, self.a_ops, self.spectra,
                                        self.dissipator, *self.opt)

        self.cte = self.compiled_qobjevo.call(0)
        self.operation_stack = []
        self._shifted = False
        self.const = False

    def _get_qobj(self, t, args={}, data=False):
        if self._shifted:
            t += self.args["_t0"]
        if args:
            a_ops = zip([aop(t, args) for aop in self.a_ops], self.spectra)
            c_ops = [cop(t, args) for cop in self.c_ops]
            R, ekets = bloch_redfield_tensor(self.H(t, args), a_ops, c_ops,
                                             *self.opt)
            base = np.hstack([psi.full() for psi in ekets])
            S = Qobj(_data.adjoint(_data.create(base)), dims=H.dims)
            R = sprepost(S.dag(), S) @ R @ sprepost(S, S.dag())
            for transform in self.operation_stack:
                R = transform(R, t, args)
        else:
            R = self.compiled_qobjevo.call(t, False, False)
            for transform in self.operation_stack:
                R = transform(R, t, self.args)
        return R

    def arguments(self, new_args):
        if not isinstance(new_args, dict):
            raise TypeError("The new args must be in a dict")
        self.args.update(new_args)
        self.H.arguments(new_args)
        for cop in self.c_ops:
            cop.arguments(new_args)
        for aop in self.a_ops:
            aop.arguments(new_args)

    def copy(self):
        new = BR_QobjEvoFunc.__new__(BR_QobjEvoFunc)
        new.__dict__ = self.__dict__.copy()
        new.H = self.H.copy()
        new.dissipator = self.dissipator.copy()
        new.a_ops = [aop.copy() for aop in self.a_ops]
        new.compiled_qobjevo = CBR_RHS(new.H, new.a_ops, new.spectra,
                                       new.dissipator, *self.opt)
        new.cte = self.cte.copy()
        new.operation_stack = [oper.copy() for oper in self.operation_stack]
        new.args = self.args.copy()
        return new

    def __reduce__(self):
        return BR_QobjEvoFunc, (self.H, zip(self.a_ops, self.spectra),
                                self.dissipator, *self.opt)

    def mul(self, t, mat):
        """
        Product of the operator quantum object at time t
        with the given matrix state.
        """
        was_Qobj = False
        was_vec = False
        was_data = False
        if not isinstance(t, (int, float)):
            raise TypeError("the time need to be a real scalar")

        if isinstance(mat, Qobj):
            if self.dims[1] != mat.dims[0]:
                raise Exception("Dimensions do not fit")
            was_Qobj = True
            dims = [self.dims[0], mat.dims[1]]
            mat = mat.data

        elif isinstance(mat, _data.Data):
            was_data = True

        elif isinstance(mat, np.ndarray):
            if mat.ndim == 1:
                mat = _data.dense.fast_from_numpy(mat)
                was_vec = True
            elif mat.ndim == 2:
                mat = _data.dense.fast_from_numpy(mat)
            else:
                raise Exception("The matrice must be 1d or 2d")

        else:
            raise TypeError("The vector must be an array or Qobj")

        if mat.shape[0] != self.shape[1]:
            raise Exception("The length do not match")

        if self.operation_stack:
            out = _data.matmul(self.__call__(t, data=True), mat)
        else:
            out = self.compiled_qobjevo.matmul(t, mat)

        if was_Qobj:
            return Qobj(out, dims=dims)
        elif was_data:
            return out
        elif was_vec:
            return out.as_ndarray()[:, 0]
        else:
            return out.as_ndarray()

    def eigenbasis(self, t):
        if self._shifted:
            t += self.args["_t0"]

        if args:
            H = self.H(t, args)
            a_ops = zip([aop(t, args) for aop in self.a_ops], self.spectra)
            c_ops = [cop(t, args) for cop in self.c_ops]
            R, ekets = bloch_redfield_tensor(H, a_ops, c_ops, *self.opt)
        else:
            R, ekets = self.compiled_qobjevo.call(t, False, True)

        if self.operation_stack:
            # TODO raise efficiency warning?
            base = np.hstack([psi.full() for psi in ekets])
            S = Qobj(_data.adjoint(_data.create(base)), dims=H.dims)
            SS = sprepost(S.dag(), S)
            R = SS @ R @ SS.dag()
            for transform in self.operation_stack:
                R = transform(R, t, args)
            R = SS.dag() @ R @ SS

        return R, ekets
