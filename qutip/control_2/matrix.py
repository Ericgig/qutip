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
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from qutip.cy.spmatfuncs import spmv
from qutip import Qobj
from qutip.sparse import sp_eigs,sp_expm
from qutip.cy.spmath import (zcsr_adjoint)

# ToDo
# trace  ".tr()"

class falselist_cte:
    """
    To remove special cases in the code, I want to use constant and lists
    in the same way. This is a constant which poses as a list.
    May be useless since a list of the same elements N times is probably
    faster and do not use that much memory.
    """
    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return self.data


class falselist_func:
    """
    To remove special cases in the code, I want to use td_Qobj and lists of
    control_matrix in the same way. This poses as a list but contain a td_Qobj
    and return a control_matrix corresponding to the time
    May be useless since a list of the same elements N times is probably
    faster and do not use that much memory.
    10% slower than a td_Qobj.
    """
    def __init__(self, data, tau, template):
        self.data = data
        self.times = []
        self.template = template
        summ = 0
        for t in tau:
            self.times.append(summ+t*0.5)
            summ += t

    def __getitem__(self, key):
        return self.template(self.data(self.times[key], data=True))


class falselist2d_cte:
    """
    To remove special cases in the code, I want to use constant and lists
    in the same way. This is a constant which poses as a list.
    May be useless since a list of the same elements N times is probably
    faster and do not use that much memory.
    """
    def __init__(self, data):
        self.data = data

    def __getitem__(self, t):
        return self.data[t[1]]

    def __len__(self):
        return len(self.data)


class falselist2d_func:
    """
    To remove special cases in the code, I want to use td_Qobj and lists of
    control_matrix in the same way. This poses as a list but contain a td_Qobj
    and return a control_matrix corresponding to the time
    May be useless since a list of the same elements N times is probably
    faster and do not use that much memory.
    10% slower than a td_Qobj.
    """
    def __init__(self, data, tau, template):
        self.data = data
        self.times = []
        self.template = template
        summ = 0
        for t in tau:
            self.times.append(summ+t*0.5)
            summ += t

    def __len__(self):
        return len(self.data)

    def __getitem__(self, t):
        return self.template(self.data[t[1]](self.times[t[0]], data=True))


matrix_opt = {
    "fact_mat_round_prec":1e-10,
    "_mem_eigen_adj":False,
    "_mem_prop":False,
    "epsilon":1e-6,
    "method":"spectral"}

class control_matrix:
    def __init__(self, obj=None):
        self._size = 0
        self.clean()

    def clean(self):
        self._factormatrix = None
        self._prop_eigen = None
        self._eig_vec = None
        self._eig_vec_dag = None
        self._prop = None

    def __add__(self, other):
        out = self.copy()
        out += other
        return out

    def __mul__(self, other):
        out = self.copy()
        out *= other
        return out

    def __radd__(self, other):
        out = self.copy()
        out += other
        return out

class control_dense(control_matrix):
    def __init__(self, obj=None):
        """
        Dense representation, obj is expected to be a Qobj operators.
        """
        super().__init__()
        self.full = True
        self.data = None
        if isinstance(obj, Qobj):
            self.data = obj.data.todense()
            self._size = self.data.shape[0]
        elif isinstance(obj, np.ndarray):
            self.data = obj
            self._size = self.data.shape[0]
        elif isinstance(obj, sp.csr_matrix):
            self.data = obj
            self._size = obj.shape[0]

    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            out = self.copy()
            out *= other
        elif isinstance(other, Qobj):
            out = self.copy()
            out.data = other.data * out.data
        elif isinstance(other, sp.csr_matrix):
            out = self.copy()
            out.data = other.data * out.data
        elif isinstance(other, np.array):
            out = self.copy()
            out.data = other * out.data
        return out

    def copy(self):
        copy_ = control_dense(self.data.copy())
        return copy_

    def __imul__(self, other):
        if isinstance(other, control_dense):
            self.data = np.matmul(self.data, other.data)
        elif isinstance(other, (int, float, complex)):
            self.data = self.data * other
        elif isinstance(other, np.ndarray):
            self.data = np.matmul(self.data, other)
        else:
            raise NotImplementedError(str(type(other)))
        return self

    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            out = self.copy()
            out *= other
        elif isinstance(other, Qobj):
            out = self.copy()
            out.data = other.data * out.data
        elif isinstance(other, sp.csr_matrix):
            out = self.copy()
            out.data = other.data * out.data
        elif isinstance(other, np.ndarray):
            out = self.copy()
            out.data = np.matmul(other, self.data)
        return out

    def __iadd__(self, other):
        if isinstance(other, control_dense):
            self.data += other.data
        elif isinstance(other, np.ndarray):
            self.data += other
        else:
            raise NotImplementedError(str(type(other)))
        return self

    def dag(self):
        cp = self.copy()
        cp.data = cp.data.T.conj()
        return cp

    def tr(self):
        return self.data.trace()

    def _spectral_decomp(self, tau):
        """
        Calculates the diagonalization of the dynamics generator
        generating lists of eigenvectors, propagators in the diagonalised
        basis, and the 'factormatrix' used in calculating the propagator
        gradient.
        """
        eig_val, eig_vec = la.eig(self.data)

        eig_val_tau = eig_val*tau
        prop_eig = np.exp(eig_val_tau)

        o = np.ones([self._size, self._size])
        eig_val_cols = eig_val_tau*o
        eig_val_diffs = eig_val_cols - eig_val_cols.T

        prop_eig_cols = prop_eig*o
        prop_eig_diffs = prop_eig_cols - prop_eig_cols.T

        degen_mask = np.abs(eig_val_diffs) < matrix_opt["fact_mat_round_prec"]
        eig_val_diffs[degen_mask] = 1
        factors = prop_eig_diffs / eig_val_diffs
        factors[degen_mask] = prop_eig_cols[degen_mask]

        self._factormatrix = factors
        self._prop_eigen = np.diagflat(prop_eig)
        self._eig_vec = eig_vec
        if matrix_opt["_mem_eigen_adj"] is not None:
            self._eig_vec_dag = eig_vec.conj().T

    @property
    def _eig_vec_adj(self):
        if matrix_opt["_mem_eigen_adj"]:
            return self._eig_vec.conj().T
        else:
            return self._eig_vec_dag

    def _exp(self, tau):
        if matrix_opt["_mem_prop"] and self._prop is not None:
            return self._prop

        if matrix_opt["method"] == "spectral":
            if self._eig_vec is None:
                self._spectral_decomp(tau)
            prop = self._eig_vec.dot(self._prop_eigen).dot(self._eig_vec_adj)

        elif matrix_opt["method"] in ["approx", "Frechet"]:
            prop = la.expm(self.data*tau)

        if matrix_opt["_mem_prop"]:
            self._prop = prop
        return prop

    def prop(self,tau):
        return control_dense(self._exp(tau))

    def dexp(self, dirr, tau, compute_expm=False):
        if matrix_opt["method"] == "Frechet":
            A = self.data*tau
            E = dirr.data*tau
            if compute_expm:
                prop, prop_grad = la.expm_frechet(A, E, compute_expm=True)
            else:
                prop_grad = la.expm_frechet(A, E, compute_expm=False)

        elif matrix_opt["method"] == "spectral":
            if self._eig_vec is None:
                self._spectral_decomp(tau)
            if compute_expm:
                prop = self._exp(tau)
            # put control dyn_gen in combined dg diagonal basis
            cdg = self._eig_vec_dag.dot(dirr.data).dot(self._eig_vec)
            # multiply (elementwise) by timeslice and factor matrix
            cdg = np.multiply(cdg*tau, self._factormatrix)
            # Return to canonical basis
            prop_grad = self._eig_vec.dot(cdg).dot(self._eig_vec_adj)

        elif matrix_opt["method"] == "approx":
            dM = (self.data+matrix_opt["epsilon"]*dirr.data)*tau
            dprop = la.expm(dM)
            prop = self._exp(tau)
            prop_grad = (dprop - prop)*(1/matrix_opt["epsilon"])

        if compute_expm:
            return control_dense(prop), control_dense(prop_grad)
        else:
            return control_dense(prop_grad)


class control_sparse(control_matrix):
    def __init__(self, obj=None):
        """
            Sparce representation, obj is expected to be a Qobj operators.
        """
        super().__init__()
        self.full = False
        if isinstance(obj, Qobj):
            self._size = obj.shape[0]
            self.data = obj.data
        elif isinstance(obj, np.ndarray):
            self._size = obj.shape[0]
            self.data = sp.csr_matrix(obj)
        elif isinstance(obj, sp.csr_matrix):
            self._size = obj.shape[0]
            self.data = obj
        self.method = "spectral"

    def copy(self):
        copy_ = control_sparse(self.data.copy())
        return copy_

    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            out = self.copy()
            out *= other
        elif isinstance(other, Qobj):
            out = self.copy()
            out.data = other.data * out.data
        elif isinstance(other, sp.csr_matrix):
            out = self.copy()
            out.data = other.data * out.data
        elif isinstance(other, np.ndarray):
            out = self.copy()
            out.data = sp.csr_matrix(other, self.data)
        return out

    def __imul__(self, other):
        if isinstance(other, control_sparse):
            self.data = self.data * other.data
        elif isinstance(other, (int, float, complex)):
            self.data = self.data * other
        elif isinstance(other, np.ndarray):
            self.data = sp.csr_matrix(spmv(self.data, other))
        else:
            raise NotImplementedError(type(other))
        return self

    def __iadd__(self, other):
        if isinstance(other, control_sparse):
            self.data = self.data + other.data
        else:
            raise NotImplementedError(type(other))
        return self

    def dag(self):
        cp = self.copy()
        cp.data = zcsr_adjoint(self.data)
        return cp

    def tr(self):
        return zcsr_trace(self.data, 0)

    def _spectral_decomp(self, tau):
        """
        Calculates the diagonalization of the dynamics generator
        generating lists of eigenvectors, propagators in the diagonalised
        basis, and the 'factormatrix' used in calculating the propagator
        gradient.
        """
        eig_val, eig_vec = sp_eigs(self.data,0)
        #eig_val, eig_vec = sp_eigs(H.data, H.isherm,
        #                           sparse=self.sparse_eigen_decomp)
        eig_vec = eig_vec.T

        eig_val_tau = eig_val*tau
        prop_eig = np.exp(eig_val_tau)

        o = np.ones([self._size, self._size])
        eig_val_cols = eig_val_tau*o
        eig_val_diffs = eig_val_cols - eig_val_cols.T

        prop_eig_cols = prop_eig*o
        prop_eig_diffs = prop_eig_cols - prop_eig_cols.T

        degen_mask = np.abs(eig_val_diffs) < matrix_opt["fact_mat_round_prec"]
        eig_val_diffs[degen_mask] = 1
        factors = prop_eig_diffs / eig_val_diffs
        factors[degen_mask] = prop_eig_cols[degen_mask]

        self._factormatrix = factors
        self._prop_eigen = np.diagflat(prop_eig)
        self._eig_vec = eig_vec
        if not matrix_opt["_mem_eigen_adj"]:
            self._eig_vec_dag = eig_vec.conj().T

    @property
    def _eig_vec_adj(self):
        if matrix_opt["_mem_eigen_adj"]:
            return self._eig_vec.conj().T
        else:
            return self._eig_vec_dag

    def _exp(self, tau):
        if matrix_opt["_mem_prop"] and self._prop:
            return self._prop
        if matrix_opt["method"] == "spectral":
            if self._eig_vec is None:
                self._spectral_decomp(tau)
            prop = self._eig_vec.dot(self._prop_eigen).dot(self._eig_vec_adj)
        elif matrix_opt["method"] in ["approx", "Frechet"]:
            prop = sp_expm(self.data*tau, sparse=True)
        if matrix_opt["_mem_prop"]:
            self._prop = prop
        return prop

    def prop(self, tau):
        return control_sparse(self._exp(tau))

    def dexp(self, dirr, tau, compute_expm=False):
        if matrix_opt["method"] == "Frechet":
            A = (self.data*tau).toarray()
            E = (dirr.data*tau).toarray()
            if compute_expm:
                prop_dense, prop_grad_dense = la.expm_frechet(A, E)
                prop = sp.csr_matrix(prop_dense)
            else:
                prop_grad_dense = la.expm_frechet(A, E,
                                                  compute_expm=compute_expm)
            prop_grad = sp.csr_matrix(prop_grad_dense)

        elif matrix_opt["method"] == "spectral":
            if self._eig_vec is None:
                self._spectral_decomp(tau)
            if compute_expm:
                prop = self._exp(tau)
            # put control dyn_gen in combined dg diagonal basis
            cdg = self._eig_vec_adj.dot(dirr.data.toarray()).dot(self._eig_vec)
            # multiply (elementwise) by timeslice and factor matrix
            cdg = np.multiply(cdg*tau, self._factormatrix)
            # Return to canonical basis
            prop_grad = self._eig_vec.dot(cdg).dot(self._eig_vec_adj)

        elif matrix_opt["method"] == "approx":
            dM = (self.data+matrix_opt["epsilon"]*dirr.data)*tau
            dprop = sp_expm(dM, sparse=True)
            prop = self._exp(tau)
            prop_grad = (dprop - prop)*(1/matrix_opt["epsilon"])

        if compute_expm:
            return control_sparse(prop), control_sparse(prop_grad)
        else:
            return control_sparse(prop_grad)
