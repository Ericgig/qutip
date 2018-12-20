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
import scipy.sparse.linalg as spla
from scipy.sparse import identity
import scipy.linalg as la
from qutip.cy.spmatfuncs import spmv
from qutip import Qobj
from qutip.sparse import sp_eigs, sp_expm
from qutip.cy.spmath import (zcsr_adjoint, zcsr_trace)


class falselist_cte:
    """
    To remove special cases in the code, I want to use constant and lists
    in the same way. This is a constant which poses as a list.
    May be useless since a list of the same elements N times is probably
    faster and do not use that much memory.

    Parameters
    ----------
    data : object
        The value contained in the falselist
    N : int
        The length of the imitated list.

    Attributes
    ----------
    data : object
        The value contained in the falselist
    shape : tuple of int
        The length of the imitated list.

    """
    def __init__(self, data, N):
        self.data = data
        self.shape = (N,)

    def __getitem__(self, key):
        return self.data

    def __len__(self):
        return self.shape[0]

class falselist_func:
    """
    To remove special cases in the code, I want to use td_Qobj and lists of
    control_matrix in the same way. This poses as a list but contain a td_Qobj
    and return a control_matrix at the time of corresponding index.
    May be useless since a list of the same elements N times is probably
    faster and do not use that much memory.
    10% slower than a td_Qobj.

    Parameters
    ----------
    data : td_Qobj
        Continuous function to use as a list.
    tau : float
        Time between list elements.
    template : control_matrix subclass
        class to return the Qobj data as: sparse or dense matrix
    N : int
        The length of the imitated list.

    Attributes
    ----------
    data : object
        The value contained in the falselist
    times : float
        Time of each list elements.
    template : control_matrix subclass
        class to return the Qobj data as: sparse or dense matrix
    shape : tuple of int
        The length of the imitated list.

    """
    def __init__(self, data, tau, template, N):
        self.data = data
        self.times = []
        self.template = template
        self.shape = (N,)
        summ = 0
        for t in tau:
            self.times.append(summ+t*0.5)
            summ += t

    def __getitem__(self, key):
        return self.template(self.data(self.times[key], data=True))

    def __len__(self):
        return self.shape[0]

class falselist2d_cte:
    """
    To remove special cases in the code, I want to use constant and lists
    in the same way. This is a constant which poses as a list.
    May be useless since a list of the same elements N times is probably
    faster and do not use that much memory.

    Parameters
    ----------
    data : list
        1d list of to poses as a 2d list.
    N : int
        The length of the imitated list.

    Attributes
    ----------
    data : object
        The value contained in the falselist
    shape : tuple of int
        The length of the imitated list.
    """
    def __init__(self, data, N):
        self.data = data
        self.shape = (N, len(data))

    def __getitem__(self, key):
        return self.data[key[1]]

    def __len__(self):
        return len(self.shape[0] * self.shape[1])

class falselist2d_func:
    """
    To remove special cases in the code, I want to use td_Qobj and lists of
    control_matrix in the same way. This poses as a list but contain a td_Qobj
    and return a control_matrix corresponding to the time
    May be useless since a list of the same elements N times is probably
    faster and do not use that much memory.
    10% slower than a td_Qobj.

    Parameters
    ----------
    data : td_Qobj
        Continuous function to use as a list.
    tau : float
        Time between list elements.
    template : control_matrix subclass
        class to return the Qobj data as: sparse or dense matrix
    N : int
        The length of the imitated list.

    Attributes
    ----------
    data : object
        The value contained in the falselist
    times : float
        Time of each list elements.
    template : control_matrix subclass
        class to return the Qobj data as: sparse or dense matrix
    shape : tuple of int
        The length of the imitated list.
    """
    def __init__(self, data, tau, template, N):
        self.data = data
        self.times = []
        self.template = template
        self.shape = (N,len(data))
        summ = 0
        for t in tau:
            self.times.append(summ+t*0.5)
            summ += t

    def __getitem__(self, t):
        return self.template(self.data[t[1]](self.times[t[0]], data=True))

    def __len__(self):
        return len(self.shape[0]*self.shape[1])


matrix_opt = {
    "fact_mat_round_prec":1e-10,
    "_mem_eigen_adj":False,
    "_mem_prop":False,
    "epsilon":1e-6,
    "method":"Frechet",
    "sparse2dense":False,
    "sparse_exp":True}

class control_matrix:
    """
    Matrix for qutip/control.
    Offer a identical interface to use space and dense matrices.
    This is parent function which define the interface.

    Parameters
    ----------
    obj : matrix-like : td_Qobj, sp.csr_matrix, np,array 2d
        matrix

    Methods
    -------
    dag:
        Adjoint (dagger) of matrix.
    tr:
        Trace of matrix.
    prop(tau):
        The exponential of the matrix
    dexp(dirr, tau, compute_expm=False)
        The deriative of the exponential in the given dirrection

    """
    def __init__(self, obj=None):
        self._size = 0
        self.clean()

    def copy(self):
        pass

    def clean(self):
        self._factormatrix = None
        self._prop_eigen = None
        self._eig_vec = None
        self._eig_vec_dag = None
        self._prop = None

    def __imul__(self, other):
        """dummy"""
        return self

    def __mul__(self, other):
        """dummy"""
        return self

    def __rmul__(self, other):
        """dummy"""
        return self

    def __iadd__(self, other):
        """dummy"""
        return self

    def __isub__(self, other):
        """dummy"""
        return self

    def __add__(self, other):
        out = self.copy()
        out += other
        return out

    def __sub__(self, other):
        out = self.copy()
        out -= other
        return out

    def dag(self):
        """Adjoint (dagger) of the matrix
        Returns
        -------
        dag : control_matrix
            Adjoint of matrix

        Notes
        -----
        Dummy
        """
        return self

    def tr(self):
        """Trace of the matrix
        Returns
        -------
        trace : complex

        Notes
        -----
        Dummy
        """
        return 0j

    def prop(self, tau):
        """Propagator in the time interval

        Parameters
        ----------
        tau : double
            time interval

        Returns
        -------
        prop : control_matrix
            exp(self*tau)

        Notes
        -----
        Dummy
        """
        return self*tau+self*tau*self*tau/2+1

    def dexp(self, dirr, tau, compute_expm=False):
        """The deriative of the exponential in the given dirrection

        Parameters
        ----------
        dirr : control_matrix

        Returns
        -------
        prop : control_matrix
            exp(self*tau) (Optional, if compute_expm)
        derr : control_matrix
            (exp((self+dirr*dt)*tau)-exp(self*tau)) / dt

        Notes
        -----
        Dummy
        """
        dt = 1/1000
        prop = self.prop(tau)
        dprop = (self + dirr*dt).prop(tau)
        derr = (dprop-prop) * 1000
        if compute_expm:
            return prop, derr
        else:
            return derr


class control_dense(control_matrix):
    def __init__(self, obj=None):
        """
        Dense representation, obj is expected to be a Qobj operators.
        """
        super().__init__()
        self.full = True
        self.data = None
        if isinstance(obj, Qobj):
            self.data = np.array(obj.data.todense())
            self._size = self.data.shape[0]
        elif isinstance(obj, np.ndarray):
            self.data = obj
            self._size = self.data.shape[0]
        elif isinstance(obj, sp.csr_matrix):
            self.data = obj.toarray()
            self._size = obj.shape[0]

    def copy(self):
        copy_ = control_dense(self.data.copy())
        return copy_

    def __imul__(self, other):
        if isinstance(other, control_dense):
            self.data = self.data @ other.data
        elif isinstance(other, (int, float, complex)):
            self.data *= other
        #elif isinstance(other, np.ndarray):
        #    self.data = np.matmul(self.data, other)
        else:
            raise NotImplementedError(str(type(other)))
        return self

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            out = self.copy()
            out *= other
        elif isinstance(other, np.ndarray):
            out = self.data @ other
        elif isinstance(other, control_dense):
            out = self.copy()
            out.data = out.data @ other.data
        else:
            raise NotImplementedError(str(type(other)))
        return out

    def __rmul__(self, other):
        if isinstance(other, np.ndarray):
            out = other @ self.data
        elif isinstance(other, (int, float, complex)):
            out = self.copy()
            out *= other
        else:
            raise NotImplementedError(str(type(other)))
        return out

    def __iadd__(self, other):
        if isinstance(other, control_dense):
            self.data += other.data
        elif isinstance(other, np.ndarray):
            self.data += other
        else:
            raise NotImplementedError(str(type(other)))
        return self

    def __isub__(self, other):
        if isinstance(other, control_dense):
            self.data -= other.data
        elif isinstance(other, np.ndarray):
            self.data -= other
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

        elif matrix_opt["method"] == "first_order":
            prop = np.eye(self.data.shape[0]) + self.data * tau

        elif matrix_opt["method"] == "second_order":
            prop = np.eye(self.data.shape[0]) + self.data * tau
            prop += self.data @ self.data * (tau * tau * 0.5)

        elif matrix_opt["method"] == "third_order":
            B = self.data * tau
            prop = np.eye(self.data.shape[0]) + B
            BB = B @ B * 0.5
            prop += BB
            prop += BB @ B * 0.3333333333333333333

        if matrix_opt["_mem_prop"]:
            self._prop = prop
        return prop

    def prop(self, tau):
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

        elif matrix_opt["method"] == "first_order":
            if compute_expm:
                prop = self._exp(tau)
            prop_grad = dirr.data * tau

        elif matrix_opt["method"] == "second_order":
            if compute_expm:
                prop = self._exp(tau)
            prop_grad = dirr.data * tau
            prop_grad += (self.data @ dirr.data + dirr.data @ self.data) \
                            * (tau * tau * 0.5)

        elif matrix_opt["method"] == "third_order":
            if compute_expm:
                prop = self._exp(tau)
            prop_grad = dirr.data * tau
            prop_grad += (self.data @ dirr.data + dirr.data @ self.data) \
                            * tau * tau * 0.5
            prop_grad += (self.data @ self.data @ dirr.data +
                          dirr.data @ self.data @ self.data +
                          self.data @ dirr.data @ self.data ) \
                            * (tau * tau * tau * 0.16666666666666666)

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
        #elif isinstance(other, Qobj):
        #    out = self.copy()
        #    out.data = other.data * out.data
        #elif isinstance(other, sp.csr_matrix):
        #    out = self.copy()
        #    out.data = other.data * out.data
        elif isinstance(other, np.ndarray):
            if len(other.shape) == 1:
                out = spmv(self.data.T, other)
            else:
                out = (self.data.T * other.T).T
        return out

    def __mul__(self, other):
        if isinstance(other, control_sparse):
            out = self.copy()
            out.data = self.data * other.data
        elif isinstance(other, (int, float, complex)):
            out = self.copy()
            out.data = self.data * other
        elif isinstance(other, np.ndarray):
            if len(other.shape) == 1:
                out = spmv(self.data, other)
            else:
                out = self.data * other

        else:
            raise NotImplementedError(type(other))
        return out

    def __imul__(self, other):
        if isinstance(other, control_sparse):
            self.data = self.data * other.data
        elif isinstance(other, (int, float, complex)):
            self.data = self.data * other
        #elif isinstance(other, np.ndarray):
        #    self.data = sp.csr_matrix(spmv(self.data, other))
        else:
            raise NotImplementedError(type(other))
        return self

    def __iadd__(self, other):
        if isinstance(other, control_sparse):
            self.data = self.data + other.data
        else:
            raise NotImplementedError(type(other))
        return self

    def __isub__(self, other):
        if isinstance(other, control_sparse):
            self.data = self.data - other.data
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
            if matrix_opt["sparse2dense"]:
                prop = la.expm(self.data.toarray()*tau)
            else:
                prop = sp_expm(self.data*tau,
                               sparse=matrix_opt["sparse_exp"])
        elif matrix_opt["method"] == "first_order":
            if matrix_opt["sparse2dense"]:
                prop = np.eye(self.data.shape[0]) + self.data.toarray() * tau
            else:
                prop = identity(self.data.shape[0], format='csr') + \
                        self.data * tau
        elif matrix_opt["method"] == "second_order":
            if matrix_opt["sparse2dense"]:
                M = self.data.toarray() * tau
                prop = np.eye(self.data.shape[0]) + M
                prop += M @ M * 0.5
            else:
                M = self.data * tau
                prop = identity(self.data.shape[0], format='csr') + M
                prop += M * M * 0.5
        elif matrix_opt["method"] == "third_order":
            if matrix_opt["sparse2dense"]:
                B = self.data.toarray() * tau
                prop = np.eye(self.data.shape[0]) + B
                BB = B @ B * 0.5
                prop += BB
                prop += BB @ B * 0.3333333333333333333
            else:
                B = self.data * tau
                prop = identity(self.data.shape[0], format='csr') + B
                BB = B * B * 0.5
                prop += BB
                prop += BB * B * 0.3333333333333333333

        if matrix_opt["_mem_prop"]:
            self._prop = prop
        return prop

    def prop(self, tau):
        if matrix_opt["sparse2dense"]:
            return control_dense(self._exp(tau))
        return control_sparse(self._exp(tau))

    def dexp(self, dirr, tau, compute_expm=False):
        if matrix_opt["method"] == "Frechet":
            A = (self.data*tau).toarray()
            E = (dirr.data*tau).toarray()
            if compute_expm:
                prop_dense, prop_grad_dense = la.expm_frechet(A, E)
                prop = prop_dense
                # prop = sp.csr_matrix(prop_dense)
            else:
                prop_grad_dense = la.expm_frechet(A, E,
                                                  compute_expm=compute_expm)
            prop_grad = prop_grad_dense
            # prop_grad = sp.csr_matrix(prop_grad_dense)

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
            if matrix_opt["sparse2dense"]:
                dM = (self.data.toarray() + \
                      matrix_opt["epsilon"] * dirr.data.toarray()) * tau
                dprop = la.expm(dM)
                prop = self._exp(tau)
                prop_grad = (dprop - prop)*(1/matrix_opt["epsilon"])
            else:
                dM = (self.data + matrix_opt["epsilon"]*dirr.data)*tau
                dprop = sp_expm(dM, sparse=matrix_opt["sparse_exp"])
                prop = self._exp(tau)
                prop_grad = (dprop - prop)*(1/matrix_opt["epsilon"])

        elif matrix_opt["method"] == "first_order":
            if compute_expm:
                prop = self._exp(tau)
            prop_grad = dirr.data * tau

        elif matrix_opt["method"] == "second_order":
            if compute_expm:
                prop = self._exp(tau)
            prop_grad = dirr.data * tau
            prop_grad += (self.data * dirr.data + dirr.data * self.data) \
                            * (tau * tau * 0.5)

        elif matrix_opt["method"] == "third_order":
            if compute_expm:
                prop = self._exp(tau)
            prop_grad = dirr.data * tau
            A = self.data * dirr.data
            B = dirr.data * self.data
            prop_grad += (A + B)  * (tau * tau * 0.5)
            prop_grad += (self.data * A + A * self.data + B * self.data ) * \
                            (tau * tau * tau * 0.16666666666666666)

        if compute_expm:
            if matrix_opt["sparse2dense"]:
                return control_dense(prop), control_dense(prop_grad)
            else:
                return control_sparse(prop), control_sparse(prop_grad)
        else:
            if matrix_opt["sparse2dense"]:
                return control_dense(prop_grad)
            else:
                return control_sparse(prop_grad)
