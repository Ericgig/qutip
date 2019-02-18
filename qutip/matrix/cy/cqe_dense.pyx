#!python
#cython: language_level=3
# distutils: language = c++
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
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
"""
Contain the cython interface of QobjEvo.
The parent class "CQobjEvo" set the interface.

CQobjCte:
  QobjEvo that does not depend on times.
  sparse matrix

CQobjCteDense:
  QobjEvo that does not depend on times.
  dense matrix
  - Hidden feature in the sense that it's not really documented and need to be
    explicitly used. Does not seems to results in significant speedup.

CQobjEvoTd:
  QobjEvo that does depend on times.
  sparse matrix

CQobjEvoTdDense:
  QobjEvo that does depend on times.
  dense matrix
  - Hidden feature in the sense that it's not really documented and need to be
    explicitly used. Does not seems to results in significant speedup.

CQobjEvoTdMatched:
  QobjEvo that does depend on times.
  sparse matrix with 0s
  - Use sparce matrices that all have the same "filling". Therefore addition of
    such matrices become a vector addition.
  - Hidden feature/ experimental.
    It reasult in a speedup in some rare cases.

In omp/cqobjevo_omp:
  Variantes which use parallel mat*vec and mat*mat product
  - CQobjCteOmp
  - CQobjEvoTdOmp
  - CQobjEvoTdMatchedOmp
"""
import numpy as np
import scipy.sparse as sp
cimport numpy as np
import cython
cimport cython
from qutip.qobj import Qobj
from qutip.cy.spmath cimport _zcsr_add_core
from qutip.cy.spmatfuncs cimport spmvpy, _spmm_c_py, _spmm_f_py
from qutip.cy.spmath import zcsr_add
from qutip.cy.cqobjevo_factor cimport CoeffFunc
cimport libc.math

from qutip.cy.complex_math cimport *
np.import_array()

cdef extern from "numpy/arrayobject.h" nogil:
    void PyDataMem_FREE(void * ptr)
    void PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyDataMem_NEW(size_t size)

cdef extern from "Python.h":
    object PyLong_FromVoidPtr(void *)
    void* PyLong_AsVoidPtr(object)


cdef class CQobjCteDense(CQobjEvo):
    def set_data(self, cte):
        self.shape0 = cte.shape[0]
        self.shape1 = cte.shape[1]
        self.dims = cte.dims
        self.cte = cte.data.toarray()
        self.super = cte.issuper

    def __getstate__(self):
        return (self.shape0, self.shape1, self.dims,
                self.super, np.array(self.cte))

    def __setstate__(self, state):
        self.shape0 = state[0]
        self.shape1 = state[1]
        self.dims = state[2]
        self.super = state[3]
        self.cte = state[4]

    def call(self, double t, int data=0):
        if data:
            return sp.csr_matrix(self.cte, dtype=complex, copy=True)
        else:
            return Qobj(self.cte, dims = self.dims)

    def call_with_coeff(self, complex[::1] coeff, int data=0):
        if data:
            return sp.csr_matrix(self.cte, dtype=complex, copy=True)
        else:
            return Qobj(self.cte, dims = self.dims)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_vec(self, double t, complex* vec, complex* out):
        cdef int i, j
        cdef complex* ptr
        for i in range(self.shape0):
            ptr = &self.cte[i,0]
            for j in range(self.shape1):
                out[i] += ptr[j]*vec[j]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_matf(self, double t, complex* mat, complex* out,
                        int nrow, int ncol):
        cdef int i, j, k
        cdef complex* ptr = &self.cte[0,0]
        for i in range(self.shape0):
            for j in range(ncol):
                for k in range(nrow):
                    out[i + j*self.shape0] += ptr[i*nrow + k]*mat[k + j*nrow]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_matc(self, double t, complex* mat, complex* out,
                        int nrow, int ncol):
        cdef int i, j, k
        cdef complex* ptr = &self.cte[0,0]
        for i in range(self.shape0):
            for j in range(ncol):
                for k in range(nrow):
                    out[i*ncol + j] += ptr[i*nrow + k]*mat[k*ncol + j]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect(self, double t, complex* vec, int isherm):
        cdef int i, j
        cdef complex dot = 0
        for i in range(self.shape0):
          for j in range(self.shape1):
            dot += conj(vec[i])*self.cte[i,j]*vec[j]

        if isherm:
            return real(dot)
        else:
            return dot

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect_super(self, double t, complex* vec, int isherm):
        cdef int row, i
        cdef int num_rows = self.shape0
        cdef int n = <int>libc.math.sqrt(num_rows)
        cdef complex dot = 0.0
        for row from 0 <= row < num_rows by n+1:
          for i in range(self.shape1):
            dot += self.cte[row,i]*vec[i]

        if isherm:
            return real(dot)
        else:
            return dot


cdef class CQobjEvoTdDense(CQobjEvo):
    def set_data(self, cte, ops):
        cdef int i, j, k
        self.shape0 = cte.shape[0]
        self.shape1 = cte.shape[1]
        self.dims = cte.dims
        self.super = cte.issuper
        self.num_ops = len(ops)
        self.cte = cte.data.toarray()
        self.ops = np.zeros((self.num_ops, self.shape0, self.shape1),
                            dtype=complex)
        self.data_t = np.empty((self.shape0, self.shape1), dtype=complex)
        self.data_ptr = &self.data_t[0,0]
        self.coeff = np.empty((self.num_ops,), dtype=complex)
        self.coeff_ptr = &self.coeff[0]
        for i, op in enumerate(ops):
          oparray = op[0].data.toarray()
          for j in range(self.shape0):
            for k in range(self.shape1):
              self.ops[i,j,k] = oparray[j,k]

    def __getstate__(self):
        return (self.shape0, self.shape1, self.dims, self.super,
                self.factor_use_cobj, self.factor_cobj,
                self.factor_func, self.num_ops,
                np.array(self.cte), np.array(self.ops))

    def __setstate__(self, state):
        self.shape0 = state[0]
        self.shape1 = state[1]
        self.dims = state[2]
        self.super = state[3]
        self.factor_use_cobj = state[4]
        if self.factor_use_cobj:
            self.factor_cobj = <CoeffFunc> state[5]
        self.factor_func = state[6]
        self.num_ops = state[7]
        self.cte = state[8]
        self.ops = state[9]
        self.data_t = np.empty((self.shape0, self.shape1), dtype=complex)
        self.data_ptr = &self.data_t[0,0]
        self.coeff = np.empty((self.num_ops,), dtype=complex)
        self.coeff_ptr = &self.coeff[0]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _call_core(self, complex[:,::1] out, complex* coeff):
        cdef int i, j
        cdef complex* ptr
        cdef complex* out_ptr
        #copy(self.cte, out)
        ptr = &self.cte[0,0]
        out_ptr = &out[0,0]
        for i in range(self.shape0 * self.shape0):
            out_ptr[i] = ptr[i]
        for i in range(self.num_ops):
            ptr = &self.ops[i,0,0]
            for j in range(self.shape0 * self.shape0):
                out_ptr[j] += ptr[j]*coeff[i]

    def call(self, double t, int data=0):
        cdef np.ndarray[complex, ndim=2] data_t = \
                  np.empty((self.shape0, self.shape1), dtype=complex)
        self._factor(t)
        self._call_core(data_t, self.coeff_ptr)

        if data:
            return sp.csr_matrix(data_t, dtype=complex, copy=True)
        else:
            return Qobj(data_t, dims = self.dims)

    def call_with_coeff(self, complex[::1] coeff, int data=0):
        cdef np.ndarray[complex, ndim=2] data_t = \
                    np.empty((self.shape0, self.shape1), dtype=complex)
        self._call_core(data_t, &coeff[0])
        if data:
            return sp.csr_matrix(data_t, dtype=complex, copy=True)
        else:
            return Qobj(data_t, dims = self.dims)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_vec(self, double t, complex* vec, complex* out):
        self._factor(t)
        self._call_core(self.data_t, self.coeff_ptr)

        cdef int i, j
        for i in range(self.shape0):
            for j in range(self.shape1):
                out[i] += self.data_t[i,j]*vec[j]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_matf(self, double t, complex* mat, complex* out,
                        int nrow, int ncol):
        cdef int i, j, k
        self._factor(t)
        self._call_core(self.data_t, self.coeff_ptr)
        for i in range(self.shape0):
            for j in range(ncol):
                for k in range(nrow):
                    out[i + j*self.shape0] += self.data_ptr[i*nrow + k] *\
                                              mat[k + j*nrow]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_matc(self, double t, complex* mat, complex* out,
                        int nrow, int ncol):
        cdef int i, j, k
        self._factor(t)
        self._call_core(self.data_t, self.coeff_ptr)
        for i in range(self.shape0):
            for j in range(ncol):
                for k in range(nrow):
                    out[i*ncol + j] += self.data_ptr[i*nrow + k]*mat[k*ncol + j]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect(self, double t, complex* vec, int isherm):
        cdef int row
        cdef complex dot = 0
        self._factor(t)
        self._call_core(self.data_t, self.coeff_ptr)
        for i in range(self.shape0):
          for j in range(self.shape1):
            dot += conj(vec[i])*self.data_t[i,j]*vec[j]
        if isherm:
            return real(dot)
        else:
            return dot

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect_super(self, double t, complex* vec, int isherm):
        cdef int row, i
        cdef int num_rows = self.shape0
        cdef int n = <int>libc.math.sqrt(num_rows)
        cdef complex dot = 0.0
        self._factor(t)
        self._call_core(self.data_t, self.coeff_ptr)

        for row from 0 <= row < num_rows by n+1:
          for i in range(self.shape1):
            dot += self.data_t[row,i]*vec[i]

        if isherm:
            return real(dot)
        else:
            return dot
