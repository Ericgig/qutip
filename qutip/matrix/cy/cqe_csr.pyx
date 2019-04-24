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
import cython
cimport cython
import numpy as np
cimport numpy as np
import scipy.sparse as sp
from qutip.matrix.cy.cqobjevo cimport CQobjEvo
from qutip.matrix.cy.csr_matrix cimport cy_csr_matrix, spmvpy, _spmm_c_py, _spmm_f_py
from qutip.matrix.cy.csr_math cimport _zcsr_add_core, zcsr_add
from qutip.cy.cqobjevo_factor cimport CoeffFunc
from qutip.qobj import Qobj
cimport libc.math

from qutip.cy.complex_math cimport *
np.import_array()

cdef extern from "Python.h":
    object PyLong_FromVoidPtr(void *)
    void* PyLong_AsVoidPtr(object)

cdef extern from "numpy/arrayobject.h" nogil:
    void PyDataMem_FREE(void * ptr)
    void PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyDataMem_NEW(size_t size)

cdef _csr_mat* set_csr_mat(cy_csr_matrix mat):
    cdef _csr_mat* out = <_csr_mat*> PyDataMem_NEW(sizeof(_csr_mat))
    out.data = mat.data
    out.indices = mat.indices
    out.indptr = mat.indptr
    return out

cdef _csr_mat_get_state(_csr_mat* mat):
    return (PyLong_FromVoidPtr(mat.data),
            PyLong_FromVoidPtr(mat.indices),
            PyLong_FromVoidPtr(mat.indptr))

cdef _csr_mat* _csr_mat_set_state(state):
    cdef _csr_mat* out = <_csr_mat*> PyDataMem_NEW(sizeof(_csr_mat))
    out.data = <complex*>PyLong_AsVoidPtr(state[0])
    out.indices = <int*>PyLong_AsVoidPtr(state[1])
    out.indptr = <int*>PyLong_AsVoidPtr(state[2])
    return out

cdef class CQobjCte(CQobjEvo):
    def set_data(self, cte):
        self.shape0 = cte.shape[0]
        self.shape1 = cte.shape[1]
        self.dims = cte.dims
        self.cte = cte.data.cdata
        self.total_elem = cte.data.data.shape[0]
        self.super = cte.issuper

    def __getstate__(self):
        csr_info = self.cte._shallow_get_state()
        return (self.shape0, self.shape1, self.dims,
                self.total_elem, self.super, csr_info)

    def __setstate__(self, state):
        self.shape0 = state[0]
        self.shape1 = state[1]
        self.dims = state[2]
        self.total_elem = state[3]
        self.super = state[4]
        self.cte._shallow_set_state(state[5])

    def call(self, double t, int data=0):
        cdef cy_csr_matrix out = self.cte.copy()
        scipy_obj = out.to_qdata()
        if data:
            return scipy_obj
        else:
            return Qobj(scipy_obj, dims=self.dims)

    def call_with_coeff(self, complex[::1] coeff, int data=0):
        cdef cy_csr_matrix out = self.cte.copy()
        scipy_obj = out.to_qdata()
        if data:
            return scipy_obj
        else:
            return Qobj(scipy_obj)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_vec(self, double t, complex* vec, complex* out):
        spmvpy(self.cte.data, self.cte.indices, self.cte.indptr, vec, 1.,
               out, self.shape0)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_matf(self, double t, complex* mat, complex* out,
                        int nrow, int ncol):
        _spmm_f_py(self.cte.data, self.cte.indices, self.cte.indptr, mat, 1.,
               out, self.shape0, nrow, ncol)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_matc(self, double t, complex* mat, complex* out,
                        int nrow, int ncol):
        _spmm_c_py(self.cte.data, self.cte.indices, self.cte.indptr, mat, 1.,
               out, self.shape0, nrow, ncol)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect(self, double t, complex* vec):
        cdef complex[::1] y = np.zeros(self.shape0, dtype=complex)
        spmvpy(self.cte.data, self.cte.indices, self.cte.indptr, vec, 1.,
               &y[0], self.shape0)
        cdef int row
        cdef complex dot = 0
        for row from 0 <= row < self.shape0:
            dot += conj(vec[row])*y[row]
        return dot

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect_super(self, double t, complex* vec):
        cdef int row
        cdef int jj, row_start, row_end
        cdef int num_rows = self.shape0
        cdef int n = <int>libc.math.sqrt(num_rows)
        cdef complex dot = 0.0

        for row from 0 <= row < num_rows by n+1:
            row_start = self.cte.indptr[row]
            row_end = self.cte.indptr[row+1]
            for jj from row_start <= jj < row_end:
                dot += self.cte.data[jj]*vec[self.cte.indices[jj]]

        return dot

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _overlapse(self, double t, complex* oper):
        """tr( self * oper )"""
        cdef int row
        cdef int jj, row_start, row_end
        cdef int num_rows = self.shape0
        cdef complex tr = 0.0

        for row in range(num_rows):
            row_start = self.cte.indptr[row]
            row_end = self.cte.indptr[row+1]
            for jj from row_start <= jj < row_end:
                tr += self.cte.data[jj]*oper[num_rows*jj + row]
        return tr


cdef class CQobjEvoTd(CQobjEvo):
    def __init__(self):
        self.num_ops = 0
        self.ops = <_csr_mat**> PyDataMem_NEW(0 * sizeof(_csr_mat*))

    def __del__(self):
        for i in range(self.num_ops):
            PyDataMem_FREE(self.ops[i])
        PyDataMem_FREE(self.ops)

    def set_data(self, cte, ops):
        cdef int i
        self.shape0 = cte.shape[0]
        self.shape1 = cte.shape[1]
        self.dims = cte.dims
        self.cte = cte.data.cdata
        cummulative_op = cte.data
        self.super = cte.issuper

        self.num_ops = len(ops)
        self.coeff = np.empty((self.num_ops,), dtype=complex)
        self.coeff_ptr = &self.coeff[0]
        PyDataMem_FREE(self.ops)
        self.ops = <_csr_mat**> PyDataMem_NEW(self.num_ops * sizeof(_csr_mat*))
        self.sum_elem = np.zeros(self.num_ops, dtype=int)
        for i, op in enumerate(ops):
            self.ops[i] = set_csr_mat(op[0].data.cdata)
            cummulative_op += op[0].data
            self.sum_elem[i] = cummulative_op.data.shape[0]

        self.total_elem = self.sum_elem[self.num_ops-1]

    def set_factor(self, func=None, ptr=False, obj=None):
        self.factor_use_cobj = 0
        if func is not None:
            self.factor_func = func
        elif obj is not None:
            self.factor_use_cobj = 1
            self.factor_cobj = obj
        else:
            raise Exception("Could not set coefficient function")

    def __getstate__(self):
        cte_info = self.cte._shallow_get_state()
        ops_info = ()
        sum_elem = ()
        for i in range(self.num_ops):
            ops_info += (_csr_mat_get_state(self.ops[i]),)
            sum_elem += (self.sum_elem[i],)

        return (self.shape0, self.shape1, self.dims, self.total_elem, self.super,
                self.factor_use_cobj, self.factor_cobj, self.factor_func,
                self.num_ops, sum_elem, cte_info, ops_info)

    def __setstate__(self, state):
        self.shape0 = state[0]
        self.shape1 = state[1]
        self.dims = state[2]
        self.total_elem = state[3]
        self.super = state[4]
        self.factor_use_cobj = state[5]
        if self.factor_use_cobj:
            self.factor_cobj = <CoeffFunc> state[6]
        self.factor_func = state[7]
        self.num_ops = state[8]
        self.cte._shallow_set_state(state[10])
        self.sum_elem = np.zeros(self.num_ops, dtype=int)
        self.ops = <_csr_mat**> PyDataMem_NEW(self.num_ops * sizeof(_csr_mat*))
        for i in range(self.num_ops):
            self.sum_elem[i] = state[9][i]
            self.ops[i] = _csr_mat_set_state(state[11][i])
        self.coeff = np.empty((self.num_ops,), dtype=complex)
        self.coeff_ptr = &self.coeff[0]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _call_core(self, cy_csr_matrix out, complex* coeff):
        cdef int i
        cdef _csr_mat* previous
        cdef _csr_mat* next

        if(self.num_ops ==1):
            _zcsr_add_core(self.cte.data, self.cte.indices, self.cte.indptr,
                           self.ops[0].data, self.ops[0].indices,
                           self.ops[0].indptr,
                           coeff[0],
                           out.data, out.indices, out.indptr,
                           self.shape0, self.shape1)
        else:
            # Ugly with a loop for 1 to N-2...
            # It save the copy of data from cte and out
            # no init/free to cte, out
            previous = <_csr_mat *>PyDataMem_NEW(sizeof(_csr_mat))
            next = <_csr_mat *>PyDataMem_NEW(sizeof(_csr_mat))
            previous.indptr = <int *>PyDataMem_NEW((self.shape0+1) * sizeof(int))
            previous.indices = <int *>PyDataMem_NEW(self.sum_elem[self.num_ops-1] * sizeof(int))
            previous.data = <double complex *>PyDataMem_NEW(self.sum_elem[self.num_ops-1] * sizeof(double complex))
            next.indptr = <int *>PyDataMem_NEW((self.shape0+1) * sizeof(int))
            next.indices = <int *>PyDataMem_NEW(self.sum_elem[self.num_ops-1] * sizeof(int))
            next.data = <double complex *>PyDataMem_NEW(self.sum_elem[self.num_ops-1] * sizeof(double complex))

            _zcsr_add_core(self.cte.data, self.cte.indices, self.cte.indptr,
                           self.ops[0].data,
                           self.ops[0].indices,
                           self.ops[0].indptr,
                           coeff[0],
                           next.data, next.indices, next.indptr,
                           self.shape0, self.shape1)
            previous, next = next, previous

            for i in range(1, self.num_ops-1):
                #init_CSR(&next, self.sum_elem[i], self.shape0, self.shape1)
                _zcsr_add_core(previous.data, previous.indices,
                               previous.indptr,
                               self.ops[i].data,
                               self.ops[i].indices,
                               self.ops[i].indptr,
                               coeff[i],
                               next.data, next.indices, next.indptr,
                               self.shape0, self.shape1)
                #free_CSR(&previous)
                previous, next = next, previous

            _zcsr_add_core(previous.data, previous.indices, previous.indptr,
                           self.ops[self.num_ops-1].data,
                           self.ops[self.num_ops-1].indices,
                           self.ops[self.num_ops-1].indptr,
                           coeff[self.num_ops-1],
                           out.data, out.indices, out.indptr,
                           self.shape0, self.shape1)
            PyDataMem_FREE(previous.indptr)
            PyDataMem_FREE(previous.indices)
            PyDataMem_FREE(previous.data)
            PyDataMem_FREE(next.indptr)
            PyDataMem_FREE(next.indices)
            PyDataMem_FREE(next.data)
            PyDataMem_FREE(previous)
            PyDataMem_FREE(next)

    def call(self, double t, int data=0):
        cdef cy_csr_matrix out = cy_csr_matrix()
        out.init_CSR(self.total_elem, self.shape0, self.shape1, self.shape0)
        self._factor(t)
        self._call_core(out, self.coeff_ptr)
        scipy_obj = out.to_qdata()
        # free_CSR(&out)? data is own by the scipy_obj?
        if data:
            return scipy_obj
        else:
            return Qobj(scipy_obj, dims=self.dims)

    def call_with_coeff(self, complex[::1] coeff, int data=0):
        cdef cy_csr_matrix out = cy_csr_matrix()
        out.init_CSR(self.total_elem, self.shape0, self.shape1, self.shape0)
        self._call_core(out, &coeff[0])
        scipy_obj = out.to_qdata()
        # free_CSR(&out)? data is own by the scipy_obj?
        if data:
            return scipy_obj
        else:
            return Qobj(scipy_obj)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_vec(self, double t, complex* vec, complex* out):
        cdef int[2] shape
        shape[0] = self.shape1
        shape[1] = 1
        self._factor_dyn(t, vec, shape)
        cdef int i
        spmvpy(self.cte.data, self.cte.indices, self.cte.indptr, vec,
               1., out, self.shape0)
        for i in range(self.num_ops):
            spmvpy(self.ops[i].data, self.ops[i].indices, self.ops[i].indptr,
                   vec, self.coeff_ptr[i], out, self.shape0)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_matf(self, double t, complex* mat, complex* out,
                        int nrow, int ncol):
        cdef int[2] shape
        shape[0] = nrow
        shape[1] = ncol
        self._factor_dyn(t, mat, shape)
        cdef int i
        _spmm_f_py(self.cte.data, self.cte.indices, self.cte.indptr, mat, 1.,
               out, self.shape0, nrow, ncol)
        for i in range(self.num_ops):
             _spmm_f_py(self.ops[i].data, self.ops[i].indices, self.ops[i].indptr,
                 mat, self.coeff_ptr[i], out, self.shape0, nrow, ncol)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_matc(self, double t, complex* mat, complex* out,
                        int nrow, int ncol):
        cdef int[2] shape
        shape[0] = nrow
        shape[1] = ncol
        self._factor_dyn(t, mat, shape)
        cdef int i
        _spmm_c_py(self.cte.data, self.cte.indices, self.cte.indptr, mat, 1.,
               out, self.shape0, nrow, ncol)
        for i in range(self.num_ops):
             _spmm_c_py(self.ops[i].data, self.ops[i].indices, self.ops[i].indptr,
                 mat, self.coeff_ptr[i], out, self.shape0, nrow, ncol)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect(self, double t, complex* vec):
        cdef complex [::1] y = np.zeros(self.shape0, dtype=complex)
        cdef int row
        cdef complex dot = 0
        self._mul_vec(t, &vec[0], &y[0])
        for row from 0 <= row < self.shape0:
            dot += conj(vec[row]) * y[row]
        return dot

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect_super(self, double t, complex* vec):
        cdef int[2] shape
        cdef int row, i
        cdef int jj, row_start, row_end
        cdef int num_rows = self.shape0
        cdef int n = <int>libc.math.sqrt(num_rows)
        cdef complex dot = 0.0
        shape[0] = n
        shape[1] = n
        self._factor_dyn(t, vec, shape)

        for row from 0 <= row < num_rows by n+1:
            row_start = self.cte.indptr[row]
            row_end = self.cte.indptr[row+1]
            for jj from row_start <= jj < row_end:
                dot += self.cte.data[jj]*vec[self.cte.indices[jj]]
        for i in range(self.num_ops):
            for row from 0 <= row < num_rows by n+1:
                row_start = self.ops[i].indptr[row]
                row_end = self.ops[i].indptr[row+1]
                for jj from row_start <= jj < row_end:
                    dot += self.ops[i].data[jj] * \
                          vec[self.ops[i].indices[jj]] * self.coeff_ptr[i]
        return dot

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _overlapse(self, double t, complex* oper):
        """tr( self * oper )"""
        cdef int jj, row_start, row_end, row
        cdef int num_rows = self.shape0
        cdef complex tr = 0.0
        cdef int[2] shape
        shape[0] = self.shape0
        shape[1] = self.shape0
        self._factor_dyn(t, oper, shape)

        for row in range(num_rows):
            row_start = self.cte.indptr[row]
            row_end = self.cte.indptr[row+1]
            for jj from row_start <= jj < row_end:
                tr += self.cte.data[jj] * oper[num_rows*jj + row]

        for i in range(self.num_ops):
            for row in range(num_rows):
                row_start = self.ops[i].indptr[row]
                row_end = self.ops[i].indptr[row+1]
                for jj from row_start <= jj < row_end:
                    tr += self.ops[i].data[jj] * oper[num_rows*jj + row] * self.coeff_ptr[i]

        return tr

""" Temporary disabled
def _zcsr_match(sparses_list):
    \"""
    For a list of csr sparse matrice A,
    set them so the their indptr and indices be all equal.
    Require keeping 0s in the data, but summation can be done in vector form.
    \"""
    full_shape = sparses_list[0].copy()
    for sparse_elem in sparses_list[1:]:
        full_shape.data *= 0.
        full_shape.data += 1.
        if sparse_elem.indptr[-1] != 0:
            full_shape = zcsr_add(
                      full_shape.data, full_shape.indices, full_shape.indptr,
                      sparse_elem.data, sparse_elem.indices, sparse_elem.indptr,
                      full_shape.shape[0], full_shape.shape[1],
                      full_shape.indptr[-1], sparse_elem.indptr[-1], 0.)
    out = []
    for sparse_elem in sparses_list[:]:
        full_shape.data *= 0.
        if sparse_elem.indptr[-1] != 0:
            out.append(zcsr_add(
                      full_shape.data, full_shape.indices, full_shape.indptr,
                      sparse_elem.data, sparse_elem.indices, sparse_elem.indptr,
                      full_shape.shape[0], full_shape.shape[1],
                      full_shape.indptr[-1], sparse_elem.indptr[-1], 1.))
        else:
            out.append(full_shape.copy())
    return out

cdef class CQobjEvoTdMatched(CQobjEvo):
    def set_data(self, cte, ops):
        cdef int i, j
        self.shape0 = cte.shape[0]
        self.shape1 = cte.shape[1]
        self.dims = cte.dims
        self.super = cte.issuper
        self.num_ops = len(ops)
        self.coeff = np.zeros((self.num_ops), dtype=complex)
        self.coeff_ptr = &self.coeff[0]

        sparse_list = []
        for op in ops:
            sparse_list.append(op[0].data)
        sparse_list += [cte.data]
        matched = _zcsr_match(sparse_list)

        self.indptr = matched[0].indptr
        self.indices = matched[0].indices
        self.cte = matched[-1].data
        self.nnz = len(self.cte)
        self.data_t = np.zeros((self.nnz), dtype=complex)
        self.data_ptr = &self.data_t[0]

        self.ops = np.zeros((self.num_ops, self.nnz), dtype=complex)
        for i, op in enumerate(matched[:-1]):
          for j in range(self.nnz):
            self.ops[i,j] = op.data[j]

    def set_factor(self, func=None, ptr=False, obj=None):
        self.factor_use_cobj = 0
        if func is not None:
            self.factor_func = func
        elif obj is not None:
            self.factor_use_cobj = 1
            self.factor_cobj = obj
        else:
            raise Exception("Could not set coefficient function")

    def __getstate__(self):
        return (self.shape0, self.shape1, self.dims, self.nnz, self.super,
                self.factor_use_cobj,
                self.factor_cobj, self.factor_func, self.num_ops,
                np.array(self.indptr), np.array(self.indices),
                np.array(self.cte), np.array(self.ops))

    def __setstate__(self, state):
        self.shape0 = state[0]
        self.shape1 = state[1]
        self.dims = state[2]
        self.nnz = state[3]
        self.super = state[4]
        self.factor_use_cobj = state[5]
        if self.factor_use_cobj:
            self.factor_cobj = <CoeffFunc> state[6]
        self.factor_func = state[7]
        self.num_ops = state[8]
        self.indptr = state[9]
        self.indices = state[10]
        self.cte = state[11]
        self.ops = state[12]
        self.coeff = np.zeros((self.num_ops), dtype=complex)
        self.coeff_ptr = &self.coeff[0]
        self.data_t = np.zeros((self.nnz), dtype=complex)
        self.data_ptr = &self.data_t[0]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _call_core(self, complex[::1] out, complex* coeff):
        cdef int i, j
        cdef complex * ptr
        ptr = &self.cte[0]
        for j in range(self.nnz):
            out[j] = ptr[j]
        for i in range(self.num_ops):
            ptr = &self.ops[i,0]
            for j in range(self.nnz):
                out[j] += ptr[j] * coeff[i]

    def call(self, double t, int data=0):
        cdef int i
        cdef complex[::1] data_t = np.empty(self.nnz, dtype=complex)
        self._factor(t)
        self._call_core(data_t, self.coeff_ptr)

        cdef CSR_Matrix out_csr
        init_CSR(&out_csr, self.nnz, self.shape0, self.shape1)
        for i in range(self.nnz):
            out_csr.data[i] = data_t[i]
            out_csr.indices[i] = self.indices[i]
        for i in range(self.shape0+1):
            out_csr.indptr[i] = self.indptr[i]
        scipy_obj = CSR_to_scipy(&out_csr)
        # free_CSR(&out)? data is own by the scipy_obj?
        if data:
            return scipy_obj
        else:
            return Qobj(scipy_obj, dims=self.dims)

    def call_with_coeff(self, complex[::1] coeff, int data=0):
        cdef complex[::1] out = np.empty(self.nnz, dtype=complex)
        self._call_core(out, &coeff[0])
        cdef CSR_Matrix out_csr
        init_CSR(&out_csr, self.nnz, self.shape0, self.shape1)
        for i in range(self.nnz):
            out_csr.data[i] = out[i]
            out_csr.indices[i] = self.indices[i]
        for i in range(self.shape0+1):
            out_csr.indptr[i] = self.indptr[i]
        scipy_obj = CSR_to_scipy(&out_csr)
        # free_CSR(&out)? data is own by the scipy_obj?
        if data:
            return scipy_obj
        else:
            return Qobj(scipy_obj, dims=self.dims)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_vec(self, double t, complex* vec, complex* out):
        cdef int[2] shape
        shape[0] = self.shape1
        shape[1] = 1
        self._factor_dyn(t, vec, shape)
        self._call_core(self.data_t, self.coeff_ptr)
        spmvpy(self.data_ptr, &self.indices[0], &self.indptr[0], vec,
               1., out, self.shape0)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_matf(self, double t, complex* mat, complex* out,
                        int nrow, int ncol):
        cdef int[2] shape
        shape[0] = nrow
        shape[1] = ncol
        self._factor_dyn(t, mat, shape)
        self._call_core(self.data_t, self.coeff_ptr)
        _spmm_f_py(self.data_ptr, &self.indices[0], &self.indptr[0], mat, 1.,
               out, self.shape0, nrow, ncol)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _mul_matc(self, double t, complex* mat, complex* out,
                        int nrow, int ncol):
        cdef int[2] shape
        shape[0] = nrow
        shape[1] = ncol
        self._factor_dyn(t, mat, shape)
        self._call_core(self.data_t, self.coeff_ptr)
        _spmm_c_py(self.data_ptr, &self.indices[0], &self.indptr[0], mat, 1.,
               out, self.shape0, nrow, ncol)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect(self, double t, complex* vec):
        cdef complex [::1] y = np.zeros(self.shape0, dtype=complex)
        cdef int row
        cdef complex dot = 0
        self._mul_vec(t, &vec[0], &y[0])
        for row from 0 <= row < self.shape0:
            dot += conj(vec[row]) * y[row]
        return dot

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef complex _expect_super(self, double t, complex* vec):
        cdef int row
        cdef int jj, row_start, row_end
        cdef int num_rows = self.shape0
        cdef int n = <int>libc.math.sqrt(num_rows)
        cdef complex dot = 0.0
        cdef int[2] shape
        shape[0] = n
        shape[1] = n
        self._factor_dyn(t, vec, shape)
        self._call_core(self.data_t, self.coeff_ptr)

        for row from 0 <= row < num_rows by n+1:
            row_start = self.indptr[row]
            row_end = self.indptr[row+1]
            for jj from row_start <= jj < row_end:
                dot += self.data_ptr[jj]*vec[self.indices[jj]]
        if isherm:
            return real(dot)
        else:
            return dot
"""
