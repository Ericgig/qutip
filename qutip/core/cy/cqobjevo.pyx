#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdvision=True
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

from libc.math cimport sqrt, NAN
from cpython.exc cimport PyErr_CheckSignals

cimport numpy as cnp
cnp.import_array()

from ..qobj import Qobj
from .. import data as _data

from qutip.core.data cimport CSR, Dense, Data
from qutip.core.data.add cimport add_csr, iadd_dense
from qutip.core.data.matmul cimport (matmul_csr, matmul_csr_dense_dense,
                                     matmul_dense)
# TODO: handle dispatch properly.  We import rather than cimport because we
# have to call with Python semantics.
from qutip.core.data.expect cimport (
    expect_csr, expect_super_csr, expect_csr_dense, expect_super_csr_dense,
    expect_dense_dense, expect_super_dense_dense,
)
from qutip.core.data.reshape cimport (column_stack_csr, column_stack_dense,
                                      column_unstack_dense)
from qutip.core.cy.coefficient cimport Coefficient


cdef extern from "<complex>" namespace "std" nogil:
    double complex conj(double complex x)


# TODO: isintance in cython can be made python?
# Worth it?


cdef Data cy_matmul(Data left, Data right):
    cdef Data out
    if isinstance(right, CSR):
        if isinstance(left, CSR):
            out = matmul_csr(left, right)
        else:
            out = _data.matmul(left, right)
    elif isinstance(right, Dense):
        if isinstance(left, CSR):
            out = matmul_csr_dense_dense(left, right)
        elif isinstance(left, Dense):
            out = matmul_dense(left, right)
        else:
            out = _data.matmul(left, right)
    else:
        out = _data.matmul(left, right)
    return out


cdef Data cy_matmul_inplace(Data left, Data right,
                            double complex scale, Data out):
    if isinstance(right, CSR):
        if isinstance(left, CSR):
            out = matmul_csr(left, right, scale)
        else:
            out = _data.matmul(left, right, scale)
    elif isinstance(right, Dense) and isinstance(out, Dense):
        if isinstance(left, CSR):
            #print("matmul_csr_dense_dense")
            matmul_csr_dense_dense(left, right, scale, out)
        elif isinstance(left, Dense):
            #print("matmul_dense")
            matmul_dense(left, right, scale, out)
        else:
            #print("_data.matmul Dense")
            out = _data.matmul(left, right, scale)
    else:
        #print("_data.matmul")
        out = _data.matmul(left, right, scale)
    return out


cdef Data cy_add(Data left, Data right, double complex scale):
    cdef Data out
    if isinstance(right, CSR):
        if isinstance(left, CSR):
            out = add_csr(left, right, scale)
        else:
            out = _data.add(left, right, scale)
    elif isinstance(right, Dense):
        if isinstance(left, Dense):
            out = iadd_dense(left, right, scale)
        else:
            out = _data.add(left, right, scale)
    else:
        out = _data.add(left, right, scale)
    return out


cdef double complex cy_expect(Data left, Data right):
    cdef double complex out
    if isinstance(right, CSR):
        if isinstance(left, CSR):
            out = expect_csr(left, right)
        else:
            out = _data.expect(left, right)
    elif isinstance(right, Dense):
        if isinstance(left, CSR):
            out = expect_csr_dense(left, right)
        elif isinstance(left, Dense):
            out = expect_dense_dense(left, right)
        else:
            out = _data.expect(left, right)
    else:
        out = _data.expect(left, right)
    return out


cdef double complex cy_expect_super(Data left, Data right):
    cdef size_t nrow
    cdef double complex out
    if isinstance(right, CSR):
        right = column_stack_csr(right)
        if isinstance(left, CSR):
            out = expect_super_csr(left, right)
        else:
            out = _data.expect_super(left, right)
    elif isinstance(right, Dense):
        nrow = right.shape[0]
        right = column_stack_dense(right, inplace=True)
        if isinstance(left, CSR):
            out = expect_super_csr_dense(left, right)
        elif isinstance(left, Dense):
            out = expect_super_dense_dense(left, right)
        else:
            out = _data.expect_super(left, right)
        column_unstack_dense(right, nrow, inplace=True)
    else:
        out = _data.expect_super(left, right)
    return out


cdef class CQobjEvo:
    """
    Data matmul(double t, Data matrix, Data out=None)
      Get the matrix multiplication of self with matrix and put the result in
      `out`, if supplied.  Always returns the object it stored the output in
      (even if it was passed).

    expect(double t, Data matrix)
      Get the expectation value at a time `t`.
      return expectation value, knows to use the super version or not.

    call(double t, double complex [:] coefficients=None, bint data=False)
      Get the full representation of this operator at time `t`.  If the
      coefficients are given, they are used instead and the underlying
      coefficient-getting functions are not called.  If `data` is True, then
      the data-layer object is returned instead of a full Qobj.
    """
    def __init__(self, constant, ops=None):
        cdef size_t i
        if not isinstance(constant, Qobj):
            raise TypeError("inputs must be Qobj")
        self.shape = constant.shape
        self.dims = constant.dims
        self.type = constant.type
        self.issuper = constant.issuper
        self.constant = constant.data
        self.n_ops = 0 if ops is None else len(ops)
        self.ops = [None] * self.n_ops
        self.coefficients = cnp.PyArray_EMPTY(1, [self.n_ops],
                                              cnp.NPY_COMPLEX128, False)
        self.coeff = [None] * self.n_ops
        for i in range(self.n_ops):
            vary = ops[i]
            qobj = vary.qobj
            if (
                qobj.shape != self.shape
                or qobj.type != self.type
                or qobj.dims != self.dims
            ):
                raise ValueError("not all inputs have the same structure")
            self.ops[i] = qobj.data
            self.coeff[i] = vary.coeff

    def call(self, double t, object coefficients=None, bint data=False):
        cdef Data out = self.constant.copy()
        cdef Py_ssize_t i
        if coefficients is None:
            self._factor(t)
            coefficients = self.coefficients
        elif len(coefficients) != self.n_ops:
            raise TypeError(
                "got " + str(len(coefficients)) + " coefficients,"
                + " but expected " + str(self.n_ops)
            )
        for i in range(len(self.ops)):
            out = cy_add(out, <Data> self.ops[i], scale=coefficients[i])
        if data:
            return out
        return Qobj(out, dims=self.dims, type=self.type,
                    superrep=self.superrep, copy=False)

    cdef void _factor(self, double t) except *:
        cdef size_t i
        cdef Coefficient coeff
        for i in range(self.n_ops):
            coeff = <Coefficient> self.coeff[i]
            self.coefficients[i] = coeff._call(t)
        return

    cpdef Data matmul(self, double t, Data matrix, Data out=None):
        cdef size_t i
        self._factor(t)
        if out is None:
            out = cy_matmul(self.constant, matrix)
        else:
            out = cy_matmul_inplace(self.constant, matrix, 1, out)
        for i in range(self.n_ops):
            out = cy_matmul_inplace(<Data> self.ops[i], matrix,
                                    self.coefficients[i], out)
        return out

    cpdef double complex expect(self, double t, Data matrix) except *:
        """
        Expectation is defined as `matrix.adjoint() @ self @ matrix` if
        `matrix` is a vector, or `matrix` is an operator and `self` is a
        superoperator.  If `matrix` is an operator and `self` is an operator,
        then expectation is `trace(self @ matrix)`.
        """
        cdef size_t i
        cdef double complex out
        self._factor(t)
        if self.issuper:
            out = cy_expect_super(self.constant, matrix)
            for i in range(self.n_ops):
                out += self.coefficients[i] * cy_expect_super(<Data> self.ops[i], matrix)
        else:
            out = cy_expect(self.constant, matrix)
            for i in range(self.n_ops):
                out += self.coefficients[i] * cy_expect(<Data> self.ops[i], matrix)
        return out


cdef class CQobjFunc(CQobjEvo):
    cdef object base
    def __init__(self, base):
        self.base = base
        self.reset_shape()

    def reset_shape(self):
        self.shape = self.base.shape
        self.dims = self.base.dims
        self.issuper = self.base.issuper

    def call(self, double t, int data=0):
        return self.base(t, data=data)

    cpdef Data matmul(self, double t, Data matrix, Data out=None):
        # TODO: like cQobjEvo, only work for CSR x Dense...
        cdef Data objdata = self.base(t, data=True)
        if out is None:
            out = cy_matmul(objdata, matrix)
        else:
            out = cy_matmul_inplace(objdata, matrix, 1, out)
        return out

    cpdef double complex expect(self, double t, Data matrix) except *:
        """
        Expectation is defined as `matrix.adjoint() @ self @ matrix` if
        `matrix` is a vector, or `matrix` is an operator and `self` is a
        superoperator.  If `matrix` is an operator and `self` is an operator,
        then expectation is `trace(self @ matrix)`.
        """
        cdef double complex out
        cdef int nrow
        cdef Data objdata = self.base(t, data=True)
        if self.issuper:
            out = cy_expect_super(objdata, matrix)
        else:
            out = cy_expect(objdata, matrix)
        return out
