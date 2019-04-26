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
cimport numpy as np
import numpy as np
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

cdef class CQobjEvo:
    """
    Interface for the CQobjEvo's variantes
    Python Methods
    --------------
    mul_vec(double t, complex[::1] vec)
      return self @ vec

    mul_mat(double t, np.ndarray[complex, ndim=2] mat)
      return self @ mat
      mat can be both "C" or "F" continuous.

    expect(double t, complex[::1] vec, int isherm)
      return expectation value, knows to use the super version or not.

    ode_mul_mat_f_vec(double t, complex[::1] mat)
      return self @ mat
      mat is in a 1d, F ordered form.
      Used with scipy solver which only accept vector.

    call(double t, int data=0)
      return this at time t

    call_with_coeff(complex[::1] coeff, int data=0)
      return this with the given coefficients

    set_data(cte, [ops])
      build the object from data from QobjEvo

    set_factor(self, func=None, ptr=False, obj=None)
      get the coefficient function from QobjEvo

    Cython Methods
    --------------
    _mul_vec(double t, complex* vec, complex* out):
        out += self * vec

    _mul_matf(double t, complex* mat, complex* out, int nrow, int ncols):
        out += self * dense mat fortran ordered

    _mul_matc(double t, complex* mat, complex* out, int nrow, int ncols):
        out += self * dense mat c ordered

    _expect(double t, complex* vec, int isherm):
        return <vec| self |vec>

    _expect_super(double t, complex* rho, int isherm):
        return tr( self * rho )
    """

    cdef void _mul_vec(self, double t, complex* vec, complex* out):
        """self * vec"""
        pass

    cdef void _mul_matf(self, double t, complex* mat, complex* out,
                    int nrow, int ncols):
        """self * dense mat fortran ordered """
        pass

    cdef void _mul_matc(self, double t, complex* mat, complex* out,
                    int nrow, int ncols):
        """self * dense mat c ordered"""
        pass

    cdef complex _expect(self, double t, complex* vec):
        """<vec| self |vec>"""
        return 0.

    cdef complex _expect_super(self, double t, complex* rho):
        """tr( self * rho )"""
        return 0.

    cdef complex _overlapse(self, double t, complex* oper):
        """tr( self * oper )"""
        return 0.

    def set_factor(self, func=None, ptr=False, obj=None):
        self.factor_use_cobj = 0
        if func is not None:
            self.factor_func = func
        elif obj is not None:
            self.factor_use_cobj = 1
            self.factor_cobj = obj
        else:
            raise Exception("Could not set coefficient function")

    cdef void _factor(self, double t):
        cdef int i
        if self.factor_use_cobj:
            self.factor_cobj._call_core(t, self.coeff_ptr)
        else:
            coeff = self.factor_func(t)
            for i in range(self.num_ops):
                self.coeff_ptr[i] = coeff[i]

    cdef void _factor_dyn(self, double t, complex* state, int[::1] shape):
        cdef int len_
        if self.dyn_args:
            if self.factor_use_cobj:
                # print("factor_use_cobj")
                self.factor_cobj._dyn_args(t, state, shape)
            else:
                len_ = shape[0] * shape[1]
                # print(len_, shape.shape[0])
                self.factor_func.dyn_args(t, np.array(<complex[:len_]> state),
                                          np.array(shape))
        self._factor(t)

    def mul_vec(self, double t, complex[::1] vec):
        cdef np.ndarray[complex, ndim=1] out = np.zeros(self.shape0,
                                                        dtype=complex)
        self._mul_vec(t, &vec[0], &out[0])
        return out

    def mul_mat(self, double t, np.ndarray[complex, ndim=2] mat):
        cdef np.ndarray[complex, ndim=2] out
        cdef unsigned int sp_rows = self.shape0
        cdef unsigned int nrows = mat.shape[0]
        cdef unsigned int ncols = mat.shape[1]
        if mat.flags["F_CONTIGUOUS"]:
            out = np.zeros((sp_rows,ncols), dtype=complex, order="F")
            self._mul_matf(t, &mat[0,0], &out[0,0], nrows, ncols)
        else:
            out = np.zeros((sp_rows,ncols), dtype=complex)
            self._mul_matc(t, &mat[0,0], &out[0,0], nrows, ncols)
        return out

    cpdef complex expect(self, double t, complex[::1] vec):
        if self.super:
            return self._expect_super(t, &vec[0])
        else:
            return self._expect(t, &vec[0])

    def ode_mul_mat_f_vec(self, double t, complex[::1] mat):
        cdef np.ndarray[complex, ndim=1] out = np.zeros(self.shape1*self.shape1,
                                                      dtype=complex)
        self._mul_matf(t, &mat[0], &out[0], self.shape1, self.shape1)
        return out

    def call(self, double t, int data=0):
        return None

    def call_with_coeff(self, complex[::1] coeff, int data=0):
        return None

    def has_dyn_args(self, int dyn_args):
        self.dyn_args = dyn_args

    def set_data(self, cte):
        pass

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        pass
