#!python
#cython: language_level=3
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, The QuTiP Project.
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
cimport numpy as cnp

from qutip.core.data cimport CSR, Dense, Data

cdef void ZHEEVR(Dense H, double * eigvals, Dense Z, int nrows)

cdef void ZGEEV(complex[::1,:] H, double * eigvals,
                complex[::1,:] Z, int nrows)

cdef double complex * ZGEMM(double complex * A, double complex * B,
                            int Arows, int Acols, int Brows, int Bcols,
                            int transA=*, int transB=*, double complex alpha=*,
                            double complex beta=*, double complex * C=*)

cdef inline void vec2mat_index(int nrows, int index, int[2] out) nogil

cdef double skew_and_dwmin(double * evals, double[:,::1] skew,
                           unsigned int nrows) nogil

cdef Dense dense_to_eigbasis(Dense A, Dense evecs, double atol)

cpdef CSR liou_from_diag_ham(double[::1] diags)

cpdef CSR cop_super_term(Dense cop, Dense evecs,
                         double complex alpha, double atol)

cpdef Data _br_term_cross(Data A, Data B,
                          double[:, ::1] skew, double[:, ::1] spectrum,
                          bint use_secular, double cutoff)

cpdef CSR _br_term(Data A, double[:, ::1] skew, double[:, ::1] spectrum,
                   bint use_secular, double cutoff)
