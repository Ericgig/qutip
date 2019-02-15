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

from qutip.matrix.cy.csr_matrix cimport cy_csr_matrix

cdef cy_csr_matrix _zcsr_add(cy_csr_matrix A, cy_csr_matrix B, double complex alpha)

cdef int _zcsr_add_core(double complex * Adata, int * Aind, int * Aptr,
                        double complex * Bdata, int * Bind, int * Bptr,
                        double complex alpha,
                        cy_csr_matrix C,
                        int nrows, int ncols) nogil

cdef cy_csr_matrix zcsr_mult(cy_csr_matrix A, cy_csr_matrix B, int sorted=*)

cpdef cy_csr_matrix zcsr_kron(cy_csr_matrix A, cy_csr_matrix B)

cdef void _zcsr_kron_core(double complex * dataA, int * indsA, int * indptrA,
                          double complex * dataB, int * indsB, int * indptrB,
                          CSR_Matrix * out,
                          int rowsA, int rowsB, int colsB) nogil

cpdef double complex zcsr_mat_elem(cy_csr_matrix A, cy_csr_matrix left,
                                   cy_csr_matrix right)
