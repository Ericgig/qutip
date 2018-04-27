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
cimport cython
import numpy as np
cimport numpy as cnp

cdef object sp_type = np.int64
#ctypedef np.int64_t sp_int
#ctypedef np.uint64_t sp_uint

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void open_scipy_for_CSR(object A, sp_int[::1] ind, sp_int[::1] ptr):
    """
    Extract the integers elements of the CSR sparce and change the type
    """
    cdef sp_int i
    for i in range(A.shape[0]+1):
        ind[i] = <sp_int>A.indices[i]

    for i in range(A.nnz):
        ptr[i] = <sp_int>A.indptr[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void open_scipy_for_COO(object A, sp_int[::1] rows, sp_int[::1] cols):
    """
    Extract the integers elements of the COO sparce and change the type
    """
    cdef sp_int i
    for i in range(A.nnz):
        rows[i] = <sp_int>A.row[i]
        cols[i] = <sp_int>A.col[i]
