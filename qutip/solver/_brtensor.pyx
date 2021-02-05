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
import warnings
import numpy as np
from qutip.settings import settings as qset
from qutip.core import Qobj, sprepost
from qutip import sprepost

import sys

from libc.math cimport fabs
from libcpp cimport bool
from libcpp.vector cimport vector

cimport numpy as np
cimport cython

from qutip.core.data cimport CSR, idxint, csr
from qutip.core.data.add cimport add_csr
from qutip.solve._brtools cimport (
    vec2mat_index, dense_to_eigbasis, ZHEEVR, skew_and_dwmin
)

__all__ = ['bloch_redfield_tensor']

np.import_array()

cdef extern from "numpy/arrayobject.h" nogil:
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
    void PyDataMem_FREE(void * ptr)
    void PyDataMem_RENEW(void * ptr, size_t size)
    void PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyDataMem_NEW(size_t size)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef complex[::1,:] dense_to_eigbasis(complex[::1,:] A, complex[::1,:] evecs,
                                    unsigned int nrows,
                                    double atol):
    cdef int kk
    cdef double complex * temp1 = ZGEMM(&A[0,0], &evecs[0,0],
                                       nrows, nrows, nrows, nrows, 0, 0)
    cdef double complex * eig_mat = ZGEMM(&evecs[0,0], temp1,
                                       nrows, nrows, nrows, nrows, 2, 0)
    PyDataMem_FREE(temp1)
    #Get view on ouput
    # Find all small elements and set to zero
    for kk in range(nrows**2):
        if cabs(eig_mat[kk]) < atol:
            eig_mat[kk] = 0
    cdef complex[:,::1] out = <complex[:nrows, :nrows]> eig_mat
    #This just gets the correct f-ordered view on the data
    cdef complex[::1,:] out_f = out.T
    return out_f


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef CSR liou_from_diag_ham(double[::1] diags):
    cdef unsigned int nrows = diags.shape[0]
    cdef CSR out = csr.empty(nrows*nrows, nrows*nrows, nrows*nrows)
    cdef size_t row, col, row_out, nnz=0
    cdef double complex val1, val2, ans

    out.row_index[0] = 0
    for row in range(nrows):
        val1 = 1j*diags[row]
        row_out = nrows*row
        for col in range(nrows):
            val2 = -1j*diags[col]
            ans = val1 + val2
            if ans != 0:
                out.data[nnz] = ans
                out.col_index[nnz] = row_out + col
                out.row_index[row_out + col + 1] = nnz+1
                nnz += 1
            else:
                out.row_index[row_out + col + 1] = nnz
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef CSR cop_super_term(complex[::1,:] cop, complex[::1,:] evecs,
                         double complex alpha, unsigned int nrows, double atol):
    cdef size_t kk
    cdef complex[::1,:] cop_eig = dense_to_eigbasis(cop, evecs, nrows, atol)
    cdef CSR c = csr.from_dense(dense.wrap(&cop_eig[0, 0], nrows, nrows, fortran=True))
    cdef size_t nnz = csr.nnz(c)
    # Multiply by alpha for time-dependence
    for kk in range(nnz):
        c.data[kk] *= alpha
    #Free data associated with cop_eig as it is no longer needed.
    PyDataMem_FREE(&cop_eig[0,0])
    cdef CSR cdc = matmul_csr(c.adjoint(), c)
    cdef CSR iden = csr.identity(nrows)
    cdef CSR out = add_csr(kron_csr(c.conj(), c), kron_csr(iden, cdc), scale=-0.5)
    return add_csr(out, kron_csr(cdc.transpose(), iden), scale=-0.5)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef CSR _br_term(complex[::1,:] A, complex[::1,:] evecs, double[:,::1] skew,
                   double dw_min, object spectral, unsigned int nrows, int
                   use_secular, double sec_cutoff, double atol):

    cdef size_t kk
    cdef size_t I, J # vector index variables
    cdef int[2] ab, cd #matrix indexing variables
    cdef complex[::1,:] A_eig = dense_to_eigbasis(A, evecs, nrows, atol)
    cdef complex elem, ac_elem, bd_elem
    cdef vector[idxint] coo_rows, coo_cols
    cdef vector[double complex] coo_data
    cdef unsigned int nnz

    for I in range(nrows**2):
        vec2mat_index(nrows, I, ab)
        for J in range(nrows**2):
            vec2mat_index(nrows, J, cd)

            if (not use_secular) or (fabs(skew[ab[0],ab[1]]-skew[cd[0],cd[1]]) < (dw_min * sec_cutoff)):
                elem = (A_eig[ab[0],cd[0]]*A_eig[cd[1],ab[1]]) * 0.5
                elem *= (spectral(skew[cd[0],ab[0]])+spectral(skew[cd[1],ab[1]]))

                if (ab[0]==cd[0]):
                    ac_elem = 0
                    for kk in range(nrows):
                        ac_elem += A_eig[cd[1],kk]*A_eig[kk,ab[1]] * spectral(skew[cd[1],kk])
                    elem -= 0.5*ac_elem

                if (ab[1]==cd[1]):
                    bd_elem = 0
                    for kk in range(nrows):
                        bd_elem += A_eig[ab[0],kk]*A_eig[kk,cd[0]] * spectral(skew[cd[0],kk])
                    elem -= 0.5*bd_elem

                if (elem != 0):
                    coo_rows.push_back(I)
                    coo_cols.push_back(J)
                    coo_data.push_back(elem)

    PyDataMem_FREE(&A_eig[0,0])

    return csr.from_coo_pointers(
        coo_rows.data(), coo_cols.data(), coo_data.data(),
        nrows*nrows, nrows*nrows, coo_rows.size())


@cython.boundscheck(False)
@cython.wraparound(False)
def bloch_redfield_tensor(object H, list a_ops,
                 list c_ops=[], bool use_secular=True,
                 double sec_cutoff=0.1,
                 double atol = qset.core['atol'],
                 bint eigenket=True, str basis='eigen'):
    """
    Calculates the time-independent Bloch-Redfield tensor for a system given
    a set of operators and corresponding spectral functions that describes the
    system's couplingto its environment.

    Parameters
    ----------

    H : :class:`qutip.qobj`
        System Hamiltonian.

    a_ops : list
        Nested list of system operators that couple to the environment,
        and the corresponding bath spectra represented as Python
        functions.

    spectra_cb : list
        Depreciated.

    c_ops : list
        List of system collapse operators.

    use_secular : bool {True, False}
        Flag that indicates if the secular approximation should
        be used.

    sec_cutoff : float {0.1}
        Threshold for secular approximation.

    tol : float {qutip.settings.core['atol']}
       Threshold for removing small parameters.

    Returns
    -------

    R, kets: :class:`qutip.Qobj`, list of :class:`qutip.Qobj`

        R is the Bloch-Redfield tensor and kets is a list eigenstates of the
        Hamiltonian.

    """
    cdef list _a_ops
    cdef object a, cop
    cdef CSR L
    cdef int kk
    cdef int nrows = H.shape[0]
    cdef list op_dims = H.dims
    cdef list ket_dims = [op_dims[0], [1] * len(op_dims[0])]
    cdef list sop_dims = [[op_dims[0], op_dims[0]], [op_dims[1], op_dims[1]]]
    cdef list ekets
    cdef double dw_min
    cdef double[:,::1] skew = np.zeros((nrows,nrows), dtype=float)
    cdef object R

    # Sanity checks for input parameters
    if not isinstance(H, Qobj):
        raise TypeError("H must be an instance of Qobj")

    for a in a_ops:
        if not isinstance(a[0], Qobj) or not a[0].isherm:
            raise TypeError("Operators in a_ops must be Hermitian Qobj.")

    cdef complex[::1,:] H0 = H.full('F')
    cdef complex[::1,:] evecs = np.zeros((nrows,nrows), dtype=complex, order='F')
    cdef double[::1] evals = np.zeros(nrows, dtype=float)

    ZHEEVR(H0, &evals[0], evecs, nrows)
    L = liou_from_diag_ham(evals)
    ekets = [Qobj(np.asarray(evecs[:,k]), dims=ket_dims) for k in range(nrows)]

    for cop in c_ops:
        L = add_csr(L, cop_super_term(cop.full('F'), evecs, 1, nrows, atol))

    if not len(a_ops) == 0:
        #has some br operators and spectra
        dw_min = skew_and_dwmin(&evals[0], skew, nrows)
        for a in a_ops:
            L = add_csr(L, _br_term(a[0].full('F'), evecs, skew, dw_min, a[1],
                        nrows, use_secular, sec_cutoff, atol))

    R = Qobj(L, dims=sop_dims, type='super', copy=False)
    if not basis == 'eigen':
        base = np.hstack([psi.full() for psi in eket])
        S = Qobj(_data.adjoint(_data.create(base)), dims=R.dims)
        R = sprepost(S.dag(), S) @ R @ sprepost(S, S.dag())

    if eigenket:
        return R, ekets
    return R
