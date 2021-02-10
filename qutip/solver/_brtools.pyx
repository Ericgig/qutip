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
from libc.math cimport fabs, fmin
from libc.float cimport DBL_MAX
from libcpp.vector cimport vector
from libcpp cimport bool

cimport numpy as cnp
import numpy as np

from scipy.linalg.cython_lapack cimport zheevr, zgeev
from scipy.linalg.cython_blas cimport zgemm, zgemv, zaxpy
from scipy.linalg cimport cython_blas as blas

cimport cython

from qutip.core.cy.cqobjevo cimport CQobjFunc, CQobjEvo
from qutip.core.data cimport Data, CSR, Dense, dense, csr, idxint
from qutip.core.data.add cimport add_csr
from qutip.core.data.add import add
from qutip.core.data.kron cimport kron_csr
from qutip.core.data.tidyup cimport tidyup_dense
from qutip.core.data.matmul cimport matmul_csr, matmul_csr_dense_dense
from qutip.core.data.matmul import matmul
from qutip.core.data.mul cimport imul_csr, imul_dense, mul_dense
from qutip.core.data.reshape cimport column_stack_dense, column_unstack_dense
from qutip.core.data.convert import to
from qutip import settings
from qutip.settings import settings as qset
from qutip.core import Qobj, sprepost

import warnings
import sys

eigh_unsafe = settings.install["eigh_unsafe"]
cnp.import_array()

cdef extern from "numpy/arrayobject.h" nogil:
    void PyArray_ENABLEFLAGS(cnp.ndarray arr, int flags)
    void PyDataMem_FREE(void * ptr)
    void PyDataMem_RENEW(void * ptr, size_t size)
    void PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyDataMem_NEW(size_t size)

cdef extern from "<complex>" namespace "std" nogil:
    double complex conj(double complex x)
    double         real(double complex x)
    double cabs   "abs" (double complex x)
    double complex sqrt(double complex x)

cdef int use_zgeev = eigh_unsafe


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void ZHEEVR(Dense H, double * eigvals, Dense Z, int nrows):
    """
    Computes the eigenvalues and vectors of a dense Hermitian matrix.
    Eigenvectors are returned in Z.

    Parameters
    ----------
    H : array_like
        Input Hermitian matrix.
    eigvals : array_like
        Input array to store eigen values.
    Z : array_like
        Output array of eigenvectors.
    nrows : int
        Number of rows in matrix.
    """
    if use_zgeev:
        ZGEEV(H.as_ndarray(), eigvals, Z.as_ndarray(), nrows)
        return
    cdef char jobz = b'V'
    cdef char rnge = b'A'
    cdef char uplo = b'L'
    cdef double vl=1, vu=1, abstol=0
    cdef int il=1, iu=1
    cdef int lwork = 18 * nrows
    cdef int lrwork = 24*nrows, liwork = 10*nrows
    cdef int info=0, M=0
    #These nee to be freed at end
    cdef int * isuppz = <int *>PyDataMem_NEW((2*nrows) * sizeof(int))
    cdef complex * work = <complex *>PyDataMem_NEW(lwork * sizeof(complex))
    cdef double * rwork = <double *>PyDataMem_NEW((24*nrows) * sizeof(double))
    cdef int * iwork = <int *>PyDataMem_NEW((10*nrows) * sizeof(int))

    zheevr(&jobz, &rnge, &uplo, &nrows, H.data, &nrows, &vl, &vu, &il, &iu,
           &abstol, &M, eigvals, Z.data, &nrows, isuppz, work, &lwork,
           rwork, &lrwork, iwork, &liwork, &info)
    PyDataMem_FREE(work)
    PyDataMem_FREE(rwork)
    PyDataMem_FREE(isuppz)
    PyDataMem_FREE(iwork)
    if info != 0:
        if info < 0:
            raise Exception("Error in parameter : %s" & abs(info))
        else:
            raise Exception("Algorithm failed to converge")


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void ZGEEV(complex[::1,:] H, double * eigvals,
                complex[::1,:] Z, int nrows):
    """
    Computes the eigenvalues and vectors of a dense Hermitian matrix.
    Eigenvectors are returned in Z.

    Parameters
    ----------
    H : array_like
        Input Hermitian matrix.
    eigvals : array_like
        Input array to store eigen values.
    Z : array_like
        Output array of eigenvectors.
    nrows : int
        Number of rows in matrix.
    """
    cdef char jobvl = b'N'
    cdef char jobvr = b'V'
    cdef int i, j, k, lwork = -1
    cdef int same_eigv = 0
    cdef complex dot
    cdef complex wkopt
    cdef int info=0
    cdef complex * work

    #These need to be freed at end
    cdef complex * eival = <complex *>PyDataMem_NEW(nrows * sizeof(complex))
    cdef complex * vl = <complex *>PyDataMem_NEW(nrows * nrows *
                                                 sizeof(complex))
    cdef complex * vr = <complex *>PyDataMem_NEW(nrows * nrows *
                                                 sizeof(complex))
    cdef double * rwork = <double *>PyDataMem_NEW((2*nrows) * sizeof(double))

    # First call to get lwork
    zgeev(&jobvl, &jobvr, &nrows, &H[0,0], &nrows,
          eival, vl, &nrows, vr, &nrows,
          &wkopt, &lwork, rwork, &info)
    lwork = int(real(wkopt))
    work = <complex *>PyDataMem_NEW(lwork * sizeof(complex))
    # Solving
    zgeev(&jobvl, &jobvr, &nrows, &H[0,0], &nrows,
          eival, vl, &nrows, vr, &nrows, #&Z[0,0],
          work, &lwork, rwork, &info)
    for i in range(nrows):
        eigvals[i] = real(eival[i])
    # After using lapack, numpy...
    # lapack does not seems to have sorting function
    # zheevr sort but not zgeev
    cdef long[:] idx = np.argsort(np.array(<double[:nrows]> eigvals))
    for i in range(nrows):
        eigvals[i] = real(eival[idx[i]])
        for j in range(nrows):
            Z[j,i] = vr[j + idx[i]*nrows]

    for i in range(1, nrows):
        if cabs(eigvals[i] - eigvals[i-1]) < 1e-12:
            same_eigv += 1
            for j in range(same_eigv):
                dot = 0.
                for k in range(nrows):
                    dot += conj(Z[k,i-j-1]) * Z[k,i]
                for k in range(nrows):
                    Z[k,i] -= Z[k,i-j-1] * dot
                dot = 0.
                for k in range(nrows):
                    dot += conj(Z[k,i]) * Z[k,i]
                dot = sqrt(dot)
                for k in range(nrows):
                    Z[k,i] /= dot
        else:
            same_eigv = 0

    PyDataMem_FREE(work)
    PyDataMem_FREE(rwork)
    PyDataMem_FREE(vl)
    PyDataMem_FREE(vr)
    PyDataMem_FREE(eival)
    if info != 0:
        if info < 0:
            raise Exception("Error in parameter : %s" & abs(info))
        else:
            raise Exception("Algorithm failed to converge")


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double complex * ZGEMM(double complex * A, double complex * B,
                            int Arows, int Acols, int Brows, int Bcols,
                            int transA=0, int transB=0, double complex alpha=1,
                            double complex beta=0, double complex * C=NULL):
    if C == NULL:
        C = <double complex *>PyDataMem_NEW((Acols*Brows)*sizeof(double complex))
    cdef char tA, tB
    if transA == 0:
        tA = b'N'
    elif transA == 1:
        tA = b'T'
    elif transA == 2:
        tA = b'C'
    else:
        raise Exception('Invalid transA value.')
    if transB == 0:
        tB = b'N'
    elif transB == 1:
        tB = b'T'
    elif transB == 2:
        tB = b'C'
    else:
        raise Exception('Invalid transB value.')

    zgemm(&tA, &tB, &Arows, &Bcols, &Brows, &alpha,
          A, &Arows, B, &Brows, &beta, C, &Arows)
    return C


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void vec2mat_index(int nrows, int index, int[2] out) nogil:
    out[1] = index // nrows
    out[0] = index - nrows * out[1]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double skew_and_dwmin(double * evals, double[:,::1] skew,
                                unsigned int nrows) nogil:
    cdef double diff
    dw_min = DBL_MAX
    cdef size_t ii, jj
    for ii in range(nrows):
        for jj in range(nrows):
            diff = evals[ii] - evals[jj]
            skew[ii,jj] = diff
            if diff != 0:
                dw_min = fmin(fabs(diff), dw_min)
    return dw_min


@cython.boundscheck(False)
@cython.wraparound(False)
cdef Dense dense_to_eigbasis(Dense A, Dense evecs, double atol):
    cdef size_t kk, nrows=A.shape[0]
    cdef Dense temp = dense.zeros(nrows, nrows)
    cdef Dense out = dense.zeros(nrows, nrows)
    ZGEMM(A.data, evecs.data, nrows, nrows, nrows, nrows, not A.fortran, 0, 1, 0, temp.data)
    ZGEMM(evecs.data, temp.data, nrows, nrows, nrows, nrows, 2, 0, 1, 0, out.data)
    tidyup_dense(out, atol, True)
    return out


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
cpdef CSR cop_super_term(Dense cop, Dense evecs,
                         double complex alpha, double atol):
    cdef size_t kk, nrows=evecs.shape[0]
    cdef Dense cop_eig = dense_to_eigbasis(cop, evecs, atol)
    cdef CSR c = csr.from_dense(cop_eig)
    cdef size_t nnz = csr.nnz(c)
    imul_csr(c, alpha)
    cdef CSR cdc = matmul_csr(c.adjoint(), c)
    cdef CSR iden = csr.identity(nrows)
    cdef CSR out = add_csr(kron_csr(c.conj(), c), kron_csr(iden, cdc), scale=-0.5)
    return add_csr(out, kron_csr(cdc.transpose(), iden), scale=-0.5)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Data _br_term_cross(Data A, Data B,
                          double[:, ::1] skew, double[:, ::1] spectrum,
                          bint use_secular, double cutoff):

    cdef size_t kk, nrows=A.shape[0]
    cdef size_t I, J # vector index variables
    cdef int[2] ab, cd #matrix indexing variables
    cdef complex[:,:] A_mat = A.to_array()
    cdef complex[:,:] B_mat = B.to_array()
    cdef complex elem, ac_elem, bd_elem
    cdef vector[idxint] coo_rows, coo_cols
    cdef vector[double complex] coo_data
    cdef unsigned int nnz

    for I in range(nrows**2):
        vec2mat_index(nrows, I, ab)
        for J in range(nrows**2):
            vec2mat_index(nrows, J, cd)
            if (not use_secular) or (
                fabs(skew[ab[0],ab[1]] - skew[cd[0],cd[1]]) < (cutoff)
            ):
                elem = (A_mat[ab[0],cd[0]] * B_mat[cd[1],ab[1]]) * 0.5
                elem *= (spectrum[cd[0],ab[0]] + spectrum[cd[1],ab[1]])

                if ab[0] == cd[0]:
                    ac_elem = 0
                    for kk in range(nrows):
                        ac_elem += A_mat[cd[1],kk] * B_mat[kk,ab[1]] * spectrum[cd[1],kk]
                    elem -= 0.5*ac_elem

                if ab[1] == cd[1]:
                    bd_elem = 0
                    for kk in range(nrows):
                        bd_elem += A_mat[ab[0],kk] * B_mat[kk,cd[0]] * spectrum[cd[0],kk]
                    elem -= 0.5*bd_elem

                if (elem != 0):
                    coo_rows.push_back(I)
                    coo_cols.push_back(J)
                    coo_data.push_back(elem)

    return csr.from_coo_pointers(
        coo_rows.data(), coo_cols.data(), coo_data.data(),
        nrows*nrows, nrows*nrows, coo_rows.size())


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef CSR _br_term(Data A, double[:, ::1] skew, double[:, ::1] spectrum,
                   bint use_secular, double cutoff):

    cdef size_t kk, nrows = A.shape[0]
    cdef size_t I, J # vector index variables
    cdef int[2] ab, cd #matrix indexing variables
    cdef complex elem, ac_elem, bd_elem
    cdef complex[:,:] A_mat = A.to_array()
    cdef vector[idxint] coo_rows, coo_cols
    cdef vector[double complex] coo_data
    cdef unsigned int nnz

    for I in range(nrows**2):
        vec2mat_index(nrows, I, ab)
        for J in range(nrows**2):
            vec2mat_index(nrows, J, cd)
            if (not use_secular) or (
                fabs(skew[ab[0],ab[1]] - skew[cd[0],cd[1]]) < (cutoff)
            ):
                elem = (A_mat[ab[0],cd[0]] * A_mat[cd[1],ab[1]]) * 0.5
                elem *= (spectrum[cd[0],ab[0]] + spectrum[cd[1],ab[1]])

                if (ab[0]==cd[0]):
                    ac_elem = 0
                    for kk in range(nrows):
                        ac_elem += A_mat[cd[1],kk] * A_mat[kk,ab[1]] * spectrum[cd[1],kk]
                    elem -= 0.5*ac_elem

                if (ab[1]==cd[1]):
                    bd_elem = 0
                    for kk in range(nrows):
                        bd_elem += A_mat[ab[0],kk] * A_mat[kk,cd[0]] * spectrum[cd[0],kk]
                    elem -= 0.5*bd_elem

                if (elem != 0):
                    coo_rows.push_back(I)
                    coo_cols.push_back(J)
                    coo_data.push_back(elem)

    return csr.from_coo_pointers(
        coo_rows.data(), coo_cols.data(), coo_data.data(),
        nrows*nrows, nrows*nrows, coo_rows.size())


@cython.boundscheck(False)
@cython.wraparound(False)
def bloch_redfield_tensor(object H, list a_ops, list c_ops=[],
                          bool use_secular=True, double sec_cutoff=0.1,
                          double atol=qset.core['atol']):
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
    cdef double[:,::1] spectrum = np.zeros((nrows,nrows), dtype=float)
    cdef object R

    # Sanity checks for input parameters
    if not isinstance(H, Qobj):
        raise TypeError("H must be an instance of Qobj")

    for a in a_ops:
        if not isinstance(a[0], Qobj) or not a[0].isherm:
            raise TypeError("Operators in a_ops must be Hermitian Qobj.")

    cdef Dense H0 = to(Dense, H.data).reorder(fortran=True)
    cdef Dense evecs = dense.zeros(nrows, nrows, fortran=True)
    cdef double[::1] evals = np.zeros(nrows, dtype=float)

    ZHEEVR(H0, &evals[0], evecs, nrows)
    L = liou_from_diag_ham(evals)
    ekets = [Qobj(np.asarray(evecs.as_ndarray()[:,k]), dims=ket_dims)
             for k in range(nrows)]

    for cop in c_ops:
        L = add_csr(L, cop_super_term(to(Dense, cop.data), evecs, 1, atol))

    if not len(a_ops) == 0:
        #has some br operators and spectra
        dw_min = skew_and_dwmin(&evals[0], skew, nrows)
        cutoff = sec_cutoff * dw_min
        for a in a_ops:
            for i in range(nrows):
                for j in range(nrows):
                    spectrum[i,j] = a[1](skew[i,j])
            a_eigen = dense_to_eigbasis(to(Dense, a[0].data), evecs, atol)
            L = add_csr(L, _br_term(a_eigen, skew, spectrum,
                                    use_secular, sec_cutoff))

    R = Qobj(L, dims=sop_dims, type='super', copy=False)
    return R, ekets


@cython.boundscheck(False)
@cython.wraparound(False)
def BR_tensor(object H, list a_ops, bool use_secular=True,
              double sec_cutoff=0.1, double atol=qset.core['atol']):
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
    cdef object a
    cdef CSR L
    cdef int kk, i, j
    cdef int nrows = H.shape[0]
    cdef list op_dims = H.dims
    cdef list ket_dims = [op_dims[0], [1] * len(op_dims[0])]
    cdef list sop_dims = [[op_dims[0], op_dims[0]], [op_dims[1], op_dims[1]]]
    cdef list ekets
    cdef double dw_min
    cdef double[:,::1] skew = np.zeros((nrows,nrows), dtype=float)
    cdef double[:,::1] spectrum = np.zeros((nrows,nrows), dtype=float)
    cdef object R

    # Sanity checks for input parameters
    if not isinstance(H, Qobj):
        raise TypeError("H must be an instance of Qobj")

    for a in a_ops:
        if not isinstance(a[0], Qobj) or not a[0].isherm:
            raise TypeError("Operators in a_ops must be Hermitian Qobj.")

    cdef Dense H0 = to(Dense, H.data).reorder(fortran=True)
    cdef Dense evecs = dense.zeros(nrows, nrows, fortran=True)
    cdef double[::1] evals = np.zeros(nrows, dtype=float)

    ZHEEVR(H0, &evals[0], evecs, nrows)
    L = csr.zeros(nrows*nrows, nrows*nrows)
    ekets = [Qobj(np.asarray(evecs.as_ndarray()[:,k]), dims=ket_dims)
             for k in range(nrows)]

    if not len(a_ops) == 0:
        #has some br operators and spectra
        dw_min = skew_and_dwmin(&evals[0], skew, nrows)
        cutoff = sec_cutoff * dw_min
        for a in a_ops:
            for i in range(nrows):
                for j in range(nrows):
                    spectrum[i,j] = a[1](skew[i,j])
            a_eigen = dense_to_eigbasis(to(Dense, a[0].data), evecs, atol)
            L = add_csr(L, _br_term(a_eigen, skew, spectrum,
                                    use_secular, sec_cutoff))

    R = Qobj(L, dims=sop_dims, type='super', copy=False)
    return R, ekets


cdef class CBR_RHS(CQobjFunc):
    cdef CQobjEvo H
    cdef CQobjEvo c_ops
    cdef list a_ops
    cdef list spectra
    cdef bint use_secular, H_fortran, has_c_op, return_eigen
    cdef double sec_cutoff, atol
    cdef size_t nrows

    cdef double[:, ::1] skew
    cdef double[:, ::1] spectrum
    cdef int * isuppz
    cdef int * iwork
    cdef complex * work
    cdef double * rwork
    cdef double * eigvals
    cdef readonly Dense evecs, out, eig_vec, temp, op_eig

    def __init__(self, H, a_ops, spectra, c_ops, use_secular, sec_cutoff, atol,
                 eigen=False):
        self.H = H.compiled_qobjevo
        self.has_c_op = c_ops is not None
        if c_ops is not None:
            self.c_ops = c_ops.compiled_qobjevo
        self.a_ops = [op.compiled_qobjevo for op in a_ops]
        self.spectra = spectra
        self.use_secular = use_secular
        self.sec_cutoff = sec_cutoff
        self.nrows = H.shape[0]
        self.atol = atol
        self.shape = self.nrows**2, self.nrows**2
        self.dims = [self.H.dims, self.H.dims]
        self.issuper = True
        self.return_eigen = eigen

        self.skew = <double[:self.nrows,:self.nrows]><double *>PyDataMem_NEW_ZEROED(self.nrows**2, sizeof(double))
        self.spectrum = <double[:self.nrows,:self.nrows]><double *>PyDataMem_NEW_ZEROED(self.nrows**2, sizeof(double))

        self.isuppz = <int *>PyDataMem_NEW(2*self.nrows * sizeof(int))
        self.work = <complex *>PyDataMem_NEW(18*self.nrows * sizeof(complex))
        self.rwork = <double *>PyDataMem_NEW(24*self.nrows * sizeof(double))
        self.iwork = <int *>PyDataMem_NEW(10*self.nrows * sizeof(int))
        self.eigvals = <double *>PyDataMem_NEW(self.nrows * sizeof(double))

        self.eig_vec = dense.zeros(self.nrows, self.nrows, fortran=True)
        self.out = dense.zeros(self.nrows, self.nrows, fortran=True)
        self.temp = dense.zeros(self.nrows, self.nrows, fortran=True)
        self.evecs = dense.zeros(self.nrows, self.nrows, fortran=True)
        self.op_eig = dense.zeros(self.nrows, self.nrows, fortran=True)

    def __dealloc__(self):
        PyDataMem_FREE(self.work)
        PyDataMem_FREE(self.rwork)
        PyDataMem_FREE(self.iwork)
        PyDataMem_FREE(self.isuppz)
        PyDataMem_FREE(self.eigvals)

        PyDataMem_FREE(&self.skew[0,0])
        PyDataMem_FREE(&self.spectrum[0,0])

    def call(self, double t, bint data=False, bint return_eigen=False):
        cdef size_t i
        cdef double dw_min, cutoff
        cdef CSR out_eigen
        cdef Data out_fock, R_out

        if False and self.has_c_op and self.eigen:
            pass
            # TODO: Better to use `cop_super_term`
            # and compute c_ops directly in eigen basis?
        elif self.has_c_op:
            out_fock = self.c_ops.call(t, data=True)
        else:
            out_fock = csr.zeros(self.nrows**2, self.nrows**2)

        out_eigen = csr.zeros(self.nrows**2, self.nrows**2)
        self.H_eigen(t)
        out_eigen = add_csr(liou_from_diag_ham(<double[:self.nrows]> self.eigvals), out_eigen)
        dw_min = skew_and_dwmin(self.eigvals, self.skew, self.nrows)
        cutoff = self.sec_cutoff * dw_min
        for i in range(len(self.a_ops)):
            for col in range(self.nrows):
                for row in range(self.nrows):
                    self.spectrum[row, col] = self.spectra[i](t, self.skew[row, col])
            self.to_eigbasis(self.a_ops[i].call(t, data=True), self.op_eig)
            out_eigen = add_csr(out_eigen, _br_term(self.op_eig, self.skew,
                                                    self.spectrum,
                                                    self.use_secular, cutoff)
                               )
        if return_eigen:
            R_out = self.D_to_eigenbasis(out_eigen, out_fock)
            if not data:
                out = Qobj(R_out, dims=self.dims, type="super", copy=False)
                ekets = [Qobj(np.asarray(self.evecs.as_ndarray()[:,k]),
                              dims=[self.dims[0], [1] * len(self.dims[0])])
                         for k in range(self.nrows)]
                return R_out, ekets
            else:
                return R_out, self.evecs.copy()
        else:
            R_out = self.R_to_fockbasis(out_eigen, out_fock)
            if data:
                return R_out
            return Qobj(R_out, dims=self.dims, type="super", copy=False)

    cpdef Dense matmul_dense(self, double t, Dense vec_fock, Dense out_fock=None):
        cdef size_t i, col, row, nrows = self.nrows
        cdef double dw_min
        cdef CSR R_mat
        if not vec_fock.fortran:
            vec_fock = vec_fock.reorder(1)
        # c_ops is done first in the "outside" basis
        if self.has_c_op:
            out_fock = self.c_ops.matmul_dense(t, vec_fock, out=out_fock)
        elif out_fock is None:
            out_fock = mul_dense(vec_fock, 0)

        self.H_eigen(t)
        dw_min = skew_and_dwmin(self.eigvals, self.skew, nrows)

        self.to_eigbasis(vec_fock, self.eig_vec)
        self.apply_liou(self.eig_vec, self.out)

        column_stack_dense(self.eig_vec, True)
        column_stack_dense(self.out, True)

        for i in range(len(self.a_ops)):
            for col in range(nrows):
                for row in range(nrows):
                    self.spectrum[row, col] = self.spectra[i](t, self.skew[row, col])
            self.to_eigbasis(self.a_ops[i].call(t, data=True), self.op_eig)
            R_mat = _br_term(self.op_eig, self.skew, self.spectrum,
                             self.use_secular, dw_min*self.sec_cutoff)
            matmul_csr_dense_dense(R_mat, self.eig_vec, scale=1, out=self.out)

        column_unstack_dense(self.eig_vec, True)
        column_unstack_dense(self.out, True)

        self.vec_to_fockbasis(self.out, out_fock)
        return out_fock

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void H_eigen(self, double t):
        cdef char jobz = b'V'
        cdef char rnge = b'A'
        cdef char uplo = b'L'
        cdef double vl = 1, vu = 1, abstol = 0
        cdef int il = 1, iu = 1, nrows = self.nrows
        cdef int lwork = 18*nrows
        cdef int lrwork = 24*nrows
        cdef int liwork = 10*nrows
        cdef int info = 0, M = 0
        cdef Dense H = self.H.call(t, data=True)
        self.H_fortran = H.fortran
        zheevr(&jobz, &rnge, &uplo, &nrows, H.data, &nrows, &vl, &vu, &il, &iu,
               &abstol, &M, self.eigvals, self.evecs.data, &nrows, self.isuppz,
               self.work, &lwork, self.rwork, &lrwork, self.iwork, &liwork,
               &info)
        if info != 0:
            if info < 0:
                raise Exception("Error in parameter : %s" & abs(info))
            else:
                raise Exception("Algorithm failed to converge")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void apply_liou(self, Dense vec, Dense out):
        cdef unsigned int nnz = 0
        cdef size_t ii, jj
        imul_dense(out, 0)
        for ii in range(self.nrows):
            for jj in range(self.nrows):
                out.data[nnz] = (-1j * vec.data[nnz] *
                                 (self.eigvals[jj] - self.eigvals[ii]))
                nnz += 1

    cpdef void to_eigbasis(self, Dense vec, Dense out):
        cdef size_t nrows = self.nrows
        if self.H_fortran:
            ZGEMM(vec.data, self.evecs.data, nrows, nrows, nrows, nrows,
                  not vec.fortran, 0, 1., 0., self.temp.data)
            ZGEMM(self.evecs.data, self.temp.data, nrows, nrows, nrows, nrows,
                  2, 0, 1., 0., out.data)
        else:
            ZGEMM(self.evecs.data, vec.data, nrows, nrows, nrows, nrows,
                  1, 1 + <int>vec.fortran, 1., 0., self.temp.data)
            ZGEMM(self.evecs.data, self.temp.data, nrows, nrows, nrows, nrows,
                  1, 2, 1., 0., out.data)

    cpdef void vec_to_fockbasis(self, Dense out_eigen, Dense out):
        cdef size_t nrows = self.nrows
        if self.H_fortran:
            ZGEMM(out_eigen.data, self.evecs.data, nrows, nrows, nrows, nrows,
                  0, 2, 1., 0., self.temp.data)
            ZGEMM(self.evecs.data, self.temp.data, nrows, nrows, nrows, nrows,
                  0, 0, 1., 1., out.data)
        else:
            ZGEMM(out_eigen.data, self.evecs.data, nrows, nrows, nrows, nrows,
                  2, 1, 1., 0., self.temp.data)
            ZGEMM(self.temp.data, self.evecs.data, nrows, nrows, nrows, nrows,
                  2, 1, 1., 1., out.data)

    cpdef Data D_to_eigenbasis(self, CSR R_eigen, Data C_fock):
        cdef size_t nrows = self.nrows
        cdef CSR S, c = csr.from_dense(self.evecs)
        if not self.H_fortran:
            S = kron_csr(c.adjoint(), c.transpose())
        else:
            S = kron_csr(c.transpose(), c.adjoint())
        return add(matmul_csr(S, matmul(C_fock, S.adjoint())), R_eigen)

    cpdef Data R_to_fockbasis(self, CSR R_eigen, Data C_fock):
        cdef size_t nrows = self.nrows
        cdef CSR S, c = csr.from_dense(self.evecs)
        if not self.H_fortran:
            S = kron_csr(c.adjoint(), c.transpose())
        else:
            S = kron_csr(c.transpose(), c.adjoint())
        return add(matmul_csr(S.adjoint(), matmul_csr(R_eigen, S)), C_fock)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void br_term_mult(self, double t, Dense A, double dw_min,
                           Dense vec, Dense out):
        cdef size_t kk
        cdef size_t I, J # vector index variables
        cdef int[2] ab, cd #matrix indexing variables
        cdef int nrows = self.nrows
        cdef double dw, cutoff = dw_min * self.sec_cutoff
        self.to_eigbasis(A, self.op_eig)
        cdef double complex[::1, :] A_eig = self.op_eig.to_array()
        cdef complex elem, ac_elem, bd_elem
        cdef vector[idxint] coo_rows, coo_cols
        cdef vector[double complex] coo_data

        for I in range(nrows**2):
            vec2mat_index(nrows, I, ab)
            for J in range(nrows**2):
                vec2mat_index(nrows, J, cd)
                dw = fabs(self.skew[ab[0], ab[1]] - self.skew[cd[0], cd[1]])

                if (not self.use_secular) or (dw < cutoff):
                    elem = (A_eig[ab[0], cd[0]] * A_eig[cd[1], ab[1]]) * 0.5
                    elem *= (self.spectrum[cd[0], ab[0]] +
                             self.spectrum[cd[1], ab[1]])

                    if (ab[0]==cd[0]):
                        ac_elem = 0
                        for kk in range(nrows):
                            ac_elem += (A_eig[cd[1], kk] *
                                        A_eig[kk, ab[1]] *
                                        self.spectrum[cd[1], kk])
                        elem -= 0.5 * ac_elem

                    if (ab[1]==cd[1]):
                        bd_elem = 0
                        for kk in range(nrows):
                            bd_elem += (A_eig[ab[0], kk] *
                                        A_eig[kk, cd[0]] *
                                        self.spectrum[cd[0], kk])
                        elem -= 0.5 * bd_elem

                    if (elem != 0):
                        coo_rows.push_back(I)
                        coo_cols.push_back(J)
                        coo_data.push_back(elem)

        cdef CSR matrix = csr.from_coo_pointers(
            coo_rows.data(), coo_cols.data(), coo_data.data(),
            nrows*nrows, nrows*nrows, coo_rows.size()
        )
        matmul_csr_dense_dense(matrix, vec, scale=1, out=out)

    @property
    def eigen_values(self):
        return np.array(<double[:self.nrows]> self.eigvals)

    cpdef Data matmul(self, double t, Data matrix):
        cdef Dense mat = to(Dense, matrix)
        return self.matmul_dense(t, mat.reorder(True))
