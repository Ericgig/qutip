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

cimport numpy as cnp
import numpy as np

from scipy.linalg.cython_lapack cimport zheevr, zgeev
from scipy.linalg.cython_blas cimport zgemm, zgemv, zaxpy
from scipy.linalg cimport cython_blas as blas

cimport cython

from qutip.core.cy.cqobjevo cimport CQobjEvo
from qutip.core.data cimport CSR, csr, Dense, dense, idxint
from qutip.core.data.add cimport add_csr
from qutip.core.data.kron cimport kron_csr
from qutip.core.data.matmul cimport matmul_csr, matmul_csr_dense_dense
from qutip import settings
eigh_unsafe = settings.install["eigh_unsafe"]

cnp.import_array()

import sys

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
cdef complex[::1,:] farray_alloc(int nrows):
    """
    Allocate a complex zero array in fortran-order for a
    square matrix.

    Parameters
    ----------
    nrows : int
        Number of rows and columns in the matrix.

    Returns
    -------
    fview : memview
        A zeroed memoryview in fortran-order.
    """
    cdef double complex * temp = <double complex *>PyDataMem_NEW_ZEROED(nrows*nrows,sizeof(complex))
    cdef complex[:,::1] cview = <double complex[:nrows, :nrows]> temp
    cdef complex[::1,:] fview = cview.T
    return fview


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void dense_add_mult(complex[::1,:] A,
                  complex[::1,:] B,
                  double complex alpha) nogil:
    """
    Performs the dense matrix multiplication A = A + (alpha*B)
    where A and B are complex 2D square matrices,
    and alpha is a complex coefficient.

    Parameters
    ----------
    A : ndarray
        Complex matrix in f-order that is to be overwritten
    B : ndarray
        Complex matrix in f-order.
    alpha : complex
        Coefficient in front of B.

    """
    cdef int nrows2 = A.shape[0]**2
    cdef int inc = 1
    zaxpy(&nrows2, &alpha, &B[0,0], &inc, &A[0,0], &inc)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void ZHEEVR(complex[::1,:] H, double * eigvals,
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
    if use_zgeev:
        ZGEEV(H, eigvals, Z, nrows)
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

    zheevr(&jobz, &rnge, &uplo, &nrows, &H[0,0], &nrows, &vl, &vu, &il, &iu,
           &abstol, &M, eigvals, &Z[0,0], &nrows, isuppz, work, &lwork,
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
cdef void ZGEMV(double complex * A, double complex * vec,
                        double complex * out,
                       int Arows, int Acols, int transA = 0,
                       double complex alpha=1, double complex beta=1):
    cdef char tA
    cdef int idx = 1, idy = 1
    if transA == 0:
        tA = b'N'
    elif transA == 1:
        tA = b'T'
    elif transA == 2:
        tA = b'C'
    else:
        raise Exception('Invalid transA value.')
    zgemv(&tA, &Arows, &Acols, &alpha, A, &Arows, vec, &idx, &beta, out, &idy)


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


cdef class BR_RHS:
    cdef CQobjEvo H
    cdef list c_ops
    cdef list a_ops
    cdef list spectra
    cdef bint use_secular
    cdef double sec_cutoff
    cdef size_t nrows
    cdef double atol

    cdef double[:, ::1] skew
    cdef double[:, ::1] spectrum
    cdef int * isuppz
    cdef int * iwork
    cdef complex * work
    cdef double * rwork
    cdef double * eigvals
    cdef Dense evecs, out, eig_vec, temp, op_eig

    def __init__(self, H, c_ops, a_ops, spectra, use_secular, sec_cutoff, atol):
        self.H = H.compiled_qobjevo
        self.c_ops = [op.compiled_qobjevo for op in c_ops]
        self.a_ops = [op.compiled_qobjevo for op in a_ops]
        self.spectra = spectra
        self.use_secular = use_secular
        self.sec_cutoff = sec_cutoff
        self.nrows = H.shape[0]
        self.atol = atol

        self.skew = <double[:self.nrows,:self.nrows]><double *>PyDataMem_NEW_ZEROED(self.nrows**2, sizeof(double))
        self.spectrum = <double[:self.nrows,:self.nrows]><double *>PyDataMem_NEW_ZEROED(self.nrows**2, sizeof(double))

        self.isuppz = <int *>PyDataMem_NEW(2*self.nrows * sizeof(int))
        self.work = <complex *>PyDataMem_NEW(18*self.nrows * sizeof(complex))
        self.rwork = <double *>PyDataMem_NEW(24*self.nrows * sizeof(double))
        self.iwork = <int *>PyDataMem_NEW(10*self.nrows * sizeof(int))
        self.eigvals = <double *>PyDataMem_NEW(self.nrows * sizeof(double))

        cdef Dense H_data = H.compiled_qobjevo.call(0, data=True)

        self.eig_vec = dense.zeros(self.nrows, self.nrows, fortran=True)
        self.out = dense.zeros(self.nrows, self.nrows, fortran=True)
        self.temp = dense.zeros(self.nrows, self.nrows, fortran=True)
        self.evecs = dense.zeros(self.nrows, self.nrows, fortran=H_data.fortran)
        self.op_eig = dense.zeros(self.nrows, self.nrows, fortran=True)

    def __dealloc__(self):
        PyDataMem_FREE(self.work)
        PyDataMem_FREE(self.rwork)
        PyDataMem_FREE(self.iwork)
        PyDataMem_FREE(self.isuppz)
        PyDataMem_FREE(self.eigvals)
        PyDataMem_FREE(&self.skew[0,0])
        PyDataMem_FREE(&self.spectrum[0,0])

    def __call__(self, t, vec):
        cdef size_t col, row, nrows = self.nrows
        cdef double dw_min

        # self.eigvals = self.H_eigen(t)
        self.H_eigen(t)
        dw_min = skew_and_dwmin(self.eigvals, self.skew, nrows)
        self.to_eigbasis(Dense(vec, copy=False), self.eig_vec)
        self.apply_liou(self.eig_vec, self.out)

        for c in self.c_ops:
            self.cop_super_mult(c.call(t, data=True),
                                self.eig_vec, self.out)

        for i in range(len(self.a_ops)):
            for col in range(nrows):
                for row in range(nrows):
                    self.spectrum[row, col] = self.spectra[i](t, self.skew[row, col])
            self.br_term_mult(t, self.a_ops[i].call(t, data=True),
                              dw_min, self.eig_vec, self.out)

        return self.vec_to_fockbasis(self.out)

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
        if use_zgeev:
            ZGEEV(H.as_ndarray(), self.eigvals,
                  self.evecs.as_ndarray(), nrows)
            return self.eigvals
        zheevr(&jobz, &rnge, &uplo, &nrows, H.data, &nrows, &vl, &vu, &il, &iu,
               &abstol, &M, self.eigvals, self.evecs.data, &nrows, self.isuppz,
               self.work, &lwork, self.rwork, &lrwork, self.iwork, &liwork,
               &info)
        if info != 0:
            if info < 0:
                raise Exception("Error in parameter : %s" & abs(info))
            else:
                raise Exception("Algorithm failed to converge")
        return self.eigvals

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void cop_super_mult(self, Dense cop, Dense vec, Dense out):
        # TODO: cop is probably sparse.
        # Is it worth converting it to CSR after changing to eigen space?
        cdef size_t kk
        cdef int nrows = self.nrows

        imul_dense(self.temp, 0.)
        self.to_eigbasis(cop, self.op_eig)
        cdef Dense cdc = matmul_dense(self.op_eig.adjoint(), self.op_eig)
        matmul_dense(self.op_eig.adjoint(), vec, scale=1, out=self.temp)
        matmul_dense(self.temp, self.op_eig, scale=1, out=out)
        matmul_dense(vec, cdc, scale=-0.5, out=out)
        matmul_dense(cdc, vec, scale=-0.5, out=out)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void apply_liou(self, Dense vec, Dense out):
        cdef unsigned int nnz = 0
        cdef size_t ii, jj
        for ii in range(self.nrows):
            for jj in range(self.nrows):
                out.data[nnz] = (1j * vec.data[nnz] *
                                 (self.eigvals[ii] - self.eigvals[jj]))
                nnz += 1

    cpdef void to_eigbasis(self, Dense vec, Dense out):
        cdef size_t nrows = self.nrows
        if self.evecs.fortran:
            ZGEMM(vec.data, self.evecs.data, nrows, nrows, nrows, nrows,
                  not vec.fortran, 0, 1., 0., self.temp.data)
            ZGEMM(self.evecs.data, self.temp.data, nrows, nrows, nrows, nrows,
                  2, 0, 1., 0., out.data)
        else:
            ZGEMM(vec.data, self.evecs.data, nrows, nrows, nrows, nrows,
                  vec.fortran, 2, 1., 0., self.temp.data)
            ZGEMM(self.temp.data, self.evecs.data, nrows, nrows, nrows, nrows,
                  1, 1, 1., 0., out.data)

    cpdef cnp.ndarray[complex, ndim=1, mode='fortran'] vec_to_fockbasis(self, Dense out_eigen):
        cdef size_t nrows = self.nrows
        cdef Dense out = dense.zeros(nrows**2, 1)
        ZGEMM(out_eigen.data, self.evecs.data, nrows, nrows, nrows, nrows,
              0, 2, 1., 0., self.temp.data)
        ZGEMM(self.evecs.data, self.temp.data, nrows, nrows, nrows, nrows,
              0, 0, 1., 0., out.data)
        return out.as_ndarray().ravel()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void br_term_mult(self, double t, Dense A, double dw_min,
                           Dense vec, Dense out):
        cdef size_t kk
        cdef size_t I, J # vector index variables
        cdef int[2] ab, cd #matrix indexing variables
        cdef int nrows = self.nrows
        cdef double dw, cutoff = self.dw_min * self.sec_cutoff
        self.to_eigbasis(A, self.op_eig)
        cdef double complex[::1, :] A_eig = self.op_eig.as_ndarray()
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
