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
from qutip.core.cy.coefficient cimport Coefficient
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


__all__ = ['make_spectra', 'bloch_redfield_tensor', 'BR_tensor', 'CBR_RHS']


@cython.overflowcheck(True)
cdef size_t _mul_checked(size_t a, size_t b) except? -1:
    return a * b

def make_spectra(f):
    if isinstance(f, Spectrum):
        return f
    elif isinstance(f, str):
        coeff = coefficient(f, args={"w":0})
        return Spectrum_Str(coeff)
    elif isinstance(f, (np.ndarray, Cubic_Spline)):
        coeff = coefficient(f)
        return Spectrum_array(coeff)
    elif callable(f):
        try:
            f(0, 0)
            return Spectrum_func_t(f)
        except Exception:
            return Spectrum(f)


cdef class Spectrum:
    # wrapper to use Coefficient for spectrum function in string format
    cdef object _func

    def __init__(self, func):
        self._func = func

    cpdef _call_t(self, double t, double w):
        return self._func(w)

    def __call__(self, double w):
        return self._func(w)


cdef class Spectrum_Str(Spectrum):
    # wrapper to use Coefficient for spectrum function in string format
    cdef Coefficient _coeff

    def __init__(self, coeff):
        self._coeff = coeff

    cpdef _call_t(self, double t, double w):
        self._coeff.arguments({"w":w})
        return real(self._coeff._call(t))

    def __call__(self, double w):
        self._coeff.arguments({"w":w})
        return real(self._coeff._call(0))


cdef class Spectrum_array(Spectrum):
    # wrapper to use Coefficient for spectrum function in string format
    cdef Coefficient _coeff

    def __init__(self, coeff):
        self._coeff = coeff

    cpdef _call_t(self, double t, double w):
        return real(self._coeff._call(w))

    def __call__(self, double w):
        return real(self._coeff._call(w))


cdef class Spectrum_func_t(Spectrum):
    # wrapper to use Coefficient for spectrum function in string format
    def __init__(self, func):
        self._func = func

    cpdef _call_t(self, double t, double w):
        return self._func(t, w)

    def __call__(self, double w):
        return self._func(0, w)


cdef class _eigensolver():
    #
    cdef int nrows
    cdef object np_datas
    cdef Dense evecs

    cdef int * isuppz
    cdef int * iwork
    cdef complex * work
    cdef double * rwork
    cdef double * eigvals

    def __init__(self, nrows):
        self.nrows = nrows
        self.np_datas = [
            np.zeros(2 * self.nrows, dtype=np.int32),
            np.zeros(18 * self.nrows, dtype=np.complex128),
            np.zeros(24 * self.nrows, dtype=np.float64),
            np.zeros(10 * self.nrows, dtype=np.int32),
            np.zeros(self.nrows, dtype=np.float64),
        ]
        self.isuppz = <int *> cnp.PyArray_GETPTR2(self.np_datas[0], 0, 0)
        self.work = <complex *> cnp.PyArray_GETPTR2(self.np_datas[1], 0, 0)
        self.rwork = <double *> cnp.PyArray_GETPTR2(self.np_datas[2], 0, 0)
        self.iwork = <int *> cnp.PyArray_GETPTR2(self.np_datas[3], 0, 0)
        self.eigvals = <double *> cnp.PyArray_GETPTR2(self.np_datas[4], 0, 0)

        self.evecs = dense.zeros(self.nrows, self.nrows, fortran=True)

    def __call__(self, Dense data):
        cdef char jobz = b'V'
        cdef char rnge = b'A'
        cdef char uplo = b'L'
        cdef double vl=1, vu=1, abstol=0
        cdef int il=1, iu=1, nrows=self.nrows
        cdef int lwork = 18 * nrows
        cdef int lrwork = 24 * nrows, liwork = 10 * nrows
        cdef int info=0, M=0

        zheevr(&jobz, &rnge, &uplo, &self.nrows, data.data, &nrows, &vl, &vu, &il, &iu,
               &abstol, &M, self.eigvals, self.evecs.data, &nrows,
               self.isuppz, self.work, &lwork,
               self.rwork, &lrwork, self.iwork, &liwork, &info)

        if info != 0:
            if info < 0:
                raise Exception("Error in parameter : %s" & abs(info))
            else:
                raise Exception("Algorithm failed to converge")

        return self.np_datas[4], self.evecs


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


cdef void _to_eigbasis(Dense evecs, Dense fock, bint H_fortran,
                       Dense out=None, Dense temp=None):
    # evecs is usually dense or diagonal
    # Use ZGEMM instead of dense.matmul since it can do A.dag @ B in one
    # operation without making copy of the array.
    cdef size_t nrows = evecs.shape[0]
    if evecs.shape[0] != evecs.shape[1]:
        raise ValueError
    if fock.shape[0] != evecs.shape[0] or fock.shape[1] != evecs.shape[1]:
        raise ValueError
    if out is None:
        out = data.dense.empty(evecs.shape[0], evecs.shape[1], H_fortran)
    elif out.shape[0] != evecs.shape[0] or out.shape[1] != evecs.shape[1]:
        raise ValueError
    if temp is None:
        temp = data.dense.zeros(evecs.shape[0], evecs.shape[1])
    elif temp.shape[0] != evecs.shape[0] or temp.shape[1] != evecs.shape[1]:
        raise ValueError

    if H_fortran a == out.fortran:
        # Z.dag @ rho @ Z
        ZGEMM(fock.data, evecs.data, nrows, nrows, nrows, nrows,
              not fock.fortran, 0, 1., 0., temp.data)
        ZGEMM(evecs.data, temp.data, nrows, nrows, nrows, nrows,
              2, 0, 1., 0., out.data)
    else:
        # eigen solver gives Z* instead of Z if not H.fortran
        # Z.T @ rho @ Z*
        ZGEMM(evecs.data, fock.data, nrows, nrows, nrows, nrows,
              1, 1 + <int>fock.fortran, 1., 0., temp.data)
        ZGEMM(evecs.data, temp.data, nrows, nrows, nrows, nrows,
              1, 2, 1., 0., out.data)


cpdef void _to_fockbasis(Dense evecs, Dense eigen, bint H_fortran,
                         Dense out=None, Dense temp=None):
    # evecs is usually dense or diagonal
    # Use ZGEMM instead of dense.matmul since it can do A.dag @ B in one
    # operation without making copy of the array.
    cdef size_t nrows = evecs.shape[0]
    if evecs.shape[0] != evecs.shape[1]:
        raise ValueError
    if eigen.shape[0] != evecs.shape[0] or eigen.shape[1] != evecs.shape[1]:
        raise ValueError
    if out is None:
        out = data.dense.empty(evecs.shape[0], evecs.shape[1], H_fortran)
    elif out.shape[0] != evecs.shape[0] or out.shape[1] != evecs.shape[1]:
        raise ValueError
    if temp is None:
        temp = data.dense.zeros(evecs.shape[0], evecs.shape[1])
    elif temp.shape[0] != evecs.shape[0] or temp.shape[1] != evecs.shape[1]:
        raise ValueError
    if H_fortran == out.fortran:
        # Z @ rho @ Z.dag
        ZGEMM(eigen.data, evecs.data, nrows, nrows, nrows, nrows,
              not eigen.fortran, 2, 1., 0., temp.data)
        ZGEMM(evecs.data, temp.data, nrows, nrows, nrows, nrows,
              0, 0, 1., 1., out.data)
    else:
        # Z* @ rho @ Z.T
        ZGEMM(eigen.data, evecs.data, nrows, nrows, nrows, nrows,
              1 + eigen.fortran, 1, 1., 0., temp.data)
        ZGEMM(temp.data, evecs.data, nrows, nrows, nrows, nrows,
              2, 1, 1., 1., out.data)


cpdef Data _superop_to_eigenbasis(Data evecs, Data fock, bint H_fortran):
    # kron only available as CSR...
    cdef size_t nrows = evecs.shape[0]
    cdef data.Data S
    if not H_fortran:
        S = data.kron(evecs.adjoint(), evecs.transpose())
    else:
        S = data.kron(evecs.transpose(), evecs.adjoint())
    return data.matmul(S, data.matmul(fock, S.adjoint()))


cpdef Data _superop_to_fockbasis(Data evecs, Data eig, bint H_fortran):
    # kron only available as CSR...
    cdef size_t nrows = evecs.shape[0]
    cdef data.Data S
    if not H_fortran:
        S = data.kron(evecs.adjoint(), evecs.transpose())
    else:
        S = data.kron(evecs.transpose(), evecs.adjoint())
    return data.matmul(S.adjoint(), data.matmul(eig, S))


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
    cdef CSR out = csr.empty(_mul_checked(nrows, nrows),
                             _mul_checked(nrows, nrows),
                             _mul_checked(nrows, nrows))
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
    L = csr.zeros(_mul_checked(nrows, nrows), _mul_checked(nrows, nrows))
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
