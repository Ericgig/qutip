#cython: language_level=3
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
from qutip.core.cy.qobjevo cimport QobjEvo
from scipy.linalg.cython_blas cimport zgemm
from qutip.core.data cimport Data, CSR, Dense
import numpy as np
from libc.float cimport DBL_MAX
from qutip.core.data.eigen import eigs
import qutip.core.data as _data
from qutip import Qobj

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


__all__ = ['SpectraCoefficient', '_EigenBasisTransform']


cdef SpectraCoefficient(Coefficient):
    """Spectrum Coefficient composed of 2 coefficients, 1 for the
    time dependence and one for the frequency denpendence.
    """
    def __init__(self, Coefficient coeff_w, Coefficient coeff_t=None, double w=0):
        self.coeff_t = coeff_t
        self.coeff_w = coeff_w
        self.w = w

    cdef complex _call(self, double t) except *:
        if self.coeff_t is None:
            return self.coeff_w(self.w)
        return self.coeff_t(t) * self.coeff_w(self.w)

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`Coefficient`."""
        return SpectraCoefficient(self.coeff_t, self.coeff_w, self.w)

    def replace(self, *, _args=None, w=None, **kwargs):
        if w is not None:
            return SpectraCoefficient(self.coeff_t, self.coeff_w, w)
        if _args:
            kwargs.update(_args)
        if 'w' in kwargs:
            return SpectraCoefficient(self.coeff_t, self.coeff_w, kwargs['w'])
        return self


cdef Data _apply_trans(Data original, int trans):
    cdef Data out
    if trans == 0:
        out = original
    elif trans == 1:
        out = original.transpose()
    elif original == 2:
        out = original.adjoin()
    elif trans == 3:
        out = original.conj()
    return out


cdef char _fetch_trans_code(int trans):
    if transleft == 0:
        return b'N'
    elif transleft == 1:
        return = b'T'
    elif transleft == 3:
        return = b'C'


cdef Data matmul_var(Data left, Data right, int transleft, int transright,
                     double complex alpha=1, Data out=None):
    """
    matmul which product matrices can be transposed or adjoint.
    out = transleft(left) @ transright(right)

    trans[left, right]:
        0 : Normal
        1 : Transpose
        2 : Conjugate
        3 : Adjoint
    """
    # TODO : Should this be supported in data.matmul?
    #        Tensorflow support this option.
    cdef int lda, ldb
    cdef double complex beta
    if not (
        type(left) is Dense
        and type(right) is Dense
        and (type(out) is None or type(out) is Dense)
    ):
        left = _apply_trans(left, transleft)
        right = _apply_trans(right, transright)
        return _data.add(out, _data.matmul(left, right), alpha)

    if out is None:
        out = _data.dense.empty(left.shape[0], right.shape[1], right.fortran)

    transleft &= left.fortran
    transright &= right.fortran

    if transleft + transright == 5:
        out = matmul_var(left, right, transleft, transright, alpha, out)
        return out.conj()

    unavail = 3 - out.fortran
    any_unavail = (transleft == unavail or transright == unavail)
    if any_unavail:
        transleft &= 1
        transright &= 1

    if out.fortran == any_unavail:
        lda = left.shape[0] if left.fortran else left.shape[1]
        ldb = right.shape[0] if right.fortran else right.shape[1]
        beta = 1
        zgemm(&_fetch_trans_code(transleft), &_fetch_trans_code(transright),
              &left.shape[0], &right.shape[1], &left.shape[1],
              &alpha, left.data, &lda, right.data, &ldb,
              &beta, out.data, &left.shape[0])
    else:
        transleft &= 1
        transright &= 1
        ldb = left.shape[0] if left.fortran else left.shape[1]
        lda = right.shape[0] if right.fortran else right.shape[1]
        beta = 1
        zgemm(&_fetch_trans_code(transright), &_fetch_trans_code(transleft),
              &right.shape[1], &left.shape[0], &left.shape[1],
              &alpha, right.data, &lda, left.data, &ldb,
              &beta, out.data, &right.shape[1])
    if any_unavail:
        out.fortran = not out.fortran
    return out


class _eigen_qevo:
    # TODO: as _Element?
    cdef:
        QobjEvo qevo
        dict args

    def __init__(self, qevo):
        self.qevo = qevo
        self.args = None

    def __call__(self, t, args):
        if args != self.args:
            self.args = args
            self.qevo = QobjEvo(self, args=args)
        data = eigs(self.qevo._call(t), True, True)
        return Qobj(data, copy=False, dims=self.qevo.dims)


cdef class _EigenBasisTransform:
    """
    For an hermitian operator, compute the eigenvalues and eigenstates and do
    the base change to and from that eigenbasis.

    parameter
    ---------
    oper : QobjEvo
        Hermitian operator for which to compute the eigenbasis.

    sparse : bool [False]
        Whether to use sparse solver for eigen decomposition.
    """
    def __init__(self, QobjEvo oper, bint sparse=False):
        if oper.dims[0] != oper.dims[1]:
            raise ValueError
        if type(oper(0)) == _data.CSR and not sparse:
            oper = oper.to(Dense)
        self._oper = oper
        self.isconstant = oper.isconstant
        self.size = oper.shape[0]

        if oper.isconstant:
            self._eigvals, self._evecs = eigs(self._oper._call(t), True, True)
        else:
            self._evecs = None
            self._eigvals = None

        self._t = np.nan
        self._inv = None
        self._skew = None
        self._dw_min = -1

    def as_Qobj(self):
        """Make an Qobj or QobjEvo of the eigenvectors."""
        if self.isconstant:
            return Qobj(self.evecs(t), dims=self._oper.dims)
        else:
            return QobjEvo(_eigen_qevo(self._oper))

    cdef void _compute_eigen(self, double t) except *:
        if self._t != t and not self.isconstant:
            self._t = t
            self._inv = None
            self._dw_min = -1
            self._eigvals, self._evecs = eigs(self._oper._call(t), True, True)

    cpdef object diagonal(self, double t):
        """
        Return the diagonal of the diagonalized operation: the eigenvalues.
        """
        self._compute_eigen(t)
        return self._eigvals

    cpdef Data evecs(self, double t):
        """
        Return the eigenstates of the diagonalized operation.
        """
        self._compute_eigen(t)
        return self._evecs

    cpdef Data inv(self, double t):
        """
        Return the eigenstates of the diagonalized operation.
        """
        if self._inv is None:
            self._inv = self.evecs(t).adjoint()
        return self._inv

    cdef Data S_converter(self, double t):
        return _data.kron(self.evecs(t).transpose(), self.inv(t))

    cdef Data S_converter_inverse(self, double t):
        return _data.kron(self.inv(t).transpose(), self.evecs(t))

    cpdef void to_eigbasis(self, double t, Data fock):
        # For Hermitian operator, the inverse of evecs is the adjoint matrix.
        # Blas include A.dag @ B in one operation. We use it if we can so we
        # don't make unneeded copy of evecs.
        cdef Data temp
        if fock.shape[0] == self.size and fock.shape[1] == 1:
            return matmul_var(self.evecs(t), fock, 3, 0)

        elif fock.shape[0] == self.size**2 and fock.shape[1] == 1:
            if type(fock) is Dense and fock.fortran:
                fock = _data.column_unstack_dense(fock, True)
                temp = _data.mul(matmul_var(self.evecs(t), fock, 3, 0),
                                 self.evecs(t))
                fock = _data.column_stack_dense(fock, True)
            else:
                fock = _data.column_unstack(fock)
                temp = _data.mul(matmul_var(self.evecs(t), fock, 3, 0),
                                 self.evecs(t))
            if type(temp) is Dense:
                return _data.column_stack_dense(temp, True)
            return _data.column_stack(temp)

        if fock.shape[0] == self.size and fock.shape[0] == fock.shape[1]:
            return _data.mul(matmul_var(self.evecs(t), fock, 3, 0),
                             self.evecs(t))

        elif fock.shape[0] == self.size**2 and fock.shape[0] == fock.shape[1]:
            temp = self.S_converter_inverse(t)
            return _data.mul(matmul_var(temp, fock, 3, 0), temp)

        raise ValueError

    cpdef void from_eigbasis(self, double t, Data eig):
        cdef Data temp
        if eig.shape[0] == self.size and eig.shape[1] == 1:
            return _data.mul(self.evecs(t), eig)

        elif eig.shape[0] == self.size**2 and eig.shape[1] == 1:
            if type(eig) is Dense and eig.fortran:
                eig = _data.column_unstack_dense(eig, True)
                temp = matmul_var(_data.mul(self.evecs(t), eig),
                                  self.evecs(t), 0, 3)
                eig = _data.column_stack_dense(eig, True)
            else:
                eig = _data.column_unstack(eig)
                temp = matmul_var(_data.mul(self.evecs(t), eig),
                                  self.evecs(t), 0, 3)
            if type(temp) is Dense:
                return _data.column_stack_dense(temp, True)
            return _data.column_stack(temp)

        elif eig.shape[0] == self.size and eig.shape[0] == eig.shape[1]:
            temp = self.evecs(t)
            return matmul_var(_data.mul(temp, eig), temp, 0, 3)

        elif eig.shape[0] == self.size**2 and eig.shape[0] == eig.shape[1]:
            temp = self.S_converter_inverse(t)
            return _data.mul(temp, matmul_var(fock, temp, 0, 3))

        raise ValueError

    cdef double dw_min(self, double t):
        """ dw_min = min(abs(skew[skew != 0]))"""
        self.skew(t)
        return self._dw_min

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double[:, :] skew(self, double t):
        """ skew[i, j] = w[i] - w[j]"""
        cdef size_t i, j
        cdef double dw
        cdef double[:] eigvals
        self._compute_eigen(t)
        if self._dw_min < 0:  # Check if already computed
            eigvals = self._eigvals
            self.skew = np.empty((self.size, self.size))
            self._dw_min = DBL_MAX
            for i in range(0, self.size):
                self._skew[i, i] = 0.
                for i in range(i, self.size):
                    dw = eigvals[i] - eigvals[j]
                    self._skew[i, j] = dw
                    self._skew[j, i] = -dw
                    self._dw_min = fmin(fabs(dw), self._dw_min)
        return self._skew





















@cython.overflowcheck(True)
cdef size_t _mul_checked(size_t a, size_t b) except? -1:
    return a * b


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
