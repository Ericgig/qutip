

from qutip.core.cy.qobjevo cimport QobjEvo
from scipy.linalg.cython_lapack cimport zheevr, zgeev
from qutip.core.data cimport Data, CSR, Dense, dense, csr, idxint
import numpy as np
cimport numpy as cnp
from qutip.core.data.eigen import eigs
import qutip.core.data as _data

def isDiagonal(M):
    test = M.reshape(-1)[:-1].reshape(M.shape[i]-1, M.shape[j]+1)
    return not np.any(test[:, 1:])


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


cdef class EigenH():
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


cdef class _DiagonalizedOperatorDense:
    """
    Diagonalized time dependent operator.
    """
    cdef:
        double t
        bint isconstant, isherm
        QobjEvo oper
        int nrows
        double[:] eigvals
        Dense evecs, temp

    def __init__(self, QobjEvo oper, bint isherm):
        if oper.shape[0] != oper.shape[1]:
            raise ValueError
        self.oper = oper.to(Dense)
        self.isconstant = oper.isconstant
        if oper.isconstant:
            self._compute_eigen(0)
        self.nrows = oper.shape[0]
        self.t = np.nan
        self.isherm = isherm

        self.evecs = dense.zeros(self.nrows, self.nrows, fortran=True)
        self.temp = dense.zeros(self.nrows, self.nrows, fortran=True)

    cpdef double[:] diagonal(self, double t):
        """
        Return the diagonal of the diagonalized operation: the eigenvalues.
        """
        if self.isconstant or t == self.t:
            return self.eigvals
        self._compute_eigen(t)
        return self.eigvals

    cpdef Dense eigenstates(self, double t):
        """
        Return the eigenstates of the diagonalized operation.
        """
        if self.isconstant or t == self.t:
            return self.evecs
        self._compute_eigen(t)
        return self.evecs

    cdef void _compute_eigen(self, double t) except *:
        self.t = t
        op_data = self.oper._call(t)
        # This lose to preallocating buffer and calling zheevr directly
        # Also the `eigh` output are arrays which limits the non-Dense type...
        # TODO: Should we fix this as dense and use zheevr?
        # Or should we have `eigs`'s `eigenvectors` output as a Data layer?
        self.eigvals, evecs = eigs(op_data, True, True)
        self.evecs = Dense(evecs, copy=True)

    cpdef Dense _to_eigbasis(self, Dense fock, Dense out=None):
        # evecs is usually dense or diagonal
        # Use ZGEMM instead of dense.matmul since it can do A.dag @ B in one
        # operation without making copy of the array.
        cdef size_t nrows = evecs.shape[0]
        if evecs.shape[0] != evecs.shape[1]:
            raise ValueError
        if fock.shape[0] != evecs.shape[0] or fock.shape[1] != evecs.shape[1]:
            raise ValueError
        if out is None:
            out = data.dense.empty(evecs.shape[0], evecs.shape[1], True)
        elif out.shape[0] != evecs.shape[0] or out.shape[1] != evecs.shape[1]:
            raise ValueError

        # Z.dag @ rho @ Z
        ZGEMM(fock.data, self.evecs.data, nrows, nrows, nrows, nrows,
              not fock.fortran, 0, 1., 0., self.temp.data)
        ZGEMM(self.evecs.data, self.temp.data, nrows, nrows, nrows, nrows,
              2, 0, 1., 0., out.data)
        return out

    cpdef Dense _to_fockbasis(self, Dense eigen, Dense out=None):
        # evecs is usually dense or diagonal
        # Use ZGEMM instead of dense.matmul since it can do A.dag @ B in one
        # operation without making copy of the array.
        cdef size_t nrows = evecs.shape[0]
        if evecs.shape[0] != evecs.shape[1]:
            raise ValueError
        if eigen.shape[0] != evecs.shape[0] or eigen.shape[1] != evecs.shape[1]:
            raise ValueError
        if out is None:
            out = data.dense.empty(evecs.shape[0], evecs.shape[0], True)
        elif out.shape[0] != evecs.shape[0] or out.shape[1] != evecs.shape[0]:
            raise ValueError

        # Z @ rho @ Z.dag
        ZGEMM(eigen.data, self.evecs.data, nrows, nrows, nrows, nrows,
              not eigen.fortran, 2, 1., 0., self.temp.data)
        ZGEMM(self.evecs.data, self.temp.data, nrows, nrows, nrows, nrows,
              0, 0, 1., 1., out.data)
        return out

    cpdef Data _superop_to_eigenbasis(self, Data fock):
        # kron only available as CSR...
        cdef size_t nrows = self.evecs.shape[0]
        cdef _data.Data S
        S = _data.kron(self.evecs.transpose(), self.evecs.adjoint())
        return _data.matmul(S, _data.matmul(fock, S.adjoint()))

    cpdef Data _superop_to_fockbasis(self, Data eig):
        # kron only available as CSR...
        cdef size_t nrows = self.evecs.shape[0]
        cdef _data.Data S
        S = _data.kron(self.evecs.transpose(), self.evecs.adjoint())
        return _data.matmul(S.adjoint(), _data.matmul(eig, S))
