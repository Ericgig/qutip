#cython: language_level=3

from qutip.core.cy.qobjevo cimport QobjEvo
from scipy.linalg.cython_blas cimport zgemm
from qutip.core.data cimport Data, CSR, Dense
import numpy as np
from libc.float cimport DBL_MAX
from qutip.core.data.eigen import eigs
import qutip.core.data as _data


def isDiagonal(M):
    test = M.reshape(-1)[:-1].reshape(M.shape[i]-1, M.shape[j]+1)
    return not np.any(test[:, 1:])


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
    #        Tensorflow has this option in matmul.
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


cdef class _DiagonalizedOperator:
    """
    Diagonalized operator.

    For a QobjEvo operator, return the eigenvalues at ``t`` and manage basis
    transformation to and from the diagonal basis.

    ==> eigenvalues wrapper
    """
    def __init__(self, QobjEvo oper, bint use_sparse=False):
        if oper.shape[0] != oper.shape[1]:
            raise ValueError
        if type(oper(0)) == _data.CSR and not use_sparse:
            oper = oper.to(Dense)
        self._oper = oper.to(Dense)
        self.isconstant = oper.isconstant
        if oper.isconstant:
            self._compute_eigen(0)
        self.size = oper.shape[0]
        self._isherm = False

        self._t = np.nan
        self._evecs = None
        self._inv = None
        self._skew = None

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
            self._inv = _data.inv(self.evecs(t))
        return self._inv

    cdef void _compute_eigen(self, double t) except *:
        # TODO : Is there an eigen solver which can reuse previous evecs?
        if self._t != t and not self.isconstant:
            self._t = t
            self._inv = None
            self._eigvals, self._evecs = eigs(self._oper._call(t), False, True)

    cdef Data S_converter(self, double t):
        return _data.kron(self.evecs(t).transpose(), self.inv(t))

    cdef Data S_converter_inverse(self, double t):
        return _data.kron(self.inv(t).transpose(), self.evecs(t))

    cpdef void to_eigbasis(self, double t, Data fock):
        cdef Data temp
        if fock.shape[0] == self.size and fock.shape[1] == 1:
            return _data.mul(self.inv(t), fock)

        elif fock.shape[0] == self.size**2 and fock.shape[1] == 1:
            fock = _data.column_unstack(fock)
            temp = _data.mul(_data.mul(self.inv(t), fock), self.evecs(t))
            return _data.column_stack(temp)

        elif fock.shape[0] == self.size and fock.shape[0] == fock.shape[1]:
            return _data.mul(_data.mul(self.inv(t), fock), self.evecs(t))

        elif fock.shape[0] == self.size**2 and fock.shape[0] == fock.shape[1]:
            return _data.mul(_data.mul(self.S_converter(t), fock),
                             self.S_converter_inverse(t))

        raise ValueError

    cpdef void from_eigbasis(self, double t, Data eig):
        cdef Data temp
        if eig.shape[0] == self.size and eig.shape[1] == 1:
            return _data.mul(self.evecs(t), eig)

        elif eig.shape[0] == self.size**2 and eig.shape[1] == 1:
            eig = _data.column_unstack(eig)
            temp = _data.mul(_data.mul(self.evecs(t), eig), self.inv(t))
            return _data.column_stack(temp)

        elif eig.shape[0] == self.size and eig.shape[0] == eig.shape[1]:
            return _data.mul(_data.mul(self.evecs(t), eig), self.inv(t))

        elif eig.shape[0] == self.size**2 and eig.shape[0] == eig.shape[1]:
            return _data.mul(_data.mul(self.S_converter_inverse(t), eig),
                             self.S_converter(t))

        raise ValueError


cdef class _DiagonalizedOperatorHermitian(_DiagonalizedOperator):
    def __init__(self, QobjEvo oper, bint use_sparse=False):
        super().__init__(txt, oper, use_sparse)
        self._skew = None
        self._dw_min = -1

    cdef void _compute_eigen(self, double t) except *:
        if self._t != t:
            self._t = t
            self._inv = None
            self._dw_min = -1
            self._eigvals, self._evecs = eigs(self._oper._call(t), True, True)

    cpdef Data inv(self, double t):
        """
        Return the eigenstates of the diagonalized operation.
        """
        if self._inv is None:
            self._inv = self.eigenstates(t).adjoint()
        return self._inv

    cdef double dw_min(self, double t):
        self.skew(t)
        return self._dw_min

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double[:, :] skew(self, double t):
        cdef size_t i, j
        cdef double dw
        cdef double[:] eigvals
        self._compute_eigen(t)
        if self._dw_min < 0:
            eigvals = self._eigvals
            self._dw_min = DBL_MAX
            if self._skew is None:
                self._skew = np.empty((self.size, self.size))
            for i in range(0, self.size):
                self._skew[i * (self.size + 1)] = 0.
                for i in range(i, self.size):
                    dw = eigvals[i] - eigvals[j]
                    self._skew[i * self.size + j] = dw
                    self._skew[j * self.size + i] = -dw
                    self._dw_min = fmin(fabs(dw), self._dw_min)
        return self._skew

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
