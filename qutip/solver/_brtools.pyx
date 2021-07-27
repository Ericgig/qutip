#cython: language_level=3
from libc.math cimport fabs, fmin
from libc.float cimport DBL_MAX

cimport numpy as cnp
import numpy as np

cimport cython

from qutip.core.cy.qobjevo cimport QobjEvo
from qutip.core.cy.coefficient cimport Coefficient
from qutip.core.data cimport Data, Dense, idxint
import qutip.core.data as _data
from qutip.core import Qobj
from scipy.linalg.cython_blas cimport zgemm
from qutip.core.data.eigen import eigs


cnp.import_array()

cdef extern from "<complex>" namespace "std" nogil:
    double complex conj(double complex x)
    double         real(double complex x)
    double cabs   "abs" (double complex x)
    double complex sqrt(double complex x)


__all__ = ['SpectraCoefficient', '_EigenBasisTransform']


cdef class SpectraCoefficient(Coefficient):
    """
    Change a Coefficient with `t` dependence to one with `w` dependence to use
    in bloch redfield tensor. Allow array based coefficients to be used as
    spectrum function.
    If 2 coefficients are passed, the first one is the frequence responce and
    the second is the time responce.


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


@cython.overflowcheck(True)
cdef size_t _mul_checked(size_t a, size_t b) except? -1:
    return a * b


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
    if trans == 0:
        return b'N'
    elif trans == 1:
        return b'T'
    elif trans == 3:
        return b'C'


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
    cdef int size
    cdef double complex beta
    cdef char left_code, right_code
    if not (
        type(left) is Dense
        and type(right) is Dense
        and (type(out) is None or type(out) is Dense)
        ) or not (
        left.shape[0] == left.shape[1]
        and left.shape[0] == right.shape[0]
        and left.shape[0] == right.shape[1]
    ):
        left = _apply_trans(left, transleft)
        right = _apply_trans(right, transright)
        return _data.add(out, _data.matmul(left, right), alpha)

    if out is None:
        out = _data.dense.empty(left.shape[0], right.shape[1], right.fortran)

    transleft &= left.fortran
    transright &= right.fortran
    size = left.shape[0]

    if transleft + transright == 5:
        out = matmul_var(left, right, transleft, transright, alpha, out)
        return out.conj()

    unavail = 3 - out.fortran
    any_unavail = (transleft == unavail or transright == unavail)
    if any_unavail:
        transleft &= 1
        transright &= 1

    if out.fortran == any_unavail:
        beta = 1
        left_code = _fetch_trans_code(transleft)
        right_code = _fetch_trans_code(transright)
        zgemm(&left_code, &right_code, &size, &size, &size,
              &alpha, (<Dense> left).data, &size, (<Dense> right).data, &size,
              &beta, (<Dense> out).data, &size)
    else:
        transleft &= 1
        transright &= 1
        beta = 1
        left_code = _fetch_trans_code(transleft)
        right_code = _fetch_trans_code(transright)
        zgemm(&right_code, &left_code, &size, &size, &size,
              &alpha, (<Dense> right).data, &size, (<Dense> left).data, &size,
              &beta, (<Dense> out).data, &size)
    if any_unavail:
        out.fortran = not out.fortran
    return out


class _eigen_qevo:
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
            self._eigvals, self._evecs = eigs(self._oper._call(0), True, True)
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
            return Qobj(self.evecs(0), dims=self._oper.dims)
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

    cpdef Data to_eigbasis(self, double t, Data fock):
        # For Hermitian operator, the inverse of evecs is the adjoint matrix.
        # Blas include A.dag @ B in one operation. We use it if we can so we
        # don't make unneeded copy of evecs.
        cdef Data temp
        if fock.shape[0] == self.size and fock.shape[1] == 1:
            return matmul_var(self.evecs(t), fock, 3, 0)

        elif fock.shape[0] == self.size**2 and fock.shape[1] == 1:
            if type(fock) is Dense and fock.fortran:
                fock = _data.column_unstack_dense(fock, True)
                temp = _data.matmul(matmul_var(self.evecs(t), fock, 3, 0),
                                    self.evecs(t))
                fock = _data.column_stack_dense(fock, True)
            else:
                fock = _data.column_unstack(fock)
                temp = _data.matmul(matmul_var(self.evecs(t), fock, 3, 0),
                                    self.evecs(t))
            if type(temp) is Dense:
                return _data.column_stack_dense(temp, True)
            return _data.column_stack(temp)

        if fock.shape[0] == self.size and fock.shape[0] == fock.shape[1]:
            return _data.matmul(matmul_var(self.evecs(t), fock, 3, 0),
                                self.evecs(t))

        elif fock.shape[0] == self.size**2 and fock.shape[0] == fock.shape[1]:
            temp = self.S_converter_inverse(t)
            return _data.matmul(matmul_var(temp, fock, 3, 0), temp)

        raise ValueError

    cpdef Data from_eigbasis(self, double t, Data eig):
        cdef Data temp
        if eig.shape[0] == self.size and eig.shape[1] == 1:
            return _data.matmul(self.evecs(t), eig)

        elif eig.shape[0] == self.size**2 and eig.shape[1] == 1:
            if type(eig) is Dense and eig.fortran:
                eig = _data.column_unstack_dense(eig, True)
                temp = matmul_var(_data.matmul(self.evecs(t), eig),
                                  self.evecs(t), 0, 3)
                eig = _data.column_stack_dense(eig, True)
            else:
                eig = _data.column_unstack(eig)
                temp = matmul_var(_data.matmul(self.evecs(t), eig),
                                  self.evecs(t), 0, 3)
            if type(temp) is Dense:
                return _data.column_stack_dense(temp, True)
            return _data.column_stack(temp)

        elif eig.shape[0] == self.size and eig.shape[0] == eig.shape[1]:
            temp = self.evecs(t)
            return matmul_var(_data.matmul(temp, eig), temp, 0, 3)

        elif eig.shape[0] == self.size**2 and eig.shape[0] == eig.shape[1]:
            temp = self.S_converter_inverse(t)
            return _data.matmul(temp, matmul_var(eig, temp, 0, 3))

        raise ValueError

    cdef double dw_min(self, double t):
        """ dw_min = min(abs(skew[skew != 0]))"""
        self.skew(t)
        return self._dw_min

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef object skew(self, double t):
        """ skew[i, j] = w[i] - w[j]"""
        cdef size_t i, j
        cdef double dw
        cdef double[:] eigvals
        self._compute_eigen(t)
        if self._dw_min < 0:  # Check if already computed
            eigvals = self._eigvals
            self._skew = np.empty((self.size, self.size))
            self._dw_min = DBL_MAX
            for i in range(0, self.size):
                self._skew[i, i] = 0.
                for j in range(i, self.size):
                    dw = eigvals[i] - eigvals[j]
                    self._skew[i, j] = dw
                    self._skew[j, i] = -dw
                    self._dw_min = fmin(fabs(dw), self._dw_min)
        return self._skew







"""
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
"""
