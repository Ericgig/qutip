#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdvision=True

from qutip.core.data.base cimport idxint
from qutip.core.data cimport Dense, Data, CSR, dense
from qutip.core.cy._element cimport _BaseElement
from qutip.core.cy.qobjevo cimport QobjEvo
from scipy.linalg cimport cython_blas as blas
from libc.math cimport sqrt

cdef extern from "<complex>" namespace "std" nogil:
    double complex conj(double complex x)

__all__ = ["make_herm_rhs"]

cdef void _check_shape(Data left, Data right, Data out=None) nogil except *:
    if left.shape[1] != right.shape[0]:
        raise ValueError(
            "incompatible matrix shapes "
            + str(left.shape)
            + " and "
            + str(right.shape)
        )
    if right.shape[1] != 1:
        raise ValueError(
            "invalid right matrix shape, must be a operator-ket"
        )
    if (
        out is not None
        and out.shape[0] != left.shape[0]
        and out.shape[1] != right.shape[1]
    ):
        raise ValueError(
            "incompatible output shape, got "
            + str(out.shape)
            + " but needed "
            + str((left.shape[0], right.shape[1]))
        )


cpdef Dense herm_matmul_csr_dense_dense(CSR left, Dense right,
                                        double complex scale=1,
                                        Dense out=None):
    """
    Perform the operation
        ``out := scale * (left @ right) + out``
    where `left`, `right` and `out` are matrices.  `scale` is a complex scalar,
    defaulting to 1.
    `left` and `right` must be chossen so `out` is hemitian.
    `left` and 'out' must be vectorized operator: `shape = (N**2, 1)`
    Made to be used in :func:`mesolve`.
    """
    _check_shape(left, right, out)
    cdef size_t N = <size_t> sqrt(right.shape[0])
    if right.shape[0] != N*N:
        raise ValueError(
            "invalid right matrix shape, must be a square operator-ket"
        )
    if out is None:
        out = dense.zeros(left.shape[0], right.shape[1], right.fortran)
    cdef idxint row, ptr, idx_r, idx_out, dm_row, dm_col
    cdef idxint nrows=left.shape[0], ncols=right.shape[1]
    cdef double complex val
    # right shape (N*N, 1) is interpreted as (N, N) and we loop only on the
    # upper triangular part.
    for dm_row in range(N):
        row = dm_row * (N+1)
        val = 0
        for ptr in range(left.row_index[row], left.row_index[row+1]):
            # diagonal part
            val += left.data[ptr] * right.data[left.col_index[ptr]]
        out.data[row] += scale * val

        for dm_col in range(dm_row+1, N):
            # upper triangular part
            row = dm_row*N + dm_col
            val = 0
            for ptr in range(left.row_index[row], left.row_index[row+1]):
                val += left.data[ptr] * right.data[left.col_index[ptr]]
            out.data[row] += scale * val
            out.data[dm_col*N + dm_row] += conj(scale * val)
    return out

cpdef Dense herm_matmul_dense(Dense left, Dense right,
                              double complex scale=1, Dense out=None):
    """
    Perform the operation
        ``out := scale * (left @ right) + out``
    where `left`, `right` and `out` are matrices.  `scale` is a complex scalar,
    defaulting to 1.
    `left` and `right` must be chossen so `out` is hemitian.
    `left` and 'out' must be vectorized operator: `shape = (N**2, 1)`
    Made to be used in :func:`mesolve`.
    """
    _check_shape(left, right, out)
    cdef int N = <size_t> sqrt(right.shape[0]), N2 = right.shape[0]
    if right.shape[0] != N*N:
        raise ValueError(
            "invalid right matrix shape, must be a square operator-ket"
        )
    if out is None:
        out = dense.zeros(left.shape[0], right.shape[1], right.fortran)
    cdef double complex val
    cdef int dm_row, dm_col, row_stride, col_stride, one=1
    row_stride = 1 if left.fortran else left.shape[1]
    col_stride = left.shape[0] if left.fortran else 1
    # right shape (N*N, 1) is interpreted as (N, N) and we loop only on the
    # upper triangular part.
    for dm_row in range(N):
        out.data[dm_row * (N+1)] += scale * blas.zdotu(&N2,
            &left.data[dm_row * row_stride * (N+1)], &col_stride,
            right.data, &one)
        for dm_col in range(dm_row+1, N):
            val = blas.zdotu(&N2,
                &left.data[(dm_row * N + dm_col) * row_stride], &col_stride,
                right.data, &one)
            out.data[dm_row*N + dm_col] += scale * val
            out.data[dm_col*N + dm_row] += conj(scale * val)
    return out


cdef class _HermProdElement(_BaseElement):
    cdef _BaseElement original

    def __init__(self, _BaseElement original):
        self.original= original

    cpdef Data data(self, double t):
        return self.original.data(t)

    cpdef object qobj(self, double t):
        return self.original.qobj(t)

    cpdef double complex coeff(self, double t) except *:
        return self.original.coeff(t)

    cdef Data matmul_data_t(_HermProdElement self, double t, Data state, Data out=None):
        if type(state) is not Dense or type(out) is not Dense:
            raise TypeError("Herm matmul support only dense state")
        cdef Data left = self.data(t)
        if type(left) is Dense:
            out = herm_matmul_dense(left, state, self.coeff(t), out)
        elif type(left) is CSR:
            out = herm_matmul_csr_dense_dense(left, state, self.coeff(t), out)
        else:
            raise NotImplementedError
        return out

def make_herm_rhs(QobjEvo system):
    """ Return a copy of the QobjEvo with the matmul overwitten """
    system = system.copy()
    system.elements = [_HermProdElement(part) for part in system.elements]
    return system
