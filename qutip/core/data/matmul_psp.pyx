#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memset

from libc.math cimport sqrt
cdef extern from "<complex>" namespace "std" nogil:
    double complex conj(double complex x)
    double real(double complex x)
    double imag(double complex x)

cimport cython
import numpy as np
cimport numpy as cnp

from qutip.core.data.base cimport idxint
from qutip.core.data.dense cimport Dense
from qutip.core.data.csr cimport CSR
from qutip.core.data.csc cimport CSC
from qutip.core.data cimport dense, csr, csc, Data

cnp.import_array()


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
            "left matrix must be a have a shape = [N,1]"
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


cpdef Dense matmul_psp_ket_csc_dense(CSC left, Dense right,
                                    double atol, double rtol,
                                    double complex a=1, Dense out=None):
    cdef idxint ii, row, col, col_start, col_end
    cdef idxint N = right.shape[0]
    cdef complex dot
    cdef double *dvec = <double*><void*> right.data
    cdef double tol, max_prob = 0

    if out is None:
        out = dense.zeros(left.shape[0], right.shape[1], right.fortran)

    if rtol != 0:
        for ii in range(N*2):
            if max_prob < dvec[ii]:
                max_prob = dvec[ii]
    tol = max_prob * rtol + atol

    for col in range(left.shape[1]):
        if dvec[col*2] > tol or dvec[col*2+1] > tol:
            col_start = left.col_index[col]
            col_end = left.col_index[col+1]
            for row in range(col_start, col_end):
                out.data[left.row_index[row]] += a * left.data[row] * right.data[col]
    return out


cpdef Dense matmul_psp_ket_dense(Dense left, Dense right,
                                    double atol, double rtol,
                                    double complex a=1, Dense out=None):
    cdef idxint ii, row, col, col_start, col_end, row_stride, col_stride
    cdef idxint N = right.shape[0]
    cdef double tol, max_prob = 0
    cdef double *dvec = <double*><void*> right.data

    if out is None:
        out = dense.zeros(left.shape[0], right.shape[1], right.fortran)

    if rtol != 0:
        for ii in range(N*2):
            if max_prob < dvec[ii]:
                max_prob = dvec[ii]
    tol = max_prob * rtol + atol

    row_stride = 1 if left.fortran else left.shape[1]
    col_stride = left.shape[0] if left.fortran else 1

    if left.fortran:
        for col in range(N):
            if dvec[col*2] > tol or dvec[col*2+1] > tol:
                for row in range(N):
                    out.data[row] += a * left.data[row + col * col_stride] * \
                                    right.data[col]
    else:
        # Raise efficiency warnings?
        for col in range(N):
            if real(right.data[col]) > tol or imag(right.data[col]) > tol:
                for row in range(N):
                    out.data[row] += a * left.data[row * row_stride + col] * \
                                    right.data[col]

    return out


cpdef Dense matmul_psp_dm_csc_dense(CSC left, Dense right,
                                    double atol, double rtol,
                                    double complex a=1, Dense out=None):
    cdef idxint ii, row, col, col_start, col_end, col_i, col_j
    cdef idxint N = int(sqrt(left.shape[0]))
    cdef double* dvec = <double*><void*> right.data
    cdef double tol, max_prob = 0

    if out is None:
        out = dense.zeros(left.shape[0], right.shape[1], right.fortran)

    if rtol != 0:
        for ii in range(N):
            if max_prob < dvec[2*(N+1)*ii]:
                max_prob = dvec[2*(N+1)*ii]
    tol = max_prob * rtol + atol

    for col in range(left.shape[1]):
        if dvec[col*2] > tol or dvec[col*2+1] > tol:
            col_start = left.col_index[col]
            col_end = left.col_index[col+1]
            for row in range(col_start, col_end):
                out.data[left.row_index[row]] += a * left.data[row] * right.data[col]
    return out


cpdef Dense matmul_psp_dm_dense(Dense left, Dense right,
                                    double atol, double rtol,
                                    double complex a=1, Dense out=None):
    """
    Perform the operation
        ``out := a * (matrix @ vector) + out``

    Matrix-vector product between a CSR matrix and a pointer to a contiguous
    array of double complex, adding to and storing the result in `out`.
    Both out and vector are 1D representation of hermitian matrix.
    Only sums on the rows listed in the rows array.
    """
    _check_shape(left, right, out)
    cdef idxint ii, row, row_i, row_j, row_stride
    cdef idxint col, row_t, col_i, col_j, col_stride
    cdef idxint N = int(sqrt(left.shape[1]))
    cdef double* dvec = <double*><void*> right.data
    cdef double tol, max_prob = 0

    if out is None:
        out = dense.zeros(left.shape[0], right.shape[1], right.fortran)

    if rtol != 0:
        for ii in range(N):
            if max_prob < dvec[2*(N+1)*ii]:
                max_prob = dvec[2*(N+1)*ii]
    tol = max_prob * rtol + atol

    row_stride = 1 if left.fortran else left.shape[1]
    col_stride = left.shape[0] if left.fortran else 1

    if left.fortran:
        for col in range(N**2):
            if dvec[col*2] > tol or dvec[col*2+1] > tol:
                for row in range(N**2):
                    out.data[row] += a * left.data[row + col * col_stride] * right.data[col]
    else:
        for col in range(N**2):
            if dvec[col*2] > tol or dvec[col*2+1] > tol:
                for row in range(N**2):
                    out.data[row] += a * left.data[row * row_stride + col] * right.data[col]
    return out
