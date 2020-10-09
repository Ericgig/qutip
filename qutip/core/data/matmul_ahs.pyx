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
from qutip.core.data.base import idxint_dtype

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


cpdef Dense matmul_trunc_ket_csr_dense(CSR left, Dense right, idxint[::1] used_idx,
                                       double complex a=1, Dense out=None):
    """
    Perform the operation
        ``out := a * (left @ right) + out``

    Matrix-vector product between a CSR matrix and a pointer to a contiguous
    array of double complex, adding to and storing the result in `out`.
    Only sums on the rows listed in the rows array
    """
    _check_shape(left, right, out)
    cdef idxint row, col_idx, row_idx, row_start, row_end
    cdef double complex dot

    if out is None:
        out = dense.zeros(left.shape[0], right.shape[1], right.fortran)

    for row_idx in range(len(used_idx)):
        row = used_idx[row_idx]
        dot = 0
        row_start = left.row_index[row]
        row_end = left.row_index[row+1]
        for col_idx in range(row_start, row_end):
            dot += left.data[col_idx] * right.data[left.col_index[col_idx]]
        out.data[row] += a * dot
    return out


cpdef Dense matmul_trunc_dm_csr_dense(CSR left, Dense right, idxint[::1] used_idx,
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
    cdef idxint row, row_t, row_i, row_j, idx_len = len(used_idx)
    cdef idxint col_idx, row_start, row_end,
    cdef idxint N = int(sqrt(left.shape[1]))
    cdef complex dot

    if out is None:
        out = dense.zeros(left.shape[0], right.shape[1], right.fortran)

    for row_i in range(idx_len):
        for row_j in range(row_i, idx_len):
            row = used_idx[row_i] * N + used_idx[row_j]
            row_t = (row // N) + (row % N) * N
            dot = 0
            row_start = left.row_index[row]
            row_end = left.row_index[row+1]
            for col_idx in range(row_start, row_end):
                dot += left.data[col_idx] * right.data[left.col_index[col_idx]]
            out.data[row] += a * dot
            if row != row_t:
                out.data[row_t] += conj(a * dot)
    return out


cpdef Dense matmul_trunc_ket_csc_dense(CSC left, Dense right, idxint[::1] used_idx,
                                       double complex a=1, Dense out=None):
    """
    Perform the operation
        ``out := a * (left @ right) + out``

    Matrix-vector product between a CSR matrix and a pointer to a contiguous
    array of double complex, adding to and storing the result in `out`.
    Only sums on the rows listed in the rows array
    """
    _check_shape(left, right, out)
    cdef idxint col, col_idx, row_idx, col_start, col_end

    if out is None:
        out = dense.zeros(left.shape[0], right.shape[1], right.fortran)

    for col_idx in range(len(used_idx)):
        col = used_idx[col_idx]
        col_start = left.col_index[col]
        col_end = left.col_index[col+1]
        for row_idx in range(col_start, col_end):
            out.data[left.row_index[row_idx]] += a * left.data[row_idx] * right.data[col]
    return out


cpdef Dense matmul_trunc_dm_csc_dense(CSC left, Dense right, idxint[::1] used_idx,
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
    cdef idxint row, idx_len = len(used_idx)
    cdef idxint col, col_t, col_i, col_j, col_idx, row_start, row_end,
    cdef idxint N = int(sqrt(left.shape[1]))

    if out is None:
        out = dense.zeros(left.shape[0], right.shape[1], right.fortran)

    for col_i in range(idx_len):
        for col_j in range(idx_len):
            col = used_idx[col_i] * N + used_idx[col_j]
            col_t = (col // N) + (col % N) * N
            col_start = left.col_index[col]
            col_end = left.col_index[col+1]
            for col_idx in range(col_start, col_end):
                out.data[left.row_index[col_idx]] += a * left.data[col_idx] * right.data[col] #\
                                                   #* 1 if col == col_t else 2
    return out


cpdef Dense matmul_trunc_ket_dense(Dense left, Dense right, idxint[::1] used_idx,
                                   double complex a=1, Dense out=None):
    """
    Perform the operation
        ``out := a * (left @ right) + out``

    Matrix-vector product between a CSR matrix and a pointer to a contiguous
    array of double complex, adding to and storing the result in `out`.
    Only sums on the rows listed in the rows array
    """
    _check_shape(left, right, out)
    cdef idxint col, row, row_stride, col_stride, idx_len = len(used_idx)
    cdef idxint

    if out is None:
        out = dense.zeros(left.shape[0], right.shape[1], right.fortran)

    row_stride = 1 if left.fortran else left.shape[1]
    col_stride = left.shape[0] if left.fortran else 1

    for col_idx in range(idx_len):
        for row_idx in range(idx_len):
            row = used_idx[row_idx]
            col = used_idx[col_idx]
            out.data[row] += a * left.data[row * row_stride + col * col_stride] * right.data[col]

    return out


cpdef Dense matmul_trunc_dm_dense(Dense left, Dense right, idxint[::1] used_idx,
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
    cdef idxint row, row_i, row_j, row_stride, idx_len = len(used_idx)
    cdef idxint col, row_t, col_i, col_j, col_stride
    cdef idxint N = int(sqrt(left.shape[1]))

    if out is None:
        out = dense.zeros(left.shape[0], right.shape[1], right.fortran)

    row_stride = 1 if left.fortran else left.shape[1]
    col_stride = left.shape[0] if left.fortran else 1

    for row_i in range(idx_len):
        for row_j in range(row_i, idx_len):
            row = used_idx[row_i] * N + used_idx[row_j]
            row_t = (row // N) + (row % N) * N
            for col_i in range(idx_len):
                for col_j in range(idx_len):
                    col = used_idx[col_i] * N + used_idx[col_j]
                    out.data[row] += a * left.data[row * row_stride + col * col_stride] * right.data[col]
            if row_t != row:
                out.data[row_t] = conj(out.data[row])
    return out


cpdef idxint[::1] get_idx_ket(Dense state, double atol, double rtol, int pad=0):
    cdef double tol, max_prob
    cdef idxint ii, N=state.shape[0],
    cdef idxint left_pad=0, last=-1
    cdef list idx = []
    if rtol != 0:
        for ii in range(N):
            if max_prob < real(state.data[ii]):
                max_prob = real(state.data[ii])
            if max_prob < imag(state.data[ii]):
                max_prob = imag(state.data[ii])
    tol = max_prob * rtol + atol

    for ii in range(state.shape[0]):
        if real(state.data[ii]) > tol or imag(state.data[ii]) > tol:
            for jj in range(max(last+1, ii-pad), ii):
                idx.append(jj)
            idx.append(ii)
            last = ii
            left_pad = pad
        elif left_pad:
            idx.append(ii)
            last = ii
            left_pad -= 1
    return np.array(idx, dtype=idxint_dtype)


cpdef idxint[::1] get_idx_dm(Dense state, double atol, double rtol, int pad=0):
    cdef double tol, max_prob
    cdef idxint ii, N=state.shape[0]
    cdef idxint left_pad=0, last=-1
    cdef list idx = []
    if rtol != 0:
        for ii in range(N):
            if max_prob < real(state.data[ii*(N+1)]):
                max_prob = real(state.data[ii*(N+1)])
    tol = max_prob * rtol + atol

    for ii in range(state.shape[0]):
        if real(state.data[ii*(N+1)]) > tol:
            for jj in range(max(last+1, ii-pad), ii):
                idx.append(jj)
            idx.append(ii)
            last = ii
            left_pad = pad
        elif left_pad:
            idx.append(ii)
            last = ii
            left_pad -= 1
    return np.array(idx, dtype=idxint_dtype)
