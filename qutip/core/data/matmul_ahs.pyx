#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memset

from libc.math cimport sqrt
cdef extern from "<complex>" namespace "std" nogil:
    double complex conj(double complex x)

cimport cython
import numpy as np
cimport numpy as cnp

from qutip.core.data.base cimport idxint
from qutip.core.data.dense cimport Dense
from qutip.core.data.csr cimport CSR
from qutip.core.data.csc cimport CSC

cnp.import_array()


cpdef void mv_ahs_csr(CSR matrix,
                 double complex[:] vec,
                 double complex[:] out,
                 idxint[:] rows):
    """
    Perform the operation
        ``out := a * (matrix @ vector) + out``

    Matrix-vector product between a CSR matrix and a pointer to a contiguous
    array of double complex, adding to and storing the result in `out`.
    Only sums on the rows listed in the rows array
    """
    cdef idxint row, jj, ii, row_start, row_end
    cdef complex dot
    for ii in range(len(rows)):
        row = rows[ii]
        dot = 0
        row_start = matrix.row_index[row]
        row_end = matrix.row_index[row+1]
        for jj in range(row_start, row_end):
            dot += matrix.data[jj]*vec[matrix.col_index[jj]]
        out[row] += dot


cpdef void mv_ahs_csr_dm(CSR matrix,
                   double complex[:] vec,
                   double complex[:] out,
                   idxint[:] rows):
    """
    Perform the operation
        ``out := a * (matrix @ vector) + out``

    Matrix-vector product between a CSR matrix and a pointer to a contiguous
    array of double complex, adding to and storing the result in `out`.
    Both out and vector are 1D representation of hermitian matrix.
    Only sums on the rows listed in the rows array.
    """
    cdef idxint row, row_t, ii, jj, kk, row_start, row_end,
    cdef idxint N = int(sqrt(len(vec)))
    cdef complex dot
    for ii in range(len(rows)):
        row = rows[ii]
        row_t = (row // N) + (row % N) * N
        dot = 0
        row_start = matrix.row_index[row]
        row_end = matrix.row_index[row+1]
        for jj in range(row_start, row_end):
            dot += matrix.data[jj] * vec[matrix.col_index[jj]]
        out[row] += dot
        if row != row_t:
            out[row_t] += conj(dot)


cpdef void mv_ahs_csc(CSC matrix,
                 double complex[:] vec,
                 double complex[:] out,
                 idxint[:] cols):
    """
    Perform the operation
        ``out := a * (matrix @ vector) + out``

    Matrix-vector product between a CSC matrix and a pointer to a contiguous
    array of double complex, adding to and storing the result in `out`.
    Only sums on the cols listed in the cols array
    """
    cdef idxint col, ii, jj, col_start, col_end
    for ii in range(len(cols)):
        col = cols[ii]
        col_start = matrix.col_index[col]
        col_end = matrix.col_index[col+1]
        for jj in range(col_start, col_end):
            out[matrix.row_index[jj]] += matrix.data[jj]*vec[col]


cpdef void mv_pseudo_ahs_csc(CSC matrix,
                             double complex[:] vec,
                             double complex[:] out,
                             double atol,
                             double rtol):
    cdef idxint ii, jj, col, col_start, col_end
    cdef idxint N = int(sqrt(len(vec)))
    cdef complex dot
    cdef double* dvec = <double*><void*> &vec[0]
    cdef double tol, max_prob = 0
    if rtol != 0:
        for ii in range(N*2):
            if max_prob < dvec[ii]:
                max_prob = dvec[ii]
    tol = max_prob * rtol + atol

    for col in range(len(vec)):
        if dvec[col*2] > tol or dvec[col*2+1] > tol:
            col_start = matrix.col_index[col]
            col_end = matrix.col_index[col+1]
            for jj in range(col_start, col_end):
                out[matrix.row_index[jj]] += matrix.data[jj] * vec[col]


cpdef void mv_pseudo_ahs_csc_dm(CSC matrix,
                                double complex[:] vec,
                                double complex[:] out,
                                double atol,
                                double rtol):
    cdef idxint ii, jj, col, col_start, col_end
    cdef idxint N = int(sqrt(len(vec)))
    cdef complex dot
    cdef double* dvec = <double*><void*> &vec[0]
    cdef double tol, max_prob = 0
    if rtol != 0:
        for ii in range(N):
            if max_prob < dvec[2*(N+1)*ii]:
                max_prob = dvec[2*(N+1)*ii]
    tol = max_prob * rtol + atol

    for col in range(len(vec)):
        if dvec[col*2] > tol or dvec[col*2+1] > tol:
            col_start = matrix.col_index[col]
            col_end = matrix.col_index[col+1]
            for jj in range(col_start, col_end):
                out[matrix.row_index[jj]] += matrix.data[jj] * vec[col]
