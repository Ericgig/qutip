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

from qutip.core.data.base cimport idxint, Data
from qutip.core.data.dense cimport Dense
from qutip.core.data.csr cimport CSR
from qutip.core.data cimport csr

cnp.import_array()


cdef mv_ahs_csr(CSR matrix,
                complex[:] vec,
                complex[:] out,
                idxint[:] rows) nogil:
    """
    Perform the operation
        ``out := a * (matrix @ vector) + out``

    Matrix-vector product between a CSR matrix and a pointer to a contiguous
    array of double complex, adding to and storing the result in `out`.
    Only sums on the rows listed in the rows array
    """
    cdef complex[:] data = csr.data
    cdef idxint row, jj, ii, row_start, row_end
    cdef complex dot
    for ii in range(len(rows)):
        row = rows[ii]
        dot = 0
        row_start = csr.row_index[row]
        row_end = csr.row_index[row+1]
        for jj in range(row_start, row_end):
            dot += data[jj]*vec[csr.col_index[jj]]
        out[row] += dot


cdef mv_ahs_csr_dm(CSR matrix,
                   complex[:] vec,
                   complex[:] out,
                   idxint[:] rows) nogil:
    """
    Perform the operation
        ``out := a * (matrix @ vector) + out``

    Matrix-vector product between a CSR matrix and a pointer to a contiguous
    array of double complex, adding to and storing the result in `out`.
    Both out and vector are 1D representation of hermitian matrix.
    Only sums on the rows listed in the rows array.
    """
    cdef complex[:] data = csr.data
    cdef idxint row, row_t, ii, jj, kk, row_start, row_end,
    cdef idxint N = int(sqrt(len(vec)))
    cdef complex dot
    for ii in range(len(rows)):
        row = rows[ii]
        row_t = (row // N) + (row % N) * N
        dot = 0
        row_start = csr.row_index[row]
        row_end = csr.row_index[row+1]
        for jj in range(row_start, row_end):
            dot += data[jj] * vec[csr.col_index[jj]]
        out[row] += dot
        if row != row_t:
            out[row_t] += conj(dot)


cdef mv_ahs_csc(CSC matrix,
                complex[:] vec,
                complex[:] out,
                idxint[:] cols) nogil:
    """
    Perform the operation
        ``out := a * (matrix @ vector) + out``

    Matrix-vector product between a CSC matrix and a pointer to a contiguous
    array of double complex, adding to and storing the result in `out`.
    Only sums on the cols listed in the cols array
    """
    cdef complex[:] data = csc.data
    cdef idxint col, ii, jj, col_start, col_end
    for ii in range(len(cols)):
        col = cols[ii]
        col_start = csc.col_index[col]
        col_end = csc.col_index[col+1]
        for jj in range(col_start, col_end):
            out[csc.row_index[jj]] += a*data[jj]*vec[col]


cdef void mv_pseudo_ahs_csc(CSC matrix,
                            double complex *vector,
                            double complex *out,
                            double atol,
                            double rtol) nogil:
    cdef complex[:] data = csr.data
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
            col_start = csc.col_index[col]
            col_end = csc.col_index[col+1]
            for jj in range(col_start, col_end):
                out[csc.row_index[jj]] += a * data[jj] * vec[col]


cdef void mv_pseudo_ahs_csc_dm(CSC matrix,
                               double complex *vector,
                               double complex *out,
                               double atol,
                               double rtol) nogil:
    cdef complex[:] data = csr.data
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
            col_start = csc.col_index[col]
            col_end = csc.col_index[col+1]
            for jj in range(col_start, col_end):
                out[csc.row_index[jj]] += a * data[jj] * vec[col]
