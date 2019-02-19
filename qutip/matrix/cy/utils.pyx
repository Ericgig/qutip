#!python
#cython: language_level=3
cimport cython
from libc.math cimport fabs
from qutip.cy.complex_math cimport real, imag
from libcpp cimport bool
cimport numpy as cnp
import numpy as np

cnp.import_array()
cdef extern from "numpy/arrayobject.h" nogil:
    void PyArray_ENABLEFLAGS(cnp.ndarray arr, int flags)
    void PyDataMem_FREE(void * ptr)
    void PyDataMem_RENEW(void * ptr, size_t size)
    void PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyDataMem_NEW(size_t size)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bool cy_tidyup(complex[::1] data, double atol, unsigned int nnz):
    """
    Performs an in-place tidyup of CSR matrix data
    """
    cdef size_t kk
    cdef double re, im
    cdef bool re_flag, im_flag, out_flag = 0
    for kk in range(nnz):
        re_flag = 0
        im_flag = 0
        re = real(data[kk])
        im = imag(data[kk])
        if fabs(re) < atol:
            re = 0
            re_flag = 1
        if fabs(im) < atol:
            im = 0
            im_flag = 1

        if re_flag or im_flag:
            data[kk] = re + 1j*im

        if re_flag and im_flag:
            out_flag = 1
    return out_flag

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef cnp.ndarray[int, ndim=2, mode='c'] select(int[::1] sel, int[::1] dims, int M):
    """
    Private function finding selected components
    """
    cdef size_t ii, jj, kk
    cdef int _sel, _prd
    cdef cnp.ndarray[int, ndim=2, mode='c'] ilist = np.zeros((M, dims.shape[0]), dtype=np.int32)
    for jj in range(sel.shape[0]):
        _sel =  sel[jj]
        _prd = 1
        for kk in range(jj+1,sel.shape[0]):
            _prd *= dims[sel[kk]]
        for ii in range(M):
            #ilist[ii, _sel] = <int>(trunc(ii / _prd) % dims[_sel])
            ilist[ii, _sel] = ((ii // _prd) % dims[_sel])
    return ilist


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def index_permute(int [::1] idx_arr,
                     int [::1] dims,
                     int [::1] order):

    cdef int ndims = dims.shape[0]
    cdef int ii, n, dim, idx, orderr

    #the fastest way to allocate memory for a temporary array
    cdef int * multi_idx = <int*> PyDataMem_NEW(sizeof(int) * ndims)

    try:
        for ii from 0 <= ii < idx_arr.shape[0]:
            idx = idx_arr[ii]

            #First, decompose long index into multi-index
            for n from ndims > n >= 0:
                dim = dims[n]
                multi_idx[n] = idx % dim
                idx = idx // dim

            #Finally, assemble new long index from reordered multi-index
            dim = 1
            idx = 0
            for n from ndims > n >= 0:
                orderr = order[n]
                idx += multi_idx[orderr] * dim
                dim *= dims[orderr]

            idx_arr[ii] = idx
    finally:
        PyDataMem_FREE(multi_idx)
