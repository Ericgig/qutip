#cython: language_level=3

import numpy as np
cimport numpy as cnp

from . cimport base

cdef class CSC(base.Data):
    cdef double complex [::1] data
    cdef base.idxint [::1] col_index
    cdef base.idxint [::1] row_index
    cdef object _scipy
    cdef bint _deallocate
    cpdef CSC copy(CSC self)
    cpdef object as_scipy(CSC self)

cpdef CSC copy_structure(CSC matrix)
cpdef void sort_indices(CSC matrix) nogil
cpdef base.idxint nnz(CSC matrix) nogil
cpdef CSC empty(base.idxint rows, base.idxint cols, base.idxint size)
cpdef CSC zeros(base.idxint rows, base.idxint cols)
cpdef CSC identity(base.idxint dimension, double complex scale=*)
cpdef CSC CSC_from_CSR(CSR matrix)
