#cython: language_level=3

from qutip.core.data.base cimport idxint, Data

cdef class LazyOperator:
    cdef readonly (idxint, idxint) shape
    cdef readonly object dtype
    cdef object function
    cdef tuple args
    cdef dict kwargs
    cdef Data _matrix
    cdef long hash

    cpdef Data get(self)
