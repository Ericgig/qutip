#!python
#cython: language_level=3


from qutip.matrix.cy.csr_matrix cimport cy_csr_matrix
from qutip.matrix.cy.cqobjevo cimport CQobjEvo

# Can't do array of python object (cy_csr_matrix)
cdef struct _csr_mat:
    double complex * data
    int * indices
    int * indptr

cdef class CQobjCte(CQobjEvo):
    cdef int total_elem
    # pointer to data
    cdef cy_csr_matrix cte


cdef class CQobjEvoTd(CQobjEvo):
    cdef long total_elem
    # pointer to data
    cdef cy_csr_matrix cte
    cdef _csr_mat ** ops
    cdef long[::1] sum_elem

    cdef void _factor(self, double t)
    cdef void _call_core(self, cy_csr_matrix out, complex* coeff)


"""cdef class CQobjEvoTdMatched(CQobjEvo):
    cdef int nnz

    # data as array
    cdef int[::1] indptr
    cdef int[::1] indices
    cdef complex[::1] cte
    cdef complex[:, ::1] ops

    # prepared buffer
    cdef complex[::1] data_t
    cdef complex* data_ptr

    cdef void _factor(self, double t)
    cdef void _call_core(self, complex[::1] out, complex* coeff)"""
