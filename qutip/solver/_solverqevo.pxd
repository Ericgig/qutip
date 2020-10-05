#cython: language_level=3

cimport qutip.core.data as _data
from qutip.core.cy.cqobjevo cimport CQobjEvo
from qutip.core.data.base cimport idxint

cdef class SolverQEvo:
    cdef CQobjEvo base
    cdef idxint ncols
    cdef bint has_dynamic_args
    cdef list dynamic_arguments
    cdef dict args
    cdef list collapse
    cdef void mul_data(self, double t, _data.Data vec, _data.Data out)
    cdef _data.Data jac_data(self, double t)
    cdef void apply_feedback(self, double t, _data.Data matrix) except *
    cpdef void arguments(self, dict args)
