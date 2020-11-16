#cython: language_level=3

cimport qutip.core.data as _data
from qutip.core.cy.cqobjevo cimport CQobjEvo
from qutip.core.data.base cimport idxint

cdef class SolverQEvo:
    cdef dict args
    cdef object base_py
    cdef CQobjEvo base
    cdef list collapse
    cdef list dynamic_arguments
    cdef bint has_dynamic_args
    cdef idxint ncols

    cdef _data.Data mul_data(self, double t, _data.Data vec)
    cdef _data.Dense mul_dense(self, double t, _data.Dense vec, _data.Dense out)
    cdef _data.Data jac_data(self, double t)
    cdef void apply_feedback(self, double t, _data.Data matrix) except *
    cpdef void arguments(self, dict args)
