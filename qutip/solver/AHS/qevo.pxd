#cython: language_level=3

from qutip.core.data.base cimport idxint
from qutip.solver._solverqevo cimport SolverQEvo
from qutip.core.data cimport Data, Dense
from qutip.core.cy.cqobjevo cimport LTYPE, CSR_TYPE, Dense_TYPE, CSC_TYPE


cdef class AHS_config:
    cdef:
        double atol
        double rtol
        double safety_rtol
        int padding
        int safety_pad
        double extra_padding
        idxint[::1] limits
        bint passed
        object np_array

cdef class SolverQEvoAHS(SolverQEvo):
    cdef AHS_config config
    cdef LTYPE layer_type
    cdef bint super

    cpdef bint resize(self, Dense state)
    cdef void mul_ahs(self, Data mat,  Dense vec, double complex a, Dense out)
    cpdef idxint[::1] get_idx_ket(self, Dense state)
    cpdef idxint[::1] get_idx_dm(self, Dense state)
