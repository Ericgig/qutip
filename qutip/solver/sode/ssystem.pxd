#cython: language_level=3
from qutip.core.data cimport Data, Dense
from qutip.core.cy.qobjevo cimport QobjEvo

cdef class StochasticSystem:
    cdef readonly int num_collapse
    cpdef Data drift(self, t, Data state)
    cpdef list diffusion(self, t, Data state)

    cpdef list expect(self, t, Data state)

cdef class StochasticSystemWithDer(StochasticSystem):
    cdef Data state
    cdef double t
    cpdef void set_state(self, double t, Data state) except *

    cpdef Data a(self)
    cpdef Data bi(self, int i)
    cpdef complex expect_i(self, int i)
    cpdef Data Libj(self, int i, int j)
    cpdef Data Lia(self, int i)
    cpdef Data L0bi(self, int i)
    cpdef Data LiLjbk(self, int i, int j, int k)
    cpdef Data L0a(self)

cdef class StochasticClosedSystem(StochasticSystem):
    cdef readonly list c_ops
    cdef readonly QobjEvo L

cdef class StochasticOpenSystem(StochasticSystemWithDer):
    cdef readonly list c_ops
    cdef readonly QobjEvo L
