#cython: language_level=3
from qutip.core cimport data as _data


cdef class Feedback:
    cdef str key
    cdef object _call(self, double t, _data.Data state)

cdef class QobjFeedback(Feedback):
    cdef:
        object dims
        object type
        object superrep
        object isherm
        object isunitary
        object shape

cdef class ExpectFeedback(Feedback):
    cdef _data.Data op
    cdef bint issuper

cdef class CollapseFeedback(Feedback):
    cdef list collapse
