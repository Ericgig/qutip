#cython: language_level=3

cdef class Feedback:
    cdef str key
    cdef object _call(self, double t, _data.Data state)

cdef class QobjFeedback(Feedback)
cdef class ExpectFeedback(Feedback)
cdef class CollapseFeedback(Feedback)
