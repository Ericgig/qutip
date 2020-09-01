





cdef class QtOdeData:
    cdef void inplace_add(self, QtOdeData outer, double factor)
    cdef void zero(self)
    cdef double norm(self):
    cdef void copy(self, other)
    cdef void empty_like(self)
    cdef object raw(self)


cdef class QtOdeFuncWrapper:
    cdef void call(self, QtOdeData out, double t QtOdeData y)
