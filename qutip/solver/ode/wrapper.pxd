#cython: language_level=3

cimport numpy as cnp
from ...core cimport data as _data


cdef class QtOdeData:
    cdef double[::1] base
    cdef object _raw
    cpdef void inplace_add(self, QtOdeData other, double factor)
    cpdef void zero(self)
    cpdef double norm(self)
    cpdef void copy(self, QtOdeData other)
    cpdef QtOdeData empty_like(self)
    cpdef object raw(self)
    cpdef _data.Data data(self)
    cpdef void set_data(self, _data.Data new)

cdef class QtOdeFuncWrapper:
    cdef object f
    cpdef void call(self, QtOdeData out, double t, QtOdeData y)


cdef class QtOdeFuncWrapperInplace(QtOdeFuncWrapper):
    pass
