
import numpy as np

cdef class QtOdeFuncWrapper:
    cdef object f

    def __init__(self, f):
        self.f = f

    cdef void call(self, QtOdeData out, double t QtOdeData y):
        out.copy(qtodedata(self.f(t, y.raw())))


cdef class QtOdeData:
    def __init__(self, val):
        self.base = val

    cdef void inplace_add(self, QtOdeData outer, double factor):
        self.base += outer * factor

    cdef void zero(self):
        self.base *= 0

    cdef double norm(self):
        return np.sum(np.abs(self.base))

    cdef void copy(self, other):
        self.base *= 0
        self.base += other

    cdef void empty_like(self):
        return self.base.copy()

    cdef object raw(self):
        return self.base

# TODO: use jake's Dispatch
def qtodedata(data):
    if isinstance(data, np.ndarray):
        return QtOdeData(data)
    raise NotImplementedError
