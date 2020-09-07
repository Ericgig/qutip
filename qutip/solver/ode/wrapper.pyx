#cython: language_level=3

import numpy as np
from .._solverqevo cimport SolverQEvo
from ...core cimport data as _data
from ...core.data.norm cimport frobenius_dense, frobenius_csr

cdef class QtOdeFuncWrapper:
    def __init__(self, f):
        self.f = f

    cpdef void call(self, QtOdeData out, double t, QtOdeData y):
        out.copy(QtOdeData(self.f(t, y.raw())))


cdef class QtOdeFuncWrapperInplace(QtOdeFuncWrapper):
    def __init__(self, f):
        self.f = f

    cpdef void call(self, QtOdeData out, double t, QtOdeData y):
        self.f(out.raw(), t, y.raw())


cdef class QtOdeFuncWrapperSolverQEvo(QtOdeFuncWrapper):
    cdef SolverQEvo evo

    def __init__(self, SolverQEvo evo):
        self.evo = evo

    cpdef void call(self, QtOdeData out, double t, QtOdeData y):
        self.evo.mul_data(t, y.data(), out.data())


cdef class QtOdeData:
    def __init__(self, val):
        self._raw = np.array(val)
        self.base = self._raw.ravel()

    cpdef void inplace_add(self, QtOdeData other, double factor):
        cdef int i, len_ = self.base.shape[0]
        for i in range(len_):
            self.base[i] += other.base[i] * factor

    cpdef void zero(self):
        cdef int i, len_ = self.base.shape[0]
        for i in range(len_):
            self.base[i] *= 0.

    cpdef double norm(self):
        cdef int i, len_ = self.base.shape[0]
        cdef double sum = 0.
        for i in range(len_):
            sum += self.base[i] * self.base[i]
        return sum

    cpdef void copy(self, QtOdeData other):
        cdef int i, len_ = self.base.shape[0]
        for i in range(len_):
            self.base[i] = other.base[i]

    cpdef QtOdeData empty_like(self):
        return QtOdeData(self.base.copy())

    cpdef object raw(self):
        return self._raw

    cpdef _data.Data data(self):
        raise NotImplementedError


cdef class QtOdeDataCSR(QtOdeData):
    cdef _data.CSR csr

    def __init__(self, val):
        self.csr = val

    cpdef void inplace_add(self, QtOdeData other, double factor):
        self.csr = _data.add_csr(self.csr, (<QtOdeDataCSR> other).csr, factor)

    cpdef void zero(self):
        self.csr = _data.mul_csr(self.csr, 0.)

    cpdef double norm(self):
        return frobenius_csr(self.csr)

    cpdef void copy(self, QtOdeData other):
        self.csr = (<QtOdeDataCSR> other).csr.copy()

    cpdef QtOdeData empty_like(self):
        return QtOdeDataCSR(self.csr.copy())

    cpdef object raw(self):
        return self.csr

    cpdef _data.Data data(self):
        return self.csr


cdef class QtOdeDataDense(QtOdeData):
    cdef _data.Dense dense

    def __init__(self, val):
        self.dense = val

    cpdef void inplace_add(self, QtOdeData other, double factor):
        _data.add_dense_eq_order_inplace(self.dense,
                                         (<QtOdeDataDense> other).dense,
                                         factor)

    cpdef void zero(self):
        cdef size_t ptr
        for ptr in range(self.dense.shape[0] * self.dense.shape[1]):
            self.dense.data[ptr] = 0.
        # _data.mul_dense_inplace(self.dense, 0.)

    cpdef double norm(self):
        return frobenius_dense(self.dense)

    cpdef void copy(self, QtOdeData other):
        cdef size_t ptr
        for ptr in range(self.dense.shape[0] * self.dense.shape[1]):
            self.dense.data[ptr] = (<QtOdeDataDense> other).dense.data[ptr]

    cpdef QtOdeData empty_like(self):
        return QtOdeDataDense(_data.dense.empty_like(self.dense))

    cpdef object raw(self):
        return self.dense

    cpdef _data.Data data(self):
        return self.dense


cdef class QtOdeDataCpxArray(QtOdeDataDense):
    def __init__(self, val):
        self.dense = _data.fast_from_numpy(val)

    cpdef object raw(self):
        return self.dense.as_ndarray()


# TODO: use jake's Dispatch
def qtodedata(data):
    if isinstance(data, np.ndarray) and data.dtype == np.double:
        return QtOdeData(data)
    if isinstance(data, np.ndarray) and data.dtype == np.complex:
        return QtOdeDataCpxArray(data)
    if isinstance(data, _data.CSR):
        return QtOdeDataCSR(data)
    if isinstance(data, _data.Dense):
        return QtOdeDataDense(data)
    raise NotImplementedError
