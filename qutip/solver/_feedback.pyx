#cython: language_level=3
from qutip import Qobj, spre
from qutip.core import data
from qutip.core cimport data as _data

cdef class Feedback:
    def __init__(self, key, state):
        self.key = key

    cdef object _call(self, double t, _data.Data state):
        return 0


cdef class QobjFeedback(Feedback):
    def __init__(self, key, state):
        self.key = key
        self.dims = state.dims
        self.type = state.type
        self.superrep = state.superrep
        self.isherm = state.isherm
        self.isunitary = state.isunitary
        self.shape = state.shape

    cdef object _call(self, double t, _data.Data state):
        cdef _data.Data matrix = data.reshape(state,
                                              self.shape[0], self.shape[1])
        return Qobj(arg=matrix, dims=self.dims,
                    type=self.type, copy=False,
                    superrep=self.superrep, isherm=self.isherm,
                    isunitary=self.isunitary)


cdef class ExpectFeedback(Feedback):
    def __init__(self, key, op, issuper):
        self.key = key
        if issuper:
            self.op = spre(op).data
        else:
            self.op = op.data
        self.issuper = issuper

    cdef object _call(self, double t, _data.Data state):
        if self.issuper:
            return data.expect_super(self.op, state)
        return data.expect(self.op, state)


cdef class CollapseFeedback(Feedback):
    def __init__(self, key, collapse):
        self.key = key
        self.collapse = collapse

    cdef object _call(self, double t, _data.Data state):
        return self.collapse
