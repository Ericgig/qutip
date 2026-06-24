#cython: language_level=3

from qutip.core.data.base cimport Data


cdef class LazyOperator:
    def __init__(self, function, inputs, args, kwargs, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.function = function
        self.inputs = inputs
        self.args = args
        self.kwargs = kwargs
        self.hash = -1
        self._matrix = None

    def __hash__(self):
        if self.hash == -1:
            hash_data = (self.function, self.args)
            if self.kwargs:
                hash_data += (
                    tuple(self.kwargs.keys()),
                    tuple(self.kwargs.values())
                )
            self.hash = hash(hash_data)
        return self.hash

    cpdef Data get(self):
        if self._matrix is None:
            self._matrix = self.function(*self.args, **self.kwargs)
        return self._matrix
