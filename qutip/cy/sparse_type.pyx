include "sparse_type.pxi"

cdef sp_int test_value = 2**32+1
cdef object type
if test_value == 1: #int32
    type = np.int32
else:
    type = np.int64

def sparse_compilation_type():
    return type
