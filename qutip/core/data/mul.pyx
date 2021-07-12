#cython: language_level=3
#cython: boundscheck=False, wrapround=False, initializedcheck=False

from qutip.core.data cimport idxint, csr, CSR, dense, Dense, Data

__all__ = [
    'mul', 'mul_csr', 'mul_dense',
    'imul', 'imul_csr', 'imul_dense', 'imul_data',
    'neg', 'neg_csr', 'neg_dense',
]


cpdef CSR imul_csr(CSR matrix, double complex value):
    """Multiply this CSR `matrix` by a complex scalar `value`."""
    cdef idxint ptr
    with nogil:
        for ptr in range(csr.nnz(matrix)):
            matrix.data[ptr] *= value
    return matrix

cpdef CSR mul_csr(CSR matrix, double complex value):
    """Multiply this CSR `matrix` by a complex scalar `value`."""
    if value == 0:
        return csr.zeros(matrix.shape[0], matrix.shape[1])
    cdef CSR out = csr.copy_structure(matrix)
    cdef idxint ptr
    with nogil:
        for ptr in range(csr.nnz(matrix)):
            out.data[ptr] = value * matrix.data[ptr]
    return out

cpdef CSR neg_csr(CSR matrix):
    """Unary negation of this CSR `matrix`.  Return a new object."""
    cdef CSR out = csr.copy_structure(matrix)
    cdef idxint ptr
    with nogil:
        for ptr in range(csr.nnz(matrix)):
            out.data[ptr] = -matrix.data[ptr]
    return out


cpdef Dense imul_dense(Dense matrix, double complex value):
    """Multiply this Dense `matrix` by a complex scalar `value`."""
    cdef size_t ptr
    with nogil:
        for ptr in range(matrix.shape[0]*matrix.shape[1]):
            matrix.data[ptr] *= value
    return matrix

cpdef Dense mul_dense(Dense matrix, double complex value):
    """Multiply this Dense `matrix` by a complex scalar `value`."""
    cdef Dense out = dense.empty_like(matrix)
    cdef size_t ptr
    with nogil:
        for ptr in range(matrix.shape[0]*matrix.shape[1]):
            out.data[ptr] = value * matrix.data[ptr]
    return out

cpdef Dense neg_dense(Dense matrix):
    """Unary negation of this CSR `matrix`.  Return a new object."""
    cdef Dense out = dense.empty_like(matrix)
    cdef size_t ptr
    with nogil:
        for ptr in range(matrix.shape[0]*matrix.shape[1]):
            out.data[ptr] = -matrix.data[ptr]
    return out


cdef void _check_shape(Data left, Data right) nogil except *:
    if left.shape[0] != right.shape[0] or left.shape[1] != right.shape[1]:
        raise ValueError(
            "incompatible matrix shapes "
            + str(left.shape)
            + " and "
            + str(right.shape)
        )

cpdef Dense point_multiplication_dense(Dense left, Dense right):
    """
    Element-wise product of the matrices:
        out[i,j] = left[i,j] * right[i,j]
    """
    return Dense(left.as_ndarray(), right.as_ndarray())

cpdef CSR point_multiplication_csr(CSR left, CSR right):
    """
    Element-wise product of the matrices:
        out[i,j] = first[i,j] * right[i,j]
    """
    _check_shape(left, right)
    cdef CSR out
    cdef idxint ptr_left, ptr_left_max, col_left, left_nnz = csr.nnz(left)
    cdef idxint ptr_right, ptr_right_max, col_right, right_nnz = csr.nnz(right)
    cdef idxint nnz=0, row, ncols = left.shape[1]
    cdef idxint worst_nnz = min(left_nnz + right_nnz)

    # Fast paths for zero matrices.
    if right_nnz == 0 or left_nnz == 0:
        return csr.zeros(left.shape[0], left.shape[1])
    # Main path.
    out = csr.empty(left.shape[0], left.shape[1], worst_nnz)

    out.row_index[0] = 0
    ptr_left_max = ptr_right_max = 0
    for row in range(left.shape[0]):
        ptr_left = ptr_left_max
        ptr_left_max = left.row_index[row + 1]
        ptr_right = ptr_right_max
        ptr_right_max = right.row_index[row + 1]
        col_left = (left.col_index[ptr_left]
                    if ptr_left < ptr_left_max else ncols + 1)
        col_right = (right.col_index[ptr_right]
                     if ptr_right < ptr_right_max else ncols + 1)
        while ptr_left < ptr_left_max or ptr_right < ptr_right_max:
            if col_left == col_right:
                out.data[nnz] = left.data[ptr_left] * right.data[ptr_right]
                out.col_index[nnz] = col_left
                nnz += 1
                ptr_left += 1
                ptr_right += 1
            elif col_left < col_right:
                ptr_left += 1
            else:
                ptr_right += 1
            col_left = (left.col_index[ptr_left]
                        if ptr_left < ptr_left_max else ncols + 1)
            col_right = (right.col_index[ptr_right]
                         if ptr_right < ptr_right_max else ncols + 1)

        out.row_index[row + 1] = nnz
    return out


from .dispatch import Dispatcher as _Dispatcher
import inspect as _inspect

mul = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('value', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='mul',
    module=__name__,
    inputs=('matrix',),
    out=True,
)
mul.__doc__ =\
    """Multiply a matrix element-wise by a scalar."""
mul.add_specialisations([
    (CSR, CSR, mul_csr),
    (Dense, Dense, mul_dense),
], _defer=True)

imul = _Dispatcher(
    # Will not be inplce if specialisation does not exist but should still
    # give expected results if used as:
    # mat = imul(mat, x)
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('value', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='imul',
    module=__name__,
    inputs=('matrix',),
    out=True,
)
imul.__doc__ =\
    """Multiply inplace a matrix element-wise by a scalar."""
imul.add_specialisations([
    (CSR, CSR, imul_csr),
    (Dense, Dense, imul_dense),
], _defer=True)

neg = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('matrix', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='neg',
    module=__name__,
    inputs=('matrix',),
    out=True,
)
neg.__doc__ =\
    """Unary element-wise negation of a matrix."""
neg.add_specialisations([
    (CSR, CSR, neg_csr),
    (Dense, Dense, neg_dense),
], _defer=True)

point_multiplication = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('left', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('right', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='point_multiplication',
    module=__name__,
    inputs=('left', 'right'),
    out=True,
)
point_multiplication.__doc__ =\
    """Element-wise matrix multiplication."""
point_multiplication.add_specialisations([
    (CSR, CSR, CSR, point_multiplication_csr),
    (Dense, Dense, Dense, point_multiplication_dense),
], _defer=True)

del _inspect, _Dispatcher


cpdef Data imul_data(Data matrix, double complex value):
    if type(matrix) is CSR:
        return imul_csr(matrix, value)
    elif type(matrix) is Dense:
        return imul_dense(matrix, value)
    else:
        return imul(matrix, value)
