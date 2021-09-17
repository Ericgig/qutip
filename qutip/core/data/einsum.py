from qutip.core.data import csr, dense, Dense, CSR
import numpy as np
import scipy.sparse

class _einsum_parser:
    def __init__(self, subscripts):
        if '->' in subscripts:
            ins, out = subscripts.split('->')
        else:
            ins = subscripts
            out = None
        if len(ins) != 5 or ins[2] != ',':
            raise ValueError
        keys = set(ins) ^ set(',')
        map_ = dict(zip(keys, range(len(keys))))

        # replace the letter indices by index.
        self.ins = [map_[key] for key in ins if key != ',']
        if out is None:
            self.out = [i for i in self.ins if self.ins.count(i) == 1]
        else:
            self.out = [map_[key] for key in out]
        if len(self.out) >= 3:
            raise ValueError
        print(self.ins, self.out, map_)

    def __call__(self, row_l, col_l, row_r, col_r):
        """For the indices of both matrices match the pattern.
        Return the indices on the out matrix if the pattern is respected or
        ``False``.
        """
        idx = [-1, -1, -1, -1]
        for pos, val in zip(self.ins, [row_l, col_l, row_r, col_r]):
            if not self._add_set(pos, val, idx):
                return False
        if len(self.out) == 0:
            return 0, 0
        if len(self.out) == 1:
            return idx[self.out[0]], 0
        return idx[self.out[0]], idx[self.out[1]]

    def _add_set(self, pos, val, idx):
        """`pos` is the indice 'i', `val` is the row or col.
        If the indice is not set (-1) save it and return True.
        If the indice is set, return whether that the val match as a bool.
        """
        if idx[pos] == -1:
            idx[pos] = val
        else:
            if idx[pos] != val:
                return False
        return True

    @property
    def return_scalar(self):
        return len(self.out) == 0


def einsum_csr(subscripts, left, right):
    parser = _einsum_parser(subscripts)
    shape = parser(*left.shape, *right.shape)
    print(subscripts, left.shape, right.shape, shape)
    if not shape:
        raise ValueError
    if shape[1] == 0:
        shape = (shape[0], 1)
    left = left.as_scipy()
    right = right.as_scipy()
    position = []
    vals = []
    for row_left in range(left.shape[0]):
        for ptr_left in range(left.indptr[row_left], left.indptr[row_left+1]):
            col_left = left.indices[ptr_left]
            data_left = left.data[ptr_left]
            for row_right in range(right.shape[0]):
                for ptr_right in range(right.indptr[row_right], right.indptr[row_right+1]):
                    col_right = right.indices[ptr_right]
                    data_right = right.data[ptr_right]
                    pos = parser(row_left, col_left, row_right, col_right)
                    if pos:
                        position.append(pos)
                        vals.append(data_left * data_right)

    if parser.return_scalar:
        return np.sum(vals)
    if len(position) == 0:
        return csr.zeros(*shape)
    return CSR(scipy.sparse.csr_matrix((vals, zip(*position)), shape=shape),
               copy=False)


def einsum_dense(subscripts, left, right):
    """Evaluates the Einstein summation convention on two data object.
    See :func:`numpy.einsum`.
    """
    out = np.einsum(subscripts, left.as_ndarray(), right.as_ndarray())
    if isinstance(out, np.ndarray):
        out = Dense(out, copy=False)
    return out


from .dispatch import Dispatcher as _Dispatcher
# We set out as False since it can return a scalar, ket or operator.
einsum = _Dispatcher(einsum_dense, name='einsum',
                     inputs=('left', 'right'), out=False)
einsum.add_specialisations([
    (CSR, CSR, einsum_csr),
    (Dense, Dense, einsum_dense),
], _defer=True)

del _Dispatcher
