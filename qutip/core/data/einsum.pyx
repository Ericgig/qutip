
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
        in_l, in_r = ins.split(",")
        keys = set(ins) ^ set(',')
        map_ = dict(zip(keys, range(len(keys))))

        # replace the letter indices by index.
        self.ins = [map_[key] for key in in_r if key is not ',']
        if out is None:
            self.out = [i for i in self.ins if self.ins.count(i) == 1]
            if len(self.out) >= 3:
                return ValueError
        else:
            self.out = [map_[key] for key in out]

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
    out_shape = _einsum_parser(left.shape, right.shape)
    if not out_shape:
        return ValueError
    position = []
    vals = []
    for col_left in range(left.shape[0]):
        for ptr_left in range(left.indptr[col_left], left.indptr[col_left+1]):
            row_left = left.indices[ptr_left]
            data_left = left.data[ptr_left]
            for col_right in range(right.shape[0]):
                for ptr_right in range(right.indptr[col_right], right.indptr[col_right+1]):
                    row_right = right.indices[ptr_right]
                    data_right = right.data[ptr_right]
                    pos = parser(row_left, col_left, row_right, col_right)
                    if pos:
                        position.append(pos)
                        vals.append(complex(data_left * data_right))

    if _einsum_parser.return_scalar:
        return np.sum(vals)
    if len(position) == 0:
        return csr.zeros(*out_shape)
    return CSR(sp.csr_matrix((vals, zip(*position)), shape=out_shape),
               copy=False)


def einsum_dense(subscripts, left, right):
    """Evaluates the Einstein summation convention on two data object.
    See :func:`numpy.einsum`.
    """
    out = np.einsum(subscripts, left.as_ndarray(), right.as_ndarray())
    if isinstance(out, np.ndarray):
        out = Dense(out, copy=False)
