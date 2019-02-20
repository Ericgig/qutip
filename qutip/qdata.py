

import numpy as np
from scipy.sparse import issparse, coo_matrix, csr_matrix

from qutip.matrix.qdata import _qdata
from qutip.matrix.csr import (csr_qmatrix, csr_qmatrix_identity,
                              csr_qmatrix_from_csr,
                              csr_qmatrix_from_coo,
                              csr_qmatrix_from_dense)

KNOWN_FORMAT = "csr"

def qdata(data, format="csr", copy=False, shape=()):
    if isinstance(data, Qobj):
        return qdata(data.data, format, copy)
    if isinstance(data, _qdata) and not copy:
        return data
    elif isinstance(data, _qdata):
        return data.copy()
    elif issparse(data):
        return qdata_from_sparse(data, format, copy)
    elif isinstance(data, np.ndarray):
        return qdata_from_dense(data, format, copy)
    elif isinstance(data, tuple) and len(data) == 3:
        # expect (data, ind, ptr)
        return qdata_from_sparse(csr_matrix(data, copy=copy, shape=shape),
                                 format, False)
    else:
        # cross fingers
        return qdata_from_dense(np.array(data), format, copy)

def qdata_from_dense(dense_data, format="csr", copy=False):
    if format == "csr":
        return csr_qmatrix_from_dense(dense_data)
    else:
        raise Exception("Unknown format")

def qdata_from_sparse(sparse, format="csr", copy=False):
    if format == "csr":
        if sparse.format == "coo":
            return csr_qmatrix_from_coo(sparse)
        if sparse.format != "csr":
            sparse = sparse.tocsr()
        return csr_qmatrix_from_csr(sparse, copy)
    else:
        raise Exception("Unknown format")

def qdata_identity(N, format="csr"):
    if format == "csr":
        return csr_qmatrix_identity(N)
    else:
        raise Exception("Unknown format")

def qdata_empty(shape=(0,0), format="csr"):
    if format == "csr":
        return csr_qmatrix(shape=shape)
    else:
        raise Exception("Unknown format")

def qdata_to_coo(qdata):
    if issparse(qdata):
        return qdata.tocoo()
    else:
        return coo_matrix(qdata.toarray())




from qutip.qobj import Qobj
