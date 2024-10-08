import sys
import numpy as np
import scipy.sparse as sp
from ctypes import c_int, byref
from numpy.ctypeslib import ndpointer
import time
from qutip.settings import settings as qset

# Load solver functions from mkl_lib
pardiso = qset.mkl_lib.pardiso
pardiso_delete = qset.mkl_lib.pardiso_handle_delete
if sys.maxsize > 2**32: #Running 64-bit
    pardiso_64 = qset.mkl_lib.pardiso_64
    pardiso_delete_64 = qset.mkl_lib.pardiso_handle_delete_64


def _pardiso_parameters(hermitian, has_perm,
                        max_iter_refine,
                        scaling_vectors,
                        weighted_matching):
    iparm = np.zeros(64, dtype=np.int32)
    iparm[0] = 1  # Do not use default values
    iparm[1] = 3  # Use openmp nested dissection
    if has_perm:
        iparm[4] = 1
    iparm[7] = max_iter_refine  # Max number of iterative refinements
    if hermitian:
        iparm[9] = 8
    else:
        iparm[9] = 13
    if not hermitian:
        iparm[10] = int(scaling_vectors)
        iparm[12] = int(weighted_matching)  # Non-symmetric weighted matching
    iparm[17] = -1
    iparm[20] = 1
    iparm[23] = 1  # Parallel factorization
    iparm[26] = 0  # Check matrix structure
    iparm[34] = 1  # Use zero-based indexing
    return iparm


# Set error messages
pardiso_error_msgs = {
    '-1': 'Input inconsistant',
    '-2': 'Out of memory',
    '-3': 'Reordering problem',
    '-4':
        'Zero pivot, numerical factorization or iterative refinement problem',
    '-5': 'Unclassified internal error',
    '-6': 'Reordering failed',
    '-7': 'Diagonal matrix is singular',
    '-8': '32-bit integer overflow',
    '-9': 'Not enough memory for OOC',
    '-10': 'Error opening OOC files',
    '-11': 'Read/write error with OOC files',
    '-12': 'Pardiso-64 called from 32-bit library',
}


def _default_solver_args():
    return {
        'hermitian': False,
        'posdef': False,
        'max_iter_refine': 10,
        'scaling_vectors': True,
        'weighted_matching': True,
        'return_info': False,
    }


class mkl_lu:
    """
    Object pointing to LU factorization of a sparse matrix
    generated by mkl_splu.

    Methods
    -------
    solve(b, verbose=False)
        Solve system of equations using given RHS vector 'b'.
        Returns solution ndarray with same shape as input.

    info()
        Returns the statistics of the factorization and
        solution in the lu.info attribute.

    delete()
        Deletes the allocated solver memory.

    """
    def __init__(self, np_pt=None, dim=None, is_complex=None, data=None,
                 indptr=None, indices=None, iparm=None, np_iparm=None,
                 mtype=None, perm=None, np_perm=None, factor_time=None):
        self._np_pt = np_pt
        self._dim = dim
        self._is_complex = is_complex
        self._data = data
        self._indptr = indptr
        self._indices = indices
        self._iparm = iparm
        self._np_iparm = np_iparm
        self._mtype = mtype
        self._perm = perm
        self._np_perm = np_perm
        self._factor_time = factor_time
        self._solve_time = None

    def solve(self, b, verbose=None):
        b_shp = b.shape
        if b.ndim == 2 and b.shape[1] == 1:
            b = b.ravel()
            nrhs = 1
        elif b.ndim == 2 and b.shape[1] != 1:
            nrhs = b.shape[1]
            b = b.ravel(order='F')
        else:
            b = b.ravel()
            nrhs = 1

        data_type = np.complex128 if self._is_complex else np.float64
        if b.dtype != data_type:
            b = b.astype(np.complex128, copy=False)

        # Create solution array (x) and pointers to x and b
        x = np.zeros(b.shape, dtype=data_type, order='C')
        np_x = x.ctypes.data_as(ndpointer(data_type, ndim=1, flags='C'))
        np_b = b.ctypes.data_as(ndpointer(data_type, ndim=1, flags='C'))

        error = np.zeros(1, dtype=np.int32)
        np_error = error.ctypes.data_as(ndpointer(np.int32, ndim=1, flags='C'))

        # Call solver
        _solve_start = time.time()
        pardiso(
            self._np_pt,
            byref(c_int(1)),
            byref(c_int(1)),
            byref(c_int(self._mtype)),
            byref(c_int(33)),
            byref(c_int(self._dim)),
            self._data,
            self._indptr,
            self._indices,
            self._np_perm,
            byref(c_int(nrhs)),
            self._np_iparm,
            byref(c_int(0)),
            np_b,
            np_x,
            np_error,
        )
        self._solve_time = time.time() - _solve_start
        if error[0] != 0:
            raise Exception(pardiso_error_msgs[str(error[0])])

        if verbose:
            print('Solution Stage')
            print('--------------')
            print('Solution time:                  ',
                  round(self._solve_time, 4))
            print('Solution memory (Mb):           ',
                  round(self._iparm[16]/1024, 4))
            print('Number of iterative refinements:',
                  self._iparm[6])
            print('Total memory (Mb):              ',
                  round(sum(self._iparm[15:17])/1024, 4))
            print()
        return np.reshape(x, b_shp, order=('C' if nrhs == 1 else 'F'))

    def info(self):
        info = {'FactorTime': self._factor_time,
                'SolveTime': self._solve_time,
                'Factormem': round(self._iparm[15]/1024, 4),
                'Solvemem': round(self._iparm[16]/1024, 4),
                'IterRefine': self._iparm[6]}
        return info

    def delete(self):
        # Delete all data
        error = np.zeros(1, dtype=np.int32)
        np_error = error.ctypes.data_as(ndpointer(np.int32, ndim=1, flags='C'))
        pardiso(
            self._np_pt,
            byref(c_int(1)),
            byref(c_int(1)),
            byref(c_int(self._mtype)),
            byref(c_int(-1)),
            byref(c_int(self._dim)),
            self._data,
            self._indptr,
            self._indices,
            self._np_perm,
            byref(c_int(1)),
            self._np_iparm,
            byref(c_int(0)),
            byref(c_int(0)),
            byref(c_int(0)),
            np_error,
        )
        if error[0] == -10:
            raise Exception('Error freeing solver memory')


_MATRIX_TYPE_NAMES = {
    4: 'Complex Hermitian positive-definite',
    -4: 'Complex Hermitian indefinite',
    2: 'Real symmetric positive-definite',
    -2: 'Real symmetric indefinite',
    11: 'Real non-symmetric',
    13: 'Complex non-symmetric',
}


def _mkl_matrix_type(dtype, solver_args):
    if not solver_args['hermitian']:
        return 13 if dtype == np.complex128 else 11
    out = 4 if dtype == np.complex128 else 2
    return out if solver_args['posdef'] else -out


def mkl_splu(A, perm=None, verbose=False, **kwargs):
    """
    Returns the LU factorization of the sparse matrix A.

    Parameters
    ----------
    A : csr_matrix
        Sparse input matrix.
    perm : ndarray (optional)
        User defined matrix factorization permutation.
    verbose : bool {False, True}
        Report factorization details.

    Returns
    -------
    lu : mkl_lu
        Returns object containing LU factorization with a
        solve method for solving with a given RHS vector.

    """
    if not sp.isspmatrix_csr(A):
        raise TypeError('Input matrix must be in sparse CSR format.')

    if A.shape[0] != A.shape[1]:
        raise Exception('Input matrix must be square')

    dim = A.shape[0]
    solver_args = _default_solver_args()
    if set(kwargs) - set(solver_args):
        raise ValueError(
            "Unknown keyword arguments pass to mkl_splu: {!r}"
            .format(set(kwargs) - set(solver_args))
        )
    solver_args.update(kwargs)

    # If hermitian, then take upper-triangle of matrix only
    if solver_args['hermitian']:
        B = sp.triu(A, format='csr')
        A = B  # This gets around making a full copy of A in triu
    is_complex = bool(A.dtype == np.complex128)
    if not is_complex:
        A = sp.csr_matrix(A, dtype=np.float64, copy=False)
    data_type = A.dtype

    # Create pointer to internal memory
    pt = np.zeros(64, dtype=int)
    np_pt = pt.ctypes.data_as(ndpointer(int, ndim=1, flags='C'))

    # Create pointers to sparse matrix arrays
    data = A.data.ctypes.data_as(ndpointer(data_type, ndim=1, flags='C'))
    indptr = A.indptr.ctypes.data_as(ndpointer(np.int32, ndim=1, flags='C'))
    indices = A.indices.ctypes.data_as(ndpointer(np.int32, ndim=1, flags='C'))

    # Setup perm array
    if perm is None:
        perm = np.zeros(dim, dtype=np.int32)
        has_perm = 0
    else:
        has_perm = 1
    np_perm = perm.ctypes.data_as(ndpointer(np.int32, ndim=1, flags='C'))

    # setup iparm
    iparm = _pardiso_parameters(
        solver_args['hermitian'],
        has_perm,
        solver_args['max_iter_refine'],
        solver_args['scaling_vectors'],
        solver_args['weighted_matching'],
    )
    np_iparm = iparm.ctypes.data_as(ndpointer(np.int32, ndim=1, flags='C'))

    # setup call parameters
    mtype = _mkl_matrix_type(data_type, solver_args)

    if verbose:
        print('Solver Initialization')
        print('---------------------')
        print('Input matrix type: ', _MATRIX_TYPE_NAMES[mtype])
        print('Input matrix shape:', A.shape)
        print('Input matrix NNZ:  ', A.nnz)
        print()

    b = np.zeros(1, dtype=data_type)  # Input dummy RHS at this phase
    np_b = b.ctypes.data_as(ndpointer(data_type, ndim=1, flags='C'))
    x = np.zeros(1, dtype=data_type)  # Input dummy solution at this phase
    np_x = x.ctypes.data_as(ndpointer(data_type, ndim=1, flags='C'))

    error = np.zeros(1, dtype=np.int32)
    np_error = error.ctypes.data_as(ndpointer(np.int32, ndim=1, flags='C'))

    # Call solver
    _factor_start = time.time()
    pardiso(
        np_pt,
        byref(c_int(1)),
        byref(c_int(1)),
        byref(c_int(mtype)),
        byref(c_int(12)),
        byref(c_int(dim)),
        data,
        indptr,
        indices,
        np_perm,
        byref(c_int(1)),
        np_iparm,
        byref(c_int(0)),
        np_b,
        np_x,
        np_error,
    )
    _factor_time = time.time() - _factor_start
    if error[0] != 0:
        raise Exception(pardiso_error_msgs[str(error[0])])

    if verbose:
        print('Analysis and Factorization Stage')
        print('--------------------------------')
        print('Factorization time:       ', round(_factor_time, 4))
        print('Factorization memory (Mb):', round(iparm[15]/1024, 4))
        print('NNZ in LU factors:        ', iparm[17])
        print()

    return mkl_lu(np_pt, dim, is_complex, data, indptr, indices,
                  iparm, np_iparm, mtype, perm, np_perm, _factor_time)


def mkl_spsolve(A, b, perm=None, verbose=False, **kwargs):
    """
    Solves a sparse linear system of equations using the
    Intel MKL Pardiso solver.

    Parameters
    ----------
    A : csr_matrix
        Sparse matrix.
    b : ndarray or sparse matrix
        The vector or matrix representing the right hand side of the equation.
        If a vector, b.shape must be (n,) or (n, 1).
    perm : ndarray (optional)
        User defined matrix factorization permutation.

    Returns
    -------
    x : ndarray or csr_matrix
        The solution of the sparse linear equation.
        If b is a vector, then x is a vector of size A.shape[1]
        If b is a matrix, then x is a matrix of size (A.shape[1], b.shape[1])

    """
    lu = mkl_splu(A, perm=perm, verbose=verbose, **kwargs)
    b_is_sparse = sp.isspmatrix(b)
    b_shp = b.shape
    if b_is_sparse and b.shape[1] == 1:
        b = b.toarray()
        b_is_sparse = False
    elif b_is_sparse and b.shape[1] != 1:
        nrhs = b.shape[1]
        if lu._is_complex:
            b = sp.csc_matrix(b, dtype=np.complex128, copy=False)
        else:
            b = sp.csc_matrix(b, dtype=np.float64, copy=False)

    # Do dense RHS solving
    if not b_is_sparse:
        x = lu.solve(b, verbose=verbose)
    # Solve each RHS vec individually and convert to sparse
    else:
        data_segs = []
        row_segs = []
        col_segs = []
        for j in range(nrhs):
            bj = b[:, j].toarray().ravel()
            xj = lu.solve(bj)
            w = np.flatnonzero(xj)
            segment_length = w.shape[0]
            row_segs.append(w)
            col_segs.append(np.ones(segment_length, dtype=np.int32)*j)
            data_segs.append(np.asarray(xj[w], dtype=xj.dtype))
        sp_data = np.concatenate(data_segs)
        sp_row = np.concatenate(row_segs)
        sp_col = np.concatenate(col_segs)
        x = sp.csr_matrix((sp_data, (sp_row, sp_col)), shape=b_shp)

    info = lu.info()
    lu.delete()
    return (x, info) if kwargs.get('return_info', False) else x
