__all__ = ['eigs', 'reshape', 'permute', 'bandwidth', 'profile', 'reverse_permute']

from qutip.matrix.sparse import sp_permute, sp_reverse_permute, sp_reshape
from scipy.sparse import isspmatrix

def eigs(data, isherm, vecs=True, sparse=False, sort='low',
              eigvals=0, tol=0, maxiter=100000):
    return data.eigs(isherm, vecs, sparse, sort, eigvals, tol, maxiter)

def reshape(A, shape, format='csr'):
    if isspmatrix(A):
        return sp_reshape(A, shape, A.format)
    else:
        # dense?
        return A.reshape(shape)

def permute(A, rperm=(), cperm=(), safe=True):
    if A.format in ["csr","csc"]:
        return sp_permute(A,rperm, cperm, safe)
    raise NotImplementedError


def reverse_permute(A, rperm=(), cperm=(), safe=True):
    if A.format in ["csr","csc"]:
        return sp_reverse_permute(A,rperm, cperm, safe)
    raise NotImplementedError

def bandwidth(A):
    return A.bandwidth()

def profile(A):
    return A.profile()
