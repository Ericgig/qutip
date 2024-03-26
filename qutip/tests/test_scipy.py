import numpy as np
import scipy
import pytest
import scipy.sparse

@pytest.mark.parametrize("N", [10, 100])
def test_sparse_eigen_eye(N):
    matrix = scipy.sparse.csr_matrix(np.eye(10))
    out = scipy.sparse.linalg.eigs(matrix, 4)

@pytest.mark.parametrize("N", [10, 100])
def test_sparse_eigen(N):
    matrix = scipy.sparse.random(N, N, format="csr", dtype=np.double, random_state=1)
    out = scipy.sparse.linalg.eigs(matrix, 4)

@pytest.mark.parametrize("N", [10, 100])
def test_sparse_eigen(N):
    matrix = scipy.sparse.random(N, N, format="csr", dtype=np.complex128)
    out = scipy.sparse.linalg.eigs(matrix, 4)

@pytest.mark.parametrize("N", [10, 100])
def test_sparse_svd(N):
    matrix = scipy.sparse.random(N, N, format="csr", dtype=np.complex128)
    out = scipy.sparse.linalg.svds(matrix, 4)
