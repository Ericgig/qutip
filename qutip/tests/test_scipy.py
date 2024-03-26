import numpy as np
import scipy
import pytest
import scipy.sparse


@pytest.mark.parametrize("N", [10, 100])
def test_sparse_eigen(N):
    matrix = scipy.sparse.random(N, N, density=1/N, format="csr", dtype=np.double, random_state=123)
    out = scipy.sparse.linalg.eigs(matrix, 4)
