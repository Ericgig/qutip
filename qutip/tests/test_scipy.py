import numpy as np
import scipy
import pytest
import scipy.sparse

import sys
print(scipy.__version__, np.__version__, sys.version_info)
scipy.show_config()

@pytest.mark.parametrize("N", [10, 100])
def test_sparse_eigen(N):
    matrix = scipy.sparse.random(N, N, format="csr", dtype=np.complex128, random_state=2)
    out = scipy.sparse.linalg.eigs(matrix, 4)
