#cython: language_level=3

from qutip.core.data.csr cimport CSR
from qutip.core.data.csc cimport CSC
from qutip.core.data.Dense cimport Dense
from qutip.core.data.base cimport idxint

cpdef Dense matmul_trunc_ket_csr_dense(CSR left, Dense right,
                                       idxint[::1] used_idx,
                                       double complex a=1, Dense out=None)

cpdef Dense matmul_trunc_ket_csc_dense(CSC left, Dense right,
                                       idxint[::1] used_idx,
                                       double complex a=1, Dense out=None)

cpdef Dense matmul_trunc_ket_dense(Dense left, Dense right,
                                   idxint[::1] used_idx,
                                   double complex a=1, Dense out=None)

cpdef Dense matmul_trunc_dm_csr_dense(CSR left,  Dense right,
                                      idxint[::1] used_idx,
                                      double complex a=1, Dense out=None)

cpdef Dense matmul_trunc_dm_cc_dense(CSC left, Dense right,
                                     idxint[::1] used_idx,
                                     double complex a=1, Dense out=None)

cpdef Dense matmul_trunc_dm_dense(Dense left, Dense right,
                                  idxint[::1] used_idx,
                                  double complex a=1, Dense out=None)

cpdef idxint[::1] get_idx_ket(Dense state, double atol, double rtol, int pad=*)

cpdef idxint[::1] get_idx_dm(Dense state, double atol, double rtol, int pad=*)
