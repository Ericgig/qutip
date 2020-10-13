#cython: language_level=3

from qutip.core.data.csr cimport CSR
from qutip.core.data.csc cimport CSC
from qutip.core.data.dense cimport Dense
from qutip.core.data.base cimport idxint

cpdef Dense matmul_trunc_ket_csr_dense(CSR left, Dense right,
                                       idxint[::1] used_idx,
                                       double complex a=*, Dense out=*)

cpdef Dense matmul_trunc_ket_csc_dense(CSC left, Dense right,
                                       idxint[::1] used_idx,
                                       double complex a=*, Dense out=*)

cpdef Dense matmul_trunc_ket_dense(Dense left, Dense right,
                                   idxint[::1] used_idx,
                                   double complex a=*, Dense out=*)

cpdef Dense matmul_trunc_dm_csr_dense(CSR left,  Dense right,
                                      idxint[::1] used_idx,
                                      double complex a=*, Dense out=*)

cpdef Dense matmul_trunc_dm_csc_dense(CSC left, Dense right,
                                      idxint[::1] used_idx,
                                      double complex a=*, Dense out=*)

cpdef Dense matmul_trunc_dm_dense(Dense left, Dense right,
                                  idxint[::1] used_idx,
                                  double complex a=*, Dense out=*)

cpdef idxint[::1] get_idx_ket(Dense state, double atol, double rtol, int pad=*)

cpdef idxint[::1] get_idx_dm(Dense state, double atol, double rtol, int pad=*)
