#cython: language_level=3

from qutip.core.data.csr cimport CSR
from qutip.core.data.csc cimport CSC
from qutip.core.data.dense cimport Dense
from qutip.core.data.base cimport idxint

cpdef Dense matmul_psp_ket_csc_dense(CSC left, Dense right,
                                    double atol, double rtol,
                                    double complex a=*, Dense out=*)

cpdef Dense matmul_psp_ket_dense(Dense left, Dense right,
                                 double atol, double rtol,
                                 double complex a=*, Dense out=*)

cpdef Dense matmul_psp_dm_csc_dense(CSC left, Dense right,
                                    double atol, double rtol,
                                    double complex a=*, Dense out=*)

cpdef Dense matmul_psp_dm_dense(Dense left, Dense right,
                                double atol, double rtol,
                                double complex a=*, Dense out=*)
