#cython: language_level=3

from qutip.core.data.csr cimport CSR
from qutip.core.data.csc cimport CSC
from qutip.core.data.base cimport idxint

cpdef void mv_ahs_csr(CSR matrix,
                     double complex[:] vec,
                     double complex[:] out,
                     idxint[:] rows)

cpdef void mv_ahs_csr_dm(CSR matrix,
                        double complex[:] vec,
                        double complex[:] out,
                        idxint[:] rows)

cpdef void mv_ahs_csc(CSC matrix,
                      double complex[:] vec,
                      double complex[:] out,
                      idxint[:] cols)

cpdef void mv_pseudo_ahs_csc(CSC matrix,
                            double complex[:] vec,
                            double complex[:] out,
                            double atol,
                            double rtol)

cpdef void mv_pseudo_ahs_csc_dm(CSC matrix,
                               double complex[:] vec,
                               double complex[:] out,
                               double atol,
                               double rtol)
