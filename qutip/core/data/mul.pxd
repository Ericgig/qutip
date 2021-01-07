#cython: language_level=3

from qutip.core.data cimport CSR, Dense

cpdef CSR imul_csr(CSR matrix, double complex value)
cpdef CSR mul_csr(CSR matrix, double complex value)
cpdef CSR neg_csr(CSR matrix)

cpdef Dense imul_dense(Dense matrix, double complex value)
cpdef Dense mul_dense(Dense matrix, double complex value)
cpdef Dense neg_dense(Dense matrix)
