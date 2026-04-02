#cython: language_level=3

from qutip.core.data.csr cimport CSR
from qutip.core.data.dia cimport Dia
from qutip.core.data.dense cimport Dense

cpdef CSR pow_csr(CSR matrix, unsigned long long n)
cpdef Dia pow_dia(Dia matrix, unsigned long long n)
cpdef Dense pow_dense(Dense matrix, unsigned long long n)
