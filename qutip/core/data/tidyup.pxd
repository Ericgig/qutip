#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from qutip.core.data cimport CSR, Dense, Dia

cpdef CSR tidyup_csr(CSR matrix, double tol)
cpdef Dense tidyup_dense(Dense matrix, double tol)
cpdef Dia tidyup_dia(Dia matrix, double tol)
