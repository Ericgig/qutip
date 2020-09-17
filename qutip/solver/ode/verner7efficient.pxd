#cython: language_level=3
# Verner 7 Efficient
# http://people.math.sfu.ca/~jverner/RKV76.IIa.Efficient.00001675585.081206.CoeffsOnlyFLOAT

cdef class vern7_eff_cte:
    cdef double[16] c
    cdef double[16][15] a
    cdef double[10] b
    cdef double[10] e
    cdef double[16][15] bi7
