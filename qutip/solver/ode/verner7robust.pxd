#cython: language_level=3
# Verner 7 robust
# http://people.math.sfu.ca/~jverner/RKV76.IIa.Robust.000027015646.081206.CoeffsOnlyFLOAT

cdef set_vern7_rob_ctes(double[:,::1] a, double[::1] b, double[::1] c,
                        double[::1] e, double[:,::1] bi7)
