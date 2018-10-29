#!python
#cython: language_level=3
import numpy as np
cimport numpy as np
import cython
cimport cython
cimport libc.math
from qutip.cy.inter import prep_cubic_spline
from qutip.cy.inter cimport (spline_complex_cte_second,
                            spline_complex_t_second)
from qutip.cy.interpolate cimport (interp, zinterp)
include "complex_math.pxi"



cdef class coeffFunc:
    def __init__(self, ops, args, tlist):
        pass

    def __call__(self, double t, args={}):
        cdef np.ndarray[ndim=1, dtype=complex] coeff = \
                                            np.zeros(self.N_ops, dtype=complex)
        self._call_core(t, &coeff[0])
        return coeff

    def set_args(self, args):
        pass

    cdef void _call_core(self, double t, complex * coeff):
        pass

    def __getstate__(self):
        pass

    def __setstate__(self, state):
        pass


cdef class interpolate_coeff(coeffFunc):
    cdef double a, b
    cdef complex[:,::1] c

    def __init__(self, ops, args, tlist):
        self.N_ops = len(ops)
        self.a = ops[0][2].a
        self.b = ops[0][2].b
        l = len(ops[0][2].coeffs)
        self.c = np.zeros((self.N_ops,l), dtype=complex)
        for i in range(self.N_ops):
            for j in range(l):
                self.c[i,j] = ops[i][2].coeffs[j]

    def __call__(self, t, args={}):
        cdef np.ndarray[ndim=1, dtype=complex] coeff = \
                                            np.zeros(self.N_ops, dtype=complex)
        self._call_core(t, &coeff[0])
        return coeff

    cdef void _call_core(self, double t, complex * coeff):
        for i in range(self.N_ops):
            coeff[i] = zinterp(t, self.a, self.b, self.c[i,:])

    def set_args(self, args):
        pass

    def __getstate__(self):
        return (self.N_ops, self.a, self.b, np.array(self.c))

    def __setstate__(self, state):
        self.N_ops = state[0]
        self.a = state[1]
        self.b = state[2]
        self.c = state[3]


cdef class inter_coeff_cte(coeffFunc):
    cdef int l
    cdef double dt
    cdef double[::1] tlist
    cdef complex[:,::1] y, M

    def __init__(self, ops, args, tlist):
        self.N_ops = len(ops)
        self.tlist = tlist
        self.l = len(tlist)
        self.dt = tlist[1]-tlist[0]
        self.y = np.zeros((self.N_ops,self.l), dtype=complex)
        self.M = np.zeros((self.N_ops,self.l), dtype=complex)

        for i in range(self.N_ops):
            m, cte = prep_cubic_spline(ops[i][2], tlist)
            if not cte:
                raise Exception("tlist not sampled uniformly")
            for j in range(self.l):
                self.y[i,j] = ops[i][2][j]
                self.M[i,j] = m[j]

    cdef void _call_core(self, double t, complex * coeff):
        for i in range(self.N_ops):
            coeff[i] = spline_complex_cte_second(t, self.tlist,
                                    self.y[i,:], self.M[i,:], self.l, self.dt)

    def set_args(self, args):
        pass

    def __getstate__(self):
        return (self.N_ops, self.l, self.dt, np.array(self.tlist),
                np.array(self.y), np.array(self.M))

    def __setstate__(self, state):
        self.N_ops = state[0]
        self.l = state[1]
        self.dt = state[2]
        self.tlist = state[3]
        self.y = state[4]
        self.M = state[5]


cdef class inter_coeff_t(coeffFunc):
    cdef int l
    cdef double dt
    cdef double[::1] tlist
    cdef complex[:,::1] y, M

    def __init__(self, ops, args, tlist):
        self.N_ops = len(ops)
        self.tlist = tlist
        self.l = len(tlist)
        self.y = np.zeros((self.N_ops,self.l), dtype=complex)
        self.M = np.zeros((self.N_ops,self.l), dtype=complex)
        for i in range(self.N_ops):
            m, cte = prep_cubic_spline(ops[i][2], tlist)
            if cte:
                print("tlist not uniformly?")
            for j in range(self.l):
                self.y[i,j] = ops[i][2][j]
                self.M[i,j] = m[j]

    cdef void _call_core(self, double t, complex * coeff):
        for i in range(self.N_ops):
            coeff[i] = spline_complex_t_second(t, self.tlist,
                                    self.y[i,:], self.M[i,:], self.l)

    def set_args(self, args):
        pass

    def __getstate__(self):
        return (self.N_ops, self.l, None, np.array(self.tlist),
                np.array(self.y), np.array(self.M))

    def __setstate__(self, state):
        self.N_ops = state[0]
        self.l = state[1]
        self.tlist = state[3]
        self.y = state[4]
        self.M = state[5]


cdef class str_coeff(coeffFunc):
    def __init__(self, ops, args, tlist):
        self.N_ops = len(ops)
        self.args = args
        self.set_args(args)

    def __call__(self, double t, args={}):
        cdef np.ndarray[ndim=1, dtype=complex] coeff = \
                                            np.zeros(self.N_ops, dtype=complex)
        if args:
            now_args = self.args.copy()
            now_args.update(args)
            self.set_args(now_args)
            self._call_core(t, &coeff[0])
            self.set_args(self.args)
        else:
            self._call_core(t, &coeff[0])
        return coeff

    def __getstate__(self):
        return (self.N_ops, self.args)

    def __setstate__(self, state):
        self.N_ops = state[0]
        self.args = state[1]
        self.set_args(self.args)
