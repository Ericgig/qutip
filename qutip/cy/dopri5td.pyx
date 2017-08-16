# distutils: language = c++

import numpy as np
cimport numpy as np

cdef extern from "Python.h":
    object PyLong_FromVoidPtr(void *)
    void* PyLong_AsVoidPtr(object)

"""cdef extern from "src/dopri5td.cpp":
    cdef cppclass ode:
        ode()
        ode(int*, double*, void (*_H)(double, complex *, complex *))
        int step(double, double, complex*, complex*)
        double integrate(double, double, double, complex*, double*)"""

cdef extern from "/home/eric/algo/qutip/qutip/qutip/cy/src/dopri5td.cpp":
    cdef cppclass ode:
        ode()
        ode(int*, double*, void (*_H)(double, complex *, complex *))
        int step(double, double, complex*, complex*)
        double integrate(double, double, double, complex*, double*)

cdef class ode_td_dopri:
    cdef ode* cobj

    def __init__(self, int l, rhs_function, config):
        _y1 = np.zeros(l,dtype=complex)
        cdef int[::1] int_option = np.zeros(2,dtype=np.intc)
        cdef double[::1] double_option = np.zeros(5,dtype=np.double)
        cdef void* rhs_ptr = PyLong_AsVoidPtr(rhs_function)
        cdef void (*rhs)(double, complex*, complex*)
        rhs = <void (*)(double, complex*, complex*)> rhs_ptr

        int_option[0]=l
        int_option[1]=config.norm_steps

        double_option[0]=config.options.atol
        double_option[1]=config.options.rtol
        double_option[2]=config.options.min_step
        double_option[3]=config.options.max_step
        double_option[4]=config.norm_tol

        self.cobj = new ode(<int*>&int_option[0],
                            <double*>&double_option[0],
                            rhs)
        if self.cobj == NULL:
            raise MemoryError('Not enough memory.')

    def __del__(self):
        del self.cobj

    cpdef double integrate(self, double _t_in, double _t_target, double rand, \
                           complex[::1] _psi, double[::1] _err):
        return self.cobj.integrate(_t_in, _t_target, rand,
                                   <complex*>&_psi[0],
                                   <double*>&_err[0])
