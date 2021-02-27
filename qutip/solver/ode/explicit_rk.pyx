#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False, nonecheck=False
"""
Provide a cython implimentation for a general Explicit runge-Kutta method.
"""
from qutip.core.data cimport Data, Dense, CSR, dense
from qutip.core.data.add cimport iadd_dense
from qutip.core.data.add import add
from qutip.core.data.mul import imul_data
from qutip.core.data.tidyup import tidyup_csr
from qutip.core.data.norm import frobenius_data
from .verner7efficient import vern7_coeff
from .verner9efficient import vern9_coeff
cimport cython
import numpy as np

euler_coeff = (
    1,
    np.array([[0.]], dtype=np.float64),
    np.array([1.], dtype=np.float64),
    np.array([0.], dtype=np.float64)
)

rk4_coeff = (
    4,
    np.array([[0., 0., 0., 0.],
              [.5, 0., 0., 0.],
              [0., .5, 0., 0.],
              [0., 0., 1., 0.]], dtype=np.float64),
    np.array([1/6, 1/3, 1/3, 1/6], dtype=np.float64),
    np.array([0., 0.5, 0.5, 1.0], dtype=np.float64)
)


cdef Data empty_like(Data state):
    if type(state) == Dense:
        return dense.empty_like(state)
    return state.copy()


cdef Data copy_to(Data in_, Data out):
    # Copy while reusing allocated buffer if possible.
    # Does not check the shape, etc.
    cdef size_t ptr
    if type(in_) is Dense:
        for ptr in range(in_.shape[0] * in_.shape[1]):
            (<Dense> out).data[ptr] = (<Dense> in_).data[ptr]
        return out
    else:
        return in_.copy()


cdef Data iadd_data(Data left, Data right, double complex factor):
    # left += right * factor
    # reusing `left' allocated buffer if possible.
    # TODO: when/if iadd_csr is added: move to data/add.pyx.
    if factor == 0:
        return left
    if type(left) is Dense:
        iadd_dense(left, right, factor)
        return left
    else:
        return add(left, right, factor)


cdef class Explicit_RungeKutta:
    def __init__(self, f, rtol=1e-6, atol=1e-8, nsteps=1000,
                 first_step=0, min_step=0, max_step=0, interpolate=True,
                 method="euler"):
        self.f = f
        self.atol = atol
        self.rtol = rtol
        self.max_numsteps = nsteps
        self.first_step = first_step
        self.min_step = min_step
        self.max_step = max_step
        self.interpolate = interpolate
        self.k = []
        self.dt_safe = atol
        self._read_method(method)

    def _read_method(self, method):
        if "vern7" == method:
            self._init_coeff(*vern7_coeff)
        elif "vern9" == method:
            self._init_coeff(*vern9_coeff)
        elif "rk4" == method:
            self._init_coeff(*rk4_coeff)
        else:
            self._init_coeff(*euler_coeff)

    def _init_coeff(self, order, a, b, c, e=None, bi=None):
        self.order = order
        self.rk_step = b.shape[0]
        self.rk_extra_step = a.shape[0]
        if (
            self.rk_step > self.rk_extra_step or
            a.shape[1] != self.rk_extra_step or
            c.shape[0] != self.rk_extra_step
        ):
            raise ValueError("Wrong rk coefficient shape")

        self.can_interpolate = bi is not None
        self.interpolate = self.can_interpolate and self.interpolate
        if self.can_interpolate:
            self.denseout_order = bi.shape[1]
            if bi.shape[0] != self.rk_extra_step:
                raise ValueError("Wrong rk coefficient shape")

        self.adaptative_step = e is not None
        if self.adaptative_step and e.shape[0] != self.rk_step:
            raise ValueError("Wrong rk coefficient shape")

        self.a = a
        self.b = b
        self.c = c
        self.e = e
        self.bi = bi
        self.b_factor_np = np.empty(self.rk_extra_step, dtype=np.float64)
        self.b_factor = self.b_factor_np

    cpdef void set_initial_value(self, Data y0, double t):
        self._t = t
        self._t_prev = t
        self._t_front = t
        self.dt_int = 0
        self._y = y0
        self.norm_tmp = frobenius_data(self._y)
        self.norm_front = self.norm_tmp
        self._status = 0

        #prepare_buffer
        for i in range(self.rk_extra_step):
            self.k.append(empty_like(self._y))
        self._y_temp = empty_like(self._y)
        self._y_front = self._y.copy()
        self._y_prev = empty_like(self._y)

        if not self.first_step:
            self.dt_safe = self.estimate_first_step(t, self._y)
        else:
            self.dt_safe = self.first_step

    cdef double estimate_first_step(self, double t, Data y0):
        if not self.adaptative_step:
            return 0.

        cdef double dt1, dt2, dt, factorial = 1
        cdef double norm = frobenius_data(y0), tmp_norm
        cdef double tol = self.atol + norm * self.rtol
        cdef int i
        imul_data(<Data> self.k[0], 0)
        self.k[0] = self.f.mul_data(t, y0, <Data> self.k[0])

        # Good approximation for linear system. But not in a general case.
        if norm == 0:
            norm = 1
        tmp_norm = frobenius_data(<Data> self.k[0])
        for i in range(1, self.order+1):
            factorial *= i
        if tmp_norm != 0:
            dt1 = (tol*factorial*norm**self.order)**(1/(self.order+1)) / tmp_norm
        else:
            dt1 = (tol*factorial*norm**self.order)**(1/(self.order+1))
        copy_to(y0, self._y_temp)
        self._y_temp = iadd_data(self._y_temp, <Data> self.k[0], dt1 / 100)
        imul_data(<Data> self.k[1], 0)
        self.k[1] = self.f.mul_data(t + dt1 / 100, self._y_temp,
                                    <Data> self.k[1])

        self.k[0] = iadd_data(<Data> self.k[0], <Data> self.k[1], -1)
        tmp_norm = frobenius_data(<Data> self.k[0])
        if tmp_norm != 0:
            dt2 = ((tol * factorial* norm**(self.order//2))**(1/(self.order+1)) /
                   (tmp_norm / dt1 * 100)**0.5)
        else:
            dt2 = 0.1
        dt = min(dt1, dt2)
        if self.max_step:
            dt = min(self.max_step, dt)
        if self.min_step:
            dt = max(self.min_step, dt)
        return dt

    cpdef integrate(Explicit_RungeKutta self, double t, bint step=False):
        cdef int nsteps = 0
        cdef double err = 0
        self._status = 0
        if t < self._t_prev:
            self._status = -3
            return

        if t == self._t:
            return

        if step and self._t < self._t_front and t > self._t_front:
            # To ensure that the self._t ... t_out interval can be covered.
            t = self._t_front

        while self._t_front < t and nsteps < self.max_numsteps:
            if err < 1:
                copy_to(self._y_front, self._y_prev)
                self._t_prev = self._t_front
            dt = self.get_timestep(t)
            if dt < 1e-15:
                self._status = -2
                break
            err = self.compute_step(dt)
            self.recompute_safe_step(err, dt)
            if err < 1:
                self.dt_int = dt
                self._t_front = self._t_prev + dt
                self.norm_front = self.norm_tmp
                if step:
                    break
            nsteps += 1

        if nsteps >= self.max_numsteps:
            self._status = -1
        if self._t_front > t + 1e-15:
            self._status = 1
            self.prep_dense_out()
            self._t = t
            self.interpolate_step(t, self._y)
        else:
            self._status = 2
            self._t = self._t_front
            copy_to(self._y_front, self._y)

    cdef double compute_step(self, double dt):
        cdef int i, j
        cdef double t = self._t_front

        for i in range(self.rk_step):
            imul_data(<Data> self.k[i], 0.)

        self.k[0] = self.f.mul_data(t, self._y_prev, <Data> self.k[0])

        for i in range(1, self.rk_step):
            copy_to(self._y_prev, self._y_temp)
            self._y_temp = self.accumulate(self._y_temp, self.a[i,:], dt, i)
            self.k[i] = self.f.mul_data(t + self.c[i]*dt,
                                        self._y_temp, <Data> self.k[i])

        copy_to(self._y_prev, self._y_front)
        self._y_front = self.accumulate(self._y_front, self.b, dt, self.rk_step)
        if type(self._y_front) is CSR:
            # issparse() test would be better.
            tidyup_csr(self._y_front, self.atol/self._y_front.shape[0], True)

        return self.error(self._y_front, dt)

    cdef double error(self, Data y_new, double dt):
        if not self.adaptative_step:
            return 0.
        cdef int j
        imul_data(self._y_temp, 0.)
        self._y_temp = self.accumulate(self._y_temp, self.e, dt, self.rk_step)
        self.norm_tmp = frobenius_data(y_new)
        return frobenius_data(self._y_temp) / (self.atol +
                max(self.norm_tmp, self.norm_front) * self.rtol)

    cdef void prep_dense_out(self):
        cdef:
            double t = self._t_prev
            double dt = self.dt_int

        for i in range(self.rk_step, self.rk_extra_step):
            imul_data(<Data> self.k[i], 0.)
            copy_to(self._y_prev, self._y_temp)
            self._y_temp = self.accumulate(self._y_temp, self.a[i,:], dt, i)
            self.k[i] = self.f.mul_data(t + self.c[i]*dt, self._y_temp, <Data> self.k[i])

    cdef void interpolate_step(self, double t, Data out):
        cdef:
            int i, j
            double t0 = self._t_prev
            double dt = self.dt_int
            double tau = (t - t0) / dt
        for i in range(self.rk_extra_step):
            self.b_factor[i] = 0.
            for j in range(self.denseout_order-1, -1, -1):
                self.b_factor[i] += self.bi[i,j]
                self.b_factor[i] *= tau

        copy_to(self._y_prev, out)
        out = self.accumulate(out, self.b_factor, dt, self.rk_extra_step)

    cdef Data accumulate(self, Data target, double[:] factors,
                         double dt, int size):
        cdef int i
        for i in range(size):
            target = iadd_data(target, <Data> self.k[i], dt * factors[i])
        return target

    @cython.cdivision(True)
    cdef double get_timestep(self, double t):
        if not self.adaptative_step:
            return t - self._t_front
        if self.interpolate:
            return self.dt_safe
        dt_needed = t - self._t_front
        if dt_needed <= self.dt_safe:
            return dt_needed
        return dt_needed / (int(dt_needed / self.dt_safe) + 1)

    cdef double recompute_safe_step(self, double err, double dt):
        cdef factor = 0.
        if not self.adaptative_step:
            factor = 1
        elif err == 0:
            factor = 10
        else:
            factor = 0.9*err**(-1/(self.order+1))
            factor = min(10, factor)
            factor = max(0.2, factor)

        self.dt_safe = dt * factor
        if self.max_step:
            self.dt_safe = min(self.max_step, self.dt_safe)
        if self.min_step:
            self.dt_safe = max(self.min_step, self.dt_safe)

    def successful(self):
        return self._status >= 0

    @property
    def status(self):
        return self._status

    @property
    def y(self):
        return self._y

    @property
    def y_prev(self):
        return self._y_prev

    @property
    def y_front(self):
        return self._y_front

    @property
    def t_front(self):
        return self._t_front

    @property
    def t_prev(self):
        return self._t_prev

    @property
    def t(self):
        return self._t

    def print_ks(self):
        for i in range(self.rk_step):
            print(self.k[i].to_array())

    def print_state(self):
        print(self._t, self._t_prev, self._t_front)
        print(self.dt_int, self.dt_safe)
        print(self._status)
        print(self.y.to_array(), self._y_prev.to_array())
        print(self.norm_tmp, self.norm_front)
        print(self.order, self.rk_step, self.rk_extra_step, self.denseout_order)
        print(self.a.shape)
        print(self.b.shape)
        print(self.c.shape)

    def print_table(self):
        for i in range(self.a.shape[0]):
            for j in range(i):
                print("a", i, j, self.a[i,j])

        for i in range(self.b.shape[0]):
            print("b", i, self.b[i])

        for i in range(self.c.shape[0]):
            print("c", i, self.c[i])

        if self.e is not None:
            for i in range(self.e.shape[0]):
                print("e", i, self.e[i])
