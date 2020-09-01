
from .verner7efficient cimport set_vern7_eff_ctes
from .wrapper cimport QtOdeData, QtOdeFuncWrapper
from .wrapper import qtodedata

cdef class vern7:
    cdef QtOdeData[16] k
    cdef QtOdeData yp
    cdef QtOdeData yp8
    cdef QtOdeData yp9
    cdef QtOdeData _y, _y_old, _y_new
    cdef QtOdeFuncWrapper f

    cdef double new_norm, norm, t, dt_safe, t_int, dt_int
    cdef double rtol, atol, first_step, min_step, max_step
    cdef int nsteps
    cdef bint interpolate, failed

    cdef double[16] c
    cdef double[16,15] a
    cdef double[10] b
    cdef double[10] e
    cdef double[16,7] bi7

    def __init__(self, QtOdeFuncWrapper f,
                  rtol=1e-6, atol=1e-8, nsteps=1000,
                  first_step=0, min_step=0, max_step=0, interpolate=True):
        self.f = f
        self.atol = atol
        self.rtol = rtol
        self.nsteps = nsteps
        self.first_step = first_step
        self.min_step = min_step
        self.max_step = max_step
        self.interpolate = interpolate
        self.failed = False
        set_vern7_eff_ctes(self.a, self.b, self.c self.e, self.bi7)

    def step(self, double t):
        cdef int nsteps = 0
        while self.t < t and nsteps < self.nsteps:
            nsteps += 1
            dt = self.get_timestep(t)
            err = self.compute_step(dt, self._y_new)
            if err < 1:
                self.t_int = self.t
                self.dt_int = dt
                self.t += dt
                self._y_old.copy(self._y)
                self._y.copy(self._y_new)
                self.norm = self.new_norm
            self.recompute_safe_step(err, dt)
        if self.t < t + 1e-15 :
            self.failed = True
        elif self.t > t - 1e-15:
            self.prep_dense_out()
            self.interpolate(self, t, self._y_new)
            return self._y_new.raw()
        else:
            return self.y

    cdef void set_initial_value(self, y0, double t):
        self.norm = y0.norm()
        self.t = t
        self._y = qtodedata(y0)
        if not self.first_step:
            self.dt_safe = self.estimate_first_step(t, self._y)
        else:
            self.dt_safe = self.first_step

        #prepare_buffer
        for i in range(16):
            self.k[i] = self._y.empty_like()
        self.yp = self._y.empty_like()
        self.yp8 = self._y.empty_like()
        self.yp9 = self._y.empty_like()
        self._y_new = self._y.empty_like()
        self._y_old = self._y.empty_like()

    cdef void compute_step(self, double dt, QtOdeData out):
        cdef int i, j
        cdef double t = self.t

        self.f.call(self.k[0], t, self._y)

        for i in range(1, 8):
            self.yp.copy(self._y)
            self.accumulate(self.yp, self.a[i,:], dt, i)
            self.f.call(self.k[i], t + self.c[i], self.yp)

        self.yp8.copy(self._y)
        self.accumulate(self.yp8, self.a[8,:], dt, 8)
        self.f.call(self.k[8], t + self.c[8], self.yp8)

        self.yp9.copy(self._y)
        self.accumulate(self.yp9, self.a[9,:], dt, 9)
        self.f.call(self.k[9], t + self.c[9], self.yp9)

        out.copy(self._y)
        self.accumulate(out, self.b, dt, 10)

        return self.error(out)

    cdef double eigen_est(self):
        self.yp.copy(self.k[9])
        self.yp.inplace_add(self.k[8], -1)
        self.yp9.inplace_add(self.yp8, -1)
        return self.yp.norm() / self.yp9.norm()

    cdef double error(self, QtOdeData y_new):
        cdef int j
        self.error.zero()
        self.accumulate(self.error, self.e, dt, 10)
        self.new_norm = y_new.norm()
        return self.error.norm() / (self.atol +
                max(self.norm, self.new_norm) * self.rtol)

    cdef void prep_dense_out(self)
        cdef:
            double t = self.t_int
            double dt = self.dt_int

        for i in range(10, 16):
            self.yp.copy(self._y_old)
            for j in range(i):
                if self.a[i, j]:
                    self.yp.inplace_add(self.k[j], dt * self.a[i, j])
            self.f.call(self.k[i], t + self.c[i], self.yp)

    cdef void interpolate(self, double t, QtOdeData out):
        cdef:
             double tau = (t-t0)/dt
             int i, j
             double b_factor[16]
             double t0 = self.t_int
             double dt = self.dt_int
        for i in range(16):
            b_factor[i] = 0.
            for j in range(6,-1,-1):
                b_factor[i] += self.bi7[i, j]
                b_factor[i] *= tau
        out.copy(self._y_old)
        self.accumulate(out, b_factor[i], dt, 16)

    cdef void accumulate(self, QtOdeData target, double[:] factors,
                         double dt, int size):
        cdef int i
        for i in range(size):
            if factors[i]:
                target.inplace_add(self.k[j], dt * self.a[i, j])

    cdef double estimate_first_step(self, double t, QtOdeData y0):
        cdef double tol = self.atol + y0.norm() * self.rtol
        cdef double dt1, dt2, dt
        cdef double norm = y0.norm()
        self.k[0].zero()
        self.f.call(self.k[0], t, y0)
        # Good approximation for linear system. But not in a general case.
        dt1 = (tol * 5040 * norm**7)**(1/8) / self.k[0].norm()
        self.yp.copy(y0)
        self.yp.inplace_add(self.k[0], dt1 / 100)
        self.k[1].zero()
        self.f.call(self.k[1], t + dt1 / 100, self.yp)
        self.k[1].inplace_add(self.k[0], -1)
        dt2 = (tol * 5040* norm**3)**(1/8) / self.k[1].norm()**0.5
        dt = min(dt1, dt2)
        if self.max_step:
            dt = min(self.max_step, dt)
        if self.min_step:
            dt = max(self.min_step, dt)
        return dt

    cdef double get_timestep(double t):
        dt_needed = self.t - t
        if dt_needed <= self.dt_safe:
            return dt_needed
        if self.interpolate:
            return self.dt_safe
        return dt_needed / (int(dt_needed / self.dt_safe) + 1)

    cdef double recompute_safe_step(double err, double dt):
        self.dt_safe = (err/2)**(1/8) * dt

    @property
    def y(self):
        return self._y.raw()
