#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False, nonecheck=False
from .verner7efficient cimport vern7_eff_cte
from .wrapper cimport QtOdeData, QtOdeFuncWrapper
from .wrapper import qtodedata

cdef class vern7:
    cdef list k # TODO make sure not type coversion are done
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
    cdef double[16][15] a
    cdef double[10] b
    cdef double[10] e
    cdef double[16][15] bi7

    def __init__(vern7 self, QtOdeFuncWrapper f,
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
        cdef vern7_eff_cte tableau = vern7_eff_cte()
        self.a = tableau.a
        self.b = tableau.b
        self.c = tableau.c
        self.e = tableau.e
        self.bi7 = tableau.bi7
        self.k = []

    cpdef step(vern7 self, double t):
        cdef int nsteps = 0
        while self.t < t and nsteps < self.nsteps:
            nsteps += 1
            dt = self.get_timestep(t)
            err = self.compute_step(dt, self._y_new)
            # print(t, self.t, dt, err, self.dt_safe)
            if err < 1:
                self.t_int = self.t
                self.dt_int = dt
                self.t += dt
                self._y_old.copy(self._y)
                self._y.copy(self._y_new)
                self.norm = self.new_norm
            self.recompute_safe_step(err, dt)
            print(self.t, dt, err, self.dt_safe)
        if self.t < t - 1e-15 :
            self.failed = True
        elif self.t > t + 1e-15:
            self.prep_dense_out()
            self.interpolate_step(t, self._y_new)
            return self._y_new.raw()
        else:
            return self.y

    def successful(self):
        return not self.failed

    def state(self):
        print(self.t, self.t_int, self.dt_int, self.dt_safe)
        print(self.y, self._y_old.raw())
        print(self.norm, self.new_norm)
        print(self.eigen_est())

    cpdef void set_initial_value(self, y0, double t):
        self.t = t
        self._y = qtodedata(y0)
        self.norm = self._y.norm()

        #prepare_buffer
        for i in range(16):
            self.k.append(self._y.empty_like())
        self.yp = self._y.empty_like()
        self.yp8 = self._y.empty_like()
        self.yp9 = self._y.empty_like()
        self._y_new = self._y.empty_like()
        self._y_old = self._y.empty_like()

        if not self.first_step:
            self.dt_safe = self.estimate_first_step(t, self._y)
        else:
            self.dt_safe = self.first_step

    cdef double compute_step(self, double dt, QtOdeData out):
        cdef int i, j
        cdef double t = self.t

        self.f.call(self.k[0], t, self._y)

        for i in range(1, 8):
            self.yp.copy(self._y)
            self.accumulate(self.yp, self.a[i], dt, i)
            self.f.call((<QtOdeData> self.k[i]), t + self.c[i], self.yp)

        self.yp8.copy(self._y)
        self.accumulate(self.yp8, self.a[8], dt, 8)
        self.f.call((<QtOdeData> self.k[8]), t + self.c[8], self.yp8)

        self.yp9.copy(self._y)
        self.accumulate(self.yp9, self.a[9], dt, 9)
        self.f.call((<QtOdeData> self.k[9]), t + self.c[9], self.yp9)

        out.copy(self._y)
        self.accumulate(out, self.b, dt, 10)

        return self.error(out, dt)

    cdef double eigen_est(self):
        self.yp.copy(self.k[9])
        self.yp.inplace_add(self.k[8], -1)
        self.yp9.inplace_add(self.yp8, -1)
        return self.yp.norm() / self.yp9.norm()

    cdef double error(self, QtOdeData y_new, double dt):
        cdef int j
        self.yp.zero()
        self.accumulate(self.yp, self.e, dt, 10)
        self.new_norm = y_new.norm()
        return self.yp.norm() / (self.atol +
                max(self.norm, self.new_norm) * self.rtol)

    cdef void prep_dense_out(self):
        cdef:
            double t = self.t_int
            double dt = self.dt_int

        for i in range(10, 16):
            self.yp.copy(self._y_old)
            for j in range(i):
                if self.a[i][j]:
                    self.yp.inplace_add((<QtOdeData> self.k[j]), dt * self.a[i][j])
            self.f.call(self.k[i], t + self.c[i], self.yp)

    cdef void interpolate_step(self, double t, QtOdeData out):
        cdef:
            int i, j
            double b_factor[16]
            double t0 = self.t_int
            double dt = self.dt_int
            double tau = (t-t0)/dt
        for i in range(16):
            b_factor[i] = 0.
            for j in range(6,-1,-1):
                b_factor[i] += self.bi7[i][j]
                b_factor[i] *= tau
        out.copy(self._y_old)
        self.accumulate(out, b_factor, dt, 16)

    cdef void accumulate(self, QtOdeData target, double[:] factors,
                         double dt, int size):
        cdef int i
        for i in range(size):
            if factors[i]:
                target.inplace_add((<QtOdeData> self.k[i]), dt * factors[i])

    cdef double estimate_first_step(self, double t, QtOdeData y0):
        cdef double tol = self.atol + y0.norm() * self.rtol
        cdef double dt1, dt2, dt
        cdef double norm = y0.norm()
        (<QtOdeData> self.k[0]).zero()
        self.f.call((<QtOdeData> self.k[0]), t, y0)
        # Good approximation for linear system. But not in a general case.
        dt1 = (tol * 5040 * norm**7)**(1/8) / (<QtOdeData> self.k[0]).norm()
        # print(dt1, (<QtOdeData> self.k[0]).norm())
        self.yp.copy(y0)
        self.yp.inplace_add((<QtOdeData> self.k[0]), dt1 / 100)
        (<QtOdeData> self.k[1]).zero()
        self.f.call((<QtOdeData> self.k[1]), t + dt1 / 100, self.yp)
        (<QtOdeData> self.k[1]).inplace_add(self.k[0], -1)
        dt2 = ((tol * 5040* norm**3)**(1/8) /
               ((<QtOdeData> self.k[1]).norm() / dt1 * 100)**0.5)
        # print(dt2, (<QtOdeData> self.k[1]).norm())
        dt = min(dt1, dt2)
        if self.max_step:
            dt = min(self.max_step, dt)
        if self.min_step:
            dt = max(self.min_step, dt)
        return dt

    cdef double get_timestep(self, double t):
        if self.interpolate:
            return self.dt_safe
        dt_needed = t - self.t
        if dt_needed <= self.dt_safe:
            return dt_needed
        return dt_needed / (int(dt_needed / self.dt_safe) + 1)

    cdef double recompute_safe_step(self, double err, double dt):
        cdef factor = 0.
        if err == 0:
            factor = 10
        factor = 0.9*err**(-1/8)
        factor = min(10, factor)
        factor = max(0.2, factor)

        self.dt_safe = dt * factor
        if self.max_step:
            self.dt_safe = min(self.max_step, self.dt_safe)
        if self.min_step:
            self.dt_safe = max(self.min_step, self.dt_safe)

    @property
    def y(self):
        return self._y.raw()
