__all__ = ['IntegratorScipyZvode', 'IntegratorScipyDop853',
           'IntegratorScipylsoda']

import numpy as np
from scipy.integrate import ode
from qutip.core import data as _data
from qutip.core.data.reshape import column_unstack_dense, column_stack_dense
from ..integrator import integrator_collection, IntegratorException
import warnings


class Integrator:
    """ A wrapper around ODE solvers.
    Ensure a common interface for Solver usage.
    Take and return states as :class:`qutip.core.data.Data`.

    Methods
    -------
    prepare()
        Prepare the ODE solver.

    step(t, copy)
        Evolve to t, must be `prepare` before.
        return the pair t, state.

    get_state()
        Obtain the state of the solver as a pair t, state

    set_state(t, state)
        Set the state of the ODE solver.

    run(state, tlist)
        Yield (t, state(t)) for t in tlist, must be `set` before.

    update_args(args)
        Change the argument of the active system.

    one_step(t, copy):
        Advance up to t by one internal solver step.
        Should be able to retreive the state at any time between the present
        time and the resulting time using `backstep`.
        return the pair t, state.

    backstep(t, copy):
        Retreive the state at time t, with t smaller than present ODE time.
        The time t will always be between the last calls of `one_step`.
        return the pair t, state.

    Parameters
    ----------
    sytem: qutip.QobjEvo_base
        Input system controling the evolution.

    options: qutip.SolverOptions
        Options for the solver.
    """
    used_options = []
    description = ""

    def __init__(self, system, options):
        self.system = system
        self._stats = {}
        self.options = options
        self.prepare()

    def prepare(self):
        """
        Initialize the solver
        """
        raise NotImplementedError

    def step(self, t, step=False, copy=True):
        """
        Evolve to t, must be `prepare` before.
        return the pair (t, state[.copy()]).

        If `step` advance up to t by one internal solver step.
        Should be able to retreive the state at any time between the present
        time and the resulting time with subsequent calls.
        """
        raise NotImplementedError

    def get_state(self, copy=True):
        """
        Obtain the state of the solver as a pair t, state
        """
        raise NotImplementedError

    def set_state(self, t, state0):
        """
        Set the state of the ODE solver.
        """
        raise NotImplementedError

    def run(self, tlist):
        """
        Yield (t, state(t)) for t in tlist, must be `set` before.
        """
        for t in tlist[1:]:
            yield self.step(t, False)

    def update_args(self, args):
        """
        Change the argument of the active system.
        """
        self.system.arguments(args)

    def update_feedback(self, what, data):
        if isinstance(self.system, QobjEvo_With_FeedBack):
            self.system.update_feedback(what, data)

    @property
    def stats(self):
        """
        Return statistic of the evolution as a dict
        """
        if not hasattr(self, "_stats"):
            self._stats = {}
        try:
            self._stats.update(self.system.stats)
        except:
            pass
        return self._stats


class IntegratorScipyZvode(Integrator):
    """
    Integrator using Scipy `ode` interface for netlib zvode integrator.
    """
    used_options = ['atol', 'rtol', 'nsteps', 'method', 'order',
                    'first_step', 'max_step', 'min_step']
    description = "scipy.integrate.ode using zvode integrator"
    long_description = """
Ordinary Differential Equation solver by netlib (http://www.netlib.org/odepack)
with python interface by scipy. Support 'Adams' and 'BDF' methods
for non-stiff and stiff systems respectively."""

    def prepare(self):
        """
        Initialize the solver
        """
        self._ode_solver = ode(self.mul_np_vec)
        opt = {key: self.options.ode[key]
               for key in self.used_options
               if key in self.options.ode}
        self._ode_solver.set_integrator('zvode', **opt)
        self.name = "scipy zvode " + opt["method"]

    def mul_np_vec(self, t, vec):
        """ Interface between scipy which use numpy
        and the driver, which use data.
        """
        state = _data.dense.fast_from_numpy(vec)
        column_unstack_dense(state, self._size, inplace=True)
        out = _data.dense.zeros(state.shape[0], state.shape[1], state.fortran)
        out = self.system.matmul_data(t, state, out)
        column_stack_dense(out, inplace=True)
        return out.as_ndarray().ravel()

    def set_state(self, t, state0):
        """
        Set the state of the ODE solver.
        """
        self._mat_state = state0.shape[1] > 1
        self._size = state0.shape[0]
        self._ode_solver.set_initial_value(
            _data.column_stack(state0).to_array().ravel(),
            t
        )

    def get_state(self, copy=True):
        """
        Obtain the state of the solver as a pair t, state
        """
        t = self._ode_solver.t
        if self._mat_state:
            state = _data.column_unstack_dense(
                _data.dense.Dense(self._ode_solver._y, copy=False),
                self._size,
                inplace=True)
        else:
            state = _data.dense.Dense(self._ode_solver._y, copy=False)
        return t, state.copy() if copy else state

    def step(self, t, step=False, copy=True):
        """ Evolve to t. """
        if step:
            self._one_step(t)
        elif self._ode_solver.t < t:
            self._ode_solver.integrate(t)
        elif self._ode_solver.t > t:
            self._backstep(t)
        self._check_failed_integration()
        return self.get_state(copy)

    def _one_step(self, t):
        """
        Advance up to t by one internal solver step.
        Should be able to retreive the state at any time between the present
        time and the resulting time using `backstep`.
        """
        # integrate(t, step=True) ignore the time and advance one step.
        # Here we want to advance up to t doing maximum one step.
        # So we check if a new step is really needed.
        self.back = self.get_state(copy=True)
        t_front = self._ode_solver._integrator.rwork[12]
        t_ode = self._ode_solver.t
        if t > t_front and t_ode >= t_front:
            # The state is at t_front, do a step
            self._ode_solver.integrate(t, step=True)
            t_front = self._ode_solver._integrator.rwork[12]
        elif t > t_front:
            # The state is at a time before t_front, advance to t_front
            t = t_front
        else:
            # t is inside the already computed integration range.
            pass
        if t_front >= t:
            self._ode_solver.integrate(t)

    def _backstep(self, t):
        """
        Retreive the state at time t, with t smaller than present ODE time.
        The time t will always be between the last calls of `one_step`.
        return the pair t, state.
        """
        # zvode can integrate with time lower than the most recent time
        # but not over all the step interval.
        # About 90% of the last step interval?
        # Scipy raise a warning, not an Error.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self._ode_solver.integrate(t)
        if not self._ode_solver.successful():
            self.set_state(*self.back)
            self._ode_solver.integrate(t)

    def _check_failed_integration(self):
        if self._ode_solver.successful():
            return
        messages = {
            -1: 'Excess work done on this call. Try to increasing '
                'the nsteps parameter in the Options class',
            -2: 'Excess accuracy requested. (Tolerances too small.)',
            -3: 'Illegal input detected.',
            -4: 'Repeated error test failures. (Check all input.)',
            -5: 'Repeated convergence failures. (Perhaps bad'
                ' Jacobian supplied or wrong choice of MF or tolerances.)',
            -6: 'Error weight became zero during problem. (Solution'
                ' component i vanished, and ATOL or ATOL(i) = 0.)'
        }
        raise IntegratorException(messages[self._ode_solver._integrator.istate])


class IntegratorScipyDop853(Integrator):
    """
    Integrator using Scipy `ode` interface with dop853 integrator.
    """
    description = "scipy.integrate.ode using dop853 integrator"
    long_description = """
Eight order runge-kutta method by Dormand & Prince. Python interface by scipy.
Fortran implementation from [E. Hairer, S.P. Norsett and G. Wanner,
Solving Ordinary Differential Equations i. Nonstiff Problems. 2nd edition.
Springer Series in Computational Mathematics, Springer-Verlag (1993)]"""
    used_options = ['atol', 'rtol', 'nsteps', 'first_step', 'max_step',
                    'ifactor', 'dfactor', 'beta']

    def prepare(self):
        """
        Initialize the solver
        """
        self._ode_solver = ode(self.mul_np_vec)
        opt = {key: self.options.ode[key]
               for key in self.used_options
               if key in self.options.ode}
        self._ode_solver.set_integrator('dop853', **opt)
        self.name = "scipy ode dop853"

    def mul_np_vec(self, t, vec):
        """ Interface between scipy which use numpy
        and the driver, which use data.
        """
        state = _data.dense.fast_from_numpy(vec.view(np.complex128))
        column_unstack_dense(state, self._size, inplace=True)
        out = _data.dense.zeros(state.shape[0], state.shape[1], state.fortran)
        out = self.system.matmul_data(t, state, out)
        column_stack_dense(out, inplace=True)
        return out.as_ndarray().ravel().view(np.float64)

    def step(self, t, step=False, copy=True):
        """
        Evolve to t. Or for one step.
        """
        if step:
            # Scipy DOP853 does not have a step function.
            # It has a safe step lengh, but can be 0 if unknown.
            dt = self._ode_solver._integrator.work[6]
            if dt:
                t = min(self._ode_solver.t + dt, t)

        if self._ode_solver.t > t:
            # While DOP853 support changing the direction of the integration,
            # it does not do so efficiently. We do it manually.
            self._ode_solver._integrator.work[6] *= -1
            self._ode_solver.integrate(t)
            self._ode_solver._integrator.work[6] *= -1
        elif self._ode_solver.t < t:
            self._ode_solver.integrate(t)

        self._check_failed_integration()
        return self.get_state(copy)

    def get_state(self, copy=True):
        """
        Obtain the state of the solver as a pair t, state
        """
        t = self._ode_solver.t
        if self._mat_state:
            state = _data.column_unstack_dense(
                _data.dense.Dense(self._ode_solver._y.view(np.complex128),
                                  copy=False),
                self._size,
                inplace=True)
        else:
            state = _data.dense.Dense(self._ode_solver._y.view(np.complex128),
                                      copy=False)
        return t, state.copy() if copy else state

    def set_state(self, t, state0):
        """
        Set the state of the ODE solver.
        """
        self._mat_state = state0.shape[1] > 1
        self._size = state0.shape[0]
        self._ode_solver.set_initial_value(
            _data.column_stack(state0).to_array().ravel().view(np.float64),
            t
        )

    def _check_failed_integration(self):
        if self._ode_solver.successful():
            return
        messages = {
            -1: 'input is not consistent',
            -2: 'larger nsteps is needed, Try to increase the nsteps '
                'parameter in the Options class.',
            -3: 'step size becomes too small. Try increasing tolerance',
            -4: 'problem is probably stiff (interrupted), try "bdf" '
                'method instead',
        }
        raise IntegratorException(messages[self._ode_solver._integrator.istate])


class IntegratorScipylsoda(IntegratorScipyDop853):
    """
    Integrator using Scipy `ode` interface of lsoda integrator from netlib.
    """
    used_options = ['atol', 'rtol', 'nsteps', 'max_order_ns', 'max_order_s',
                    'first_step', 'max_step', 'min_step']
    description = "scipy.integrate.ode using lsoda integrator"
    long_description = """
Ordinary Differential Equation solver by netlib (http://www.netlib.org/odepack)
with python interface by scipy. Automatically choose between 'Adams' and 'BDF'
methods to solve both stiff and non-stiff systems."""

    def prepare(self):
        """
        Initialize the solver
        """
        self._ode_solver = ode(self.mul_np_vec)
        opt = {key: self.options.ode[key]
               for key in self.used_options
               if key in self.options.ode}
        self._ode_solver.set_integrator('lsoda', **opt)
        self.name = "scipy lsoda"

    def step(self, t, step=False, copy=True):
        """ Evolve to t. """
        if step:
            self._one_step(t)
        elif self._ode_solver.t < t:
            self._ode_solver.integrate(t)
        elif self._ode_solver.t > t:
            self._backstep(t)
        self._check_failed_integration()
        return self.get_state(copy)

    def _one_step(self, t):
        """
        Advance up to t by one internal solver step.
        Should be able to retreive the state at any time between the present
        time and the resulting time using `backstep`.
        """
        # integrate(t, step=True) ignore the time and advance one step.
        # Here we want to advance up to t doing maximum one step.
        # So we check if a new step is really needed.
        self.back = self.get_state(copy=True)
        t_front = self._ode_solver._integrator.rwork[12]
        t_ode = self._ode_solver.t
        if t > t_front and t_ode >= t_front:
            # The state is at t_front, do a step
            self._ode_solver.integrate(t, step=True)
            t_front = self._ode_solver._integrator.rwork[12]
        elif t > t_front:
            # The state is at a time before t_front, advance to t_front
            t = t_front
        else:
            # t is inside the already computed integration range.
            pass
        if t_front >= t:
            self._ode_solver.integrate(t)

    def _backstep(self, t):
        """
        Retreive the state at time t, with t smaller than present ODE time.
        The time t will always be between the last calls of `one_step`.
        return the pair t, state.
        """
        # like zvode, lsoda can step with time lower than the most recent.
        # But not all the step interval.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self._ode_solver.integrate(t)
        if not self._ode_solver.successful():
            self.set_state(*self.back)
            self._ode_solver.integrate(t)

    def _check_failed_integration(self):
        if self._ode_solver.successful():
            return
        messages = {
            -1: "Excess work done on this call."
                "Try to increase the nsteps parameter in the Options class.",
            -2: "Excess accuracy requested (tolerances too small).",
            -3: "Illegal input detected (internal error).",
            -4: "Repeated error test failures (internal error).",
            -5: "Repeated convergence failures (perhaps bad Jacobian or tolerances).",
            -6: "Error weight became zero during problem.",
            -7: "Internal workspace insufficient to finish (internal error)."
        }
        raise IntegratorException(messages[self._ode_solver._integrator.istate])


integrator_collection.add_method(IntegratorScipyZvode, keys=['adams', 'bdf'],
                                 solver=['sesolve', 'mesolve', 'mcsolve'],
                                 use_QobjEvo_matmul=True, time_dependent=True)

integrator_collection.add_method(IntegratorScipyDop853, keys=['dop853'],
                                 solver=['sesolve', 'mesolve', 'mcsolve'],
                                 use_QobjEvo_matmul=True, time_dependent=True)

integrator_collection.add_method(IntegratorScipylsoda, keys=['lsoda'],
                                 solver=['sesolve', 'mesolve', 'mcsolve'],
                                 use_QobjEvo_matmul=True, time_dependent=True)
