from ..evolver import Evolver, get_evolver
from .qevo import SolverQEvoAHS

class AHSEvolver(Evolver):
    def __init__(self, system, options, args, feedback_args):
        self.system = SolverQEvoAHS(system, options.rhs, args, feedback_args)
        self.options = options.ode
        self._evolver = get_evolver(system, options, args, feedback_args)
        self._evolver.system = self.system

    def set(self, state, t0, options=None):
        self.options = options or self.options
        self._evolver.set(state, t0, self.options)
        self.system.resize(state)

    def run(self, tlist):
        """ Yield (t, state(t)) for t in tlist, must be `set` before. """
        t_prev = self._evolver.t
        state_prev = self.get_state()
        for t in tlist[1:]:
            yield t, self.step(t, False, True)

    def step(self, t, step=None, resize=False):
        """ Evolve to t, must be `set` before. """
        tries = 0
        y_old = self.get_state()
        t_old = self.t
        while self.t < t and tries < 10:
            tries += 1
            state = self._evolver.step(t, step=1)
            if resize and not self.system.resize(state):
                self.set_state(y_old, t_old)
            else:
                t_old = self.t
                y_old = state.copy()
                tries = 0
                if step:
                    break
        return y_old

    def backstep(self, t, t_old, y_old):
        return self._evolver.backstep(t, t_old, y_old)

    def resize(self):
        return self.system.resize(self.get_state())

    def get_state(self):
        return self._evolver.get_state()

    def set_state(self, state0, t):
        self._evolver.set_state(state0, t)
        self.system.resize(state0)

    @property
    def t(self):
        return self._evolver.t
