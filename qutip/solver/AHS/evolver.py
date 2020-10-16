from ..evolver import Evolver, get_evolver
from .qevo import SolverQEvoAHS

class AHSEvolver(Evolver):
    def __init__(self, system, options, args, feedback_args):
        self.system = SolverQEvoAHS(system, options, args, feedback_args)
        self.options = options
        self._evolver = get_evolver(system, options, args, feedback_args)
        self._evolver.system = self.system

    def set(self, state, t0, options=None):
        self.options = options or self.options
        self._evolver.set(state, t0, self.options)
        self.system.resize(state)

    def run(self, tlist):
        """ Yield (t, state(t)) for t in tlist, must be `set` before. """
        t_prev = self._ode_solver.t
        state_prev = self.get_state()
        for t in tlist[1:]:
            yield t, self.step(t, False, True)

    def step(self, t, step=None, resize=False):
        """ Evolve to t, must be `set` before. """
        done = False
        while not done:
            self._evolver.step(t, step)
            state = self.get_state()
            if resize and not self.system.resize(state):
                pass
            else:
                done = True
        return state

    def resize(self):
        return self.system.resize(self.get_state())

    def get_state(self):
        self._evolver.get_state()

    def set_state(self, state0, t):
        self._evolver.set_state(state0, t)
        self.system.resize(state0)
