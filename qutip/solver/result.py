

import numpy as np
from ..core import Qobj, QobjEvo, spre, issuper


__all__ = ["Result", "MultiTrajResult", "MultiTrajResultAveraged"]


class Result:
    """
    Result for one trajectory of an solver evolution.

    Property
    --------
    states : list of Qobj
        Every state of the evolution

    final_state : Qobj
        Last state of the evolution

    expect : list
        list of list of expectation values
        expect[e_ops][t]

    times : list
        list of the times at which the expectation values and
        states where taken.
    """
    def __init__(self, e_ops, options, super_):
        self.e_ops = e_ops
        self.times = []

        self._raw_e_ops = e_ops
        self._states = []
        self._expects = []
        self._last_state = None
        self._super = super_

        self._read_e_ops(super_)
        self._read_options(options)
        self.collapse = None
        self.stats = {}

    def _read_e_ops(self, _super):
        self._e_ops_dict = False
        self._e_num = 0
        self._e_ops = []
        self._e_type = []

        if isinstance(self._raw_e_ops, (Qobj, QobjEvo)):
            e_ops = [self._raw_e_ops]
        elif isinstance(self._raw_e_ops, dict):
            self._e_ops_dict = self._raw_e_ops
            e_ops = [e for e in self._raw_e_ops.values()]
        elif callable(self._raw_e_ops):
            e_ops = [self._raw_e_ops]
        else:
            e_ops = self._raw_e_ops

        for e in e_ops:
            if isinstance(e, Qobj):
                if not issuper(e) and _super:
                    e = spre(e)
                self._e_ops.append(QobjEvo(e).expect)
                self._e_type.append(e.isherm)
            elif isinstance(e, QobjEvo):
                if not issuper(e.cte) and _super:
                    e = spre(e)
                self._e_ops.append(e.expect)
                self._e_type.append(e.isherm)
            elif callable(e):
                self._e_ops.append(e)
                self._e_type.append(False)
            self._expects.append([])

        self._e_num = len(e_ops)

    def _read_options(self, options):
        self._store_states = self._e_num == 0 or options['store_states']
        self._store_final_state = options['store_final_state']
        if self._super:
            self._normalize_outputs = options['normalize_output'] in ['dm', True, 'all']
        else:
            self._normalize_outputs = options['normalize_output'] in ['ket', True, 'all']

    def _normalize(self, state):
        if state.shape[1] == 1:
            return state * (1/state.norm())
        elif state.shape[1] == state.shape[0] and self._super:
            return state * (1/state.norm())
        else:
            # TODO add normalization for propagator evolution.
            pass

    def add(self, t, state):
        """
        Add a state to the results for the time t of the evolution.
        The state is expected to be a Qobj with the right dims.
        """
        self.times.append(t)
        # this is so we don't make a copy if normalize is
        # false and states are not stored
        state_norm = False
        if self._normalize_outputs:
            state_norm = self._normalize(state)

        if self._store_states:
            self._states.append(state_norm or state.copy())
        elif self._store_final_state:
            self._last_state = state_norm or state.copy()

        for i, e_call in enumerate(self._e_ops):
            self._expects[i].append(e_call(t, state_norm or state))

    def copy(self):
        return Result(self._raw_e_ops, self.options, self.super)

    @property
    def states(self):
        return self._states

    @property
    def final_state(self):
        if self._store_states:
            return self._states[-1]
        elif self._store_final_state:
            return self._last_state
        else:
            return None

    @property
    def expect(self):
        result = []
        for expect_vals, isreal  in zip(self._expects, self._e_type):
            es = np.array(expect_vals)
            if isreal:
                es = es.real
            result.append(es)
        if self._e_ops_dict:
            result = {e: result[n]
                      for n, e in enumerate(self._e_ops_dict.keys())}
        return result

    @property
    def num_expect(self):
        return self._e_num

    def __repr__(self):
        out = ""
        out += self.stats['solver'] + "\n"
        out += "solver : " + self.stats['method'] + "\n"
        out += "number of expect : {}\n".format(self._e_num)
        if self._store_states:
            out += "State saved\n"
        elif self._store_final_state:
            out += "Final state saved\n"
        else:
            out += "State not available\n"
        out += "times from {} to {} in {} steps\n".format(self.times[0],
                                                          self.times[-1],
                                                          len(self.times))
        return out


class MultiTrajResult:
    """
    Contain result of simulations with multiple trajectories.
    """
    def __init__(self, num_c_ops=0):
        """
        Parameters:
        -----------
        num_c_ops: int
            Number of collapses operator used in the McSolver
        """
        self.trajectories = []
        self._to_dm = True # MCsolve
        self.num_c_ops = num_c_ops
        self.tlist = None

    def add(self, one_traj):
        self.trajectories.append(one_traj)

    @property
    def runs_states(self):
        return [traj.states for traj in self.trajectories]

    @property
    def average_states(self):
        if self._to_dm:
            finals = [state.proj() for state in self.trajectories[0].states]
            for i in range(1, len(self.trajectories)):
                finals = [state.proj() + final for final, state
                          in zip(finals, self.trajectories[i].states)]
        else:
            finals = [state for state in self.trajectories[0].states]
            for i in range(1, len(self.trajectories)):
                finals = [state + final for final, state
                          in zip(finals, self.trajectories[i].states)]
        return [final / len(self.trajectories) for final in finals]

    @property
    def runs_final_states(self):
        return [traj.final_state for traj in self.trajectories]

    @property
    def average_final_state(self):
        if self._to_dm:
            final = sum(traj.final_state.proj() for traj in self.trajectories)
        else:
            final = sum(traj.final_state for traj in self.trajectories)
        return final / len(self.trajectories)

    @property
    def steady_state(self):
        avg = self.average_states
        return sum(avg) / len(avg)

    @property
    def average_expect(self):
        num_e = self.trajectories[0]._e_num
        _e_type = self.trajectories[0]._e_type
        _e_ops_dict = self.trajectories[0]._e_ops_dict
        avg = [np.mean(np.stack([traj._expects[i]
                    for traj in self.trajectories]), axis=0)
               for i in range(num_e)]

        result = []
        for expect_vals, isreal  in zip(avg, _e_type):
            if isreal:
                expect_vals = expect_vals.real
            result.append(expect_vals)

        if _e_ops_dict:
            result = {e: result[n]
                      for n, e in enumerate(_e_ops_dict.keys())}
        return result

    @property
    def std_expect(self):
        num_e = self.trajectories[0]._e_num
        _e_type = self.trajectories[0]._e_type
        _e_ops_dict = self.trajectories[0]._e_ops_dict
        avg = [np.std(np.stack([traj._expects[i]
                    for traj in self.trajectories]), axis=0)
               for i in range(num_e)]

        result = []
        for expect_vals, isreal  in zip(avg, _e_type):
            if isreal:
                expect_vals = expect_vals.real
            result.append(expect_vals)

        if _e_ops_dict:
            result = {e: result[n]
                      for n, e in enumerate(_e_ops_dict.keys())}
        return result

    @property
    def runs_expect(self):
        num_e = self.trajectories[0]._e_num
        _e_type = self.trajectories[0]._e_type
        _e_ops_dict = self.trajectories[0]._e_ops_dict
        avg = [np.stack([traj._expects[i] for traj in self.trajectories])
               for i in range(num_e)]

        result = []
        for expect_vals, isreal  in zip(avg, _e_type):
            if isreal:
                expect_vals = expect_vals.real
            result.append(expect_vals)

        if _e_ops_dict:
            result = {e: result[n]
                      for n, e in enumerate(_e_ops_dict.keys())}
        return result

    def expect_traj_avg(self, ntraj=-1):
        num_e = self.trajectories[0]._e_num
        _e_type = self.trajectories[0]._e_type
        _e_ops_dict = self.trajectories[0]._e_ops_dict
        avg = [np.mean(np.stack([traj._expects[i]
                    for traj in self.trajectories[:ntraj]]), axis=0)
               for i in range(num_e)]

        result = []
        for expect_vals, isreal  in zip(avg, _e_type):
            if isreal:
                expect_vals = expect_vals.real
            result.append(expect_vals)

        if _e_ops_dict:
            result = {e: result[n]
                      for n, e in enumerate(_e_ops_dict.keys())}
        return result

    def expect_traj_std(self, ntraj=-1):
        num_e = self.trajectories[0]._e_num
        _e_type = self.trajectories[0]._e_type
        _e_ops_dict = self.trajectories[0]._e_ops_dict
        avg = [np.std(np.stack([traj._expects[i]
                    for traj in self.trajectories[:ntraj]]), axis=0)
               for i in range(num_e)]

        result = []
        for expect_vals, isreal  in zip(avg, _e_type):
            if isreal:
                expect_vals = expect_vals.real
            result.append(expect_vals)

        if _e_ops_dict:
            result = {e: result[n]
                      for n, e in enumerate(_e_ops_dict.keys())}
        return result

    @property
    def collapse(self):
        return [traj.collapse for traj in self.trajectories]

    @property
    def col_times(self):
        out = []
        for col_ in self.collapse:
            col = list(zip(*col_))
            col = ([] if len(col) == 0 else col[0])
            out.append(col)
        return out

    @property
    def col_which(self):
        out = []
        for col_ in self.collapse:
            col = list(zip(*col_))
            col = ([] if len(col) == 0 else col[1])
            out.append(col)
        return out

    @property
    def photocurrent(self):
        cols = {}
        tlist = self.trajectories[0].times
        for traj in self.trajectories:
            for t, which in traj.collapse:
                if which in cols:
                    cols[which].append(t)
                else:
                    cols[which] = [t]
        mesurement = []
        for i in range(self.num_c_ops):
            mesurement += [(np.histogram(cols.get(i,[]), tlist)[0]
                          / np.diff(tlist) / len(self.trajectories))]
        return mesurement

    @property
    def run_stats(self):
        return self.trajectories[0].stats

    def __repr__(self):
        out = ""
        out += self.run_stats['solver'] + "\n"
        out += "solver : " + self.stats['method'] + "\n"
        out += "{} runs saved\n".format(self.num_traj)
        out += "number of expect : {}\n".format(self.trajectories[0]._e_num)
        if self.trajectories[0]._store_states:
            out += "Runs states saved\n"
        elif self.trajectories[0]._store_final_state:
            out += "Runs final state saved\n"
        else:
            out += "State not available\n"
        out += "times from {} to {} in {} steps\n".format(self.times[0],
                                                          self.times[-1],
                                                          len(self.times))
        return out

    @property
    def times(self):
        return self.trajectories[0].times

    @property
    def states(self):
        return self.average_states

    @property
    def expect(self):
        return self.average_expect

    @property
    def final_state(self):
        return self.average_final_state

    @property
    def num_traj(self):
        return len(self.trajectories)


class MultiTrajResultAveraged:
    """
    Contain result of simulations with multiple trajectories.
    """
    def __init__(self, num_c_ops=0):
        """
        Parameters:
        -----------
        num_c_ops: int
            Number of collapses operator used in the McSolver
        """
        self.trajectories = None
        self._sum_states = None
        self._sum_last_states = None
        self._sum_expect = None
        self._sum2_expect = None
        self._to_dm = True # MCsolve
        self.num_c_ops = num_c_ops
        self._num = 0
        self._collapse = []

    def add(self, one_traj):
        if self._num == 0:
            self.trajectories = one_traj
            if self._to_dm and one_traj.states:
                self._sum_states = [state.proj() for state in one_traj.states]
            else:
                self._sum_states = one_traj.states
            if self._to_dm and one_traj.final_state:
                self._sum_last_states = one_traj.final_state.proj()
            else:
                self._sum_last_states = one_traj.final_state
            self._sum_expect = [np.array(expect) for expect in one_traj._expects]
            self._sum2_expect = [np.array(expect)**2 for expect in one_traj._expects]
        else:
            if self._to_dm:
                if self._sum_states:
                    self._sum_states = [state.proj() + accu for accu, state
                                    in zip(self._sum_states, one_traj.states)]
                if self._sum_last_states:
                    self._sum_last_states += one_traj.final_state.proj()
            else:
                if self._sum_states:
                    self._sum_states = [state + accu for accu, state
                                    in zip(self._sum_states, one_traj.states)]
                if self._sum_last_states:
                    self._sum_last_states += one_traj.final_state
            if self._sum_expect:
                self._sum_expect = [np.array(one) + accu for one, accu in
                                    zip(one_traj._expects, self._sum_expect)]
                self._sum2_expect = [np.array(one)**2 + accu for one, accu in
                                     zip(one_traj._expects, self._sum2_expect)]
        self._collapse.append(one_traj.collapse)
        self._num += 1

    @property
    def runs_states(self):
        return None

    @property
    def average_states(self):
        return [final / self._num for final in self._sum_states]

    @property
    def runs_final_states(self):
        return None

    @property
    def average_final_state(self):
        return self._sum_last_states / self._num

    @property
    def steady_state(self):
        avg = self._sum_states
        return sum(avg) / len(avg)

    @property
    def average_expect(self):
        num_e = self.trajectories._e_num
        _e_type = self.trajectories._e_type
        _e_ops_dict = self.trajectories._e_ops_dict
        avg = [_sum / self._num for _sum in self._sum_expect]

        result = []
        for expect_vals, isreal  in zip(avg, _e_type):
            if isreal:
                expect_vals = expect_vals.real
            result.append(expect_vals)

        if _e_ops_dict:
            result = {e: result[n]
                      for n, e in enumerate(_e_ops_dict.keys())}
        return result

    @property
    def std_expect(self):
        num_e = self.trajectories._e_num
        _e_type = self.trajectories._e_type
        _e_ops_dict = self.trajectories._e_ops_dict
        avg = [_sum / self._num for _sum in self._sum_expect]
        avg2 = [_sum2 / self._num for _sum2 in self._sum2_expect]
        std = [np.sqrt(a2 - a*a) for a, a2 in zip(avg, avg2)]

        result = []
        for expect_vals, isreal  in zip(std, _e_type):
            if isreal:
                expect_vals = expect_vals.real
            result.append(expect_vals)

        if _e_ops_dict:
            result = {e: result[n]
                      for n, e in enumerate(_e_ops_dict.keys())}
        return result

    @property
    def runs_expect(self):
        return None

    def expect_traj_avg(self, ntraj=-1):
        return None

    def expect_traj_std(self, ntraj=-1):
        return None

    @property
    def collapse(self):
        return self._collapse

    @property
    def col_times(self):
        out = []
        for col_ in self.collapse:
            col = list(zip(*col_))
            col = ([] if len(col) == 0 else col[0])
            out.append(col)
        return out

    @property
    def col_which(self):
        out = []
        for col_ in self.collapse:
            col = list(zip(*col_))
            col = ([] if len(col) == 0 else col[1])
            out.append(col)
        return out

    @property
    def photocurrent(self):
        cols = {}
        tlist = self.trajectories.times
        for collapses in self.collapse:
            for t, which in collapses:
                if which in cols:
                    cols[which].append(t)
                else:
                    cols[which] = [t]
        mesurement = []
        for i in range(self.num_c_ops):
            mesurement += [(np.histogram(cols.get(i,[]), tlist)[0]
                          / np.diff(tlist) / self._num)]
        return mesurement

    @property
    def run_stats(self):
        return self.trajectories.stats

    def __repr__(self):
        out = ""
        out += self.run_stats['solver'] + "\n"
        out += "solver : " + self.stats['method'] + "\n"
        out += "{} trajectories averaged\n".format(self.num_traj)
        out += "number of expect : {}\n".format(self.trajectories._e_num)
        if self.trajectories._store_states:
            out += "States saved\n"
        elif self.trajectories._store_final_state:
            out += "Final state saved\n"
        else:
            out += "State not available\n"
        out += "times from {} to {} in {} steps\n".format(self.times[0],
                                                          self.times[-1],
                                                          len(self.times))
        return out

    @property
    def times(self):
        return self.trajectories.times

    @property
    def states(self):
        return self.average_states

    @property
    def expect(self):
        return self.average_expect

    @property
    def final_state(self):
        return self.average_final_state

    @property
    def num_traj(self):
        return self._num
