

import numpy as np
from ..core import Qobj, QobjEvo, spre, issuper


__all__ = ["Result"]


class Result:
    def __init__(self, e_ops_raw, e_ops, options, example_state, super):
        self._dims = example_state.dims
        self._type = example_state.type
        self._isherm = example_state.isherm
        self._isunitary = example_state.isunitary

        self.e_ops = e_ops_raw
        self._raw_e_ops = e_ops
        self._num_saved = 0
        self._states = []
        self.times = []
        self._expects = []
        self._last_state = None

        self._read_e_ops(super)
        self._read_options(options)

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

    def add(self, t, state):
        self.times.append(t)

        if self._store_states:
            self._states.append(state.copy())
        elif self._store_final_state:
            self._last_state = state.copy()

        for i, e_call in enumerate(self._e_ops):
            self._expects[i].append(e_call(t, state))

    def copy(self):
        return Run(self._raw_e_ops, self.options)

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
