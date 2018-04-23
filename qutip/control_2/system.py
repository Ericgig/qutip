# -*- coding: utf-8 -*-
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2014 and later, Alexander J G Pitchford
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

# @author: Alexander Pitchford
# @author: Eric Giguère


import os
import warnings
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
# QuTiP
from qutip import Qobj
from qutip.sparse import sp_eigs, _dense_eigs
import qutip.settings as settings
# QuTiP logging
import qutip.logging_utils as logging
logger = logging.get_logger()
# QuTiP control modules
import qutip.control.errors as errors
import qutip.control.tslotcomp as tslotcomp
import qutip.control.fidcomp as fidcomp
import qutip.control.propcomp as propcomp
import qutip.control.symplectic as sympl
import qutip.control.dump as qtrldump

DEF_NUM_TSLOTS = 10
DEF_EVO_TIME = 1.0





class dynamics:
    """
    This class compute the error and gradient for the GRAPE systems.*

    * Other object do the actual computation to cover the multiple situations.

    methods:
        cost(x):
            x: np.ndarray, state of the pulse
            return 1-fidelity

        gradient(x):
            x: np.ndarray, state of the pulse
            return the error gradient

        control_system():


    """

    def __init__(self, initial, target, H, ctrl,
                 times=None, tau=None, T=0, t_step=0, _num_x=0, _filter=None,
                 phase,
                 prop_method = None,
                 **kwarg):
        self.initial = initial      # Initial state/rho/operator as Qobj
        self.target = target        # Target state/rho/operator as Qobj
        self.drift_dyn_gen = H      # Hamiltonians or Liouvillian as a Qobj
        self.ctrl_dyn_gen = ctrl    # Control operator [Qobj]
        self._num_ctrls = len(ctrl)

        if _filter in None:
            self.filter = filters.pass_througth()
        else:
            self.filter = _filter

        self._x_shape, self.time = self.filter.init_timeslots(times, tau, T,
                                                    t_step, _num_x, _num_ctrls)
        self._num_tslots = len(self.time)-1
        self._evo_time = self.time[-1]
        if np.allclose(np.diff(self.time), self.time[1]-self.time[0]):
            self._tau = self.time[1]-self.time[0]
        else:
            self._tau = np.diff(self.time)
        # state and gradient before filter
        self._x = np.zeros(self._x_shape)
        self.gradient_x = np.zeros(self._x_shape)
        # state and gradient after filter
        self._ctrl_amps = np.zeros((self._num_tslots, self._num_ctrls))
        self._gradient_u = np.zeros((self._num_tslots, self._num_ctrls))

        self._set_memory_optimizations(**kwarg)

        if isinstance(initial, np.ndarray):
            self._initial = initial
            self._target = target
        elif isinstance(initial, Qobj) and initial.isoper:
            self._initial = matrice(initial, dense=self.oper_dense)
            self._target = matrice(target, dense=self.oper_dense)
        else:
            self._initial = np.ndarray(initial.data)
            self._target = np.ndarray(target.data)

        if not self.cache_drift_at_T:
            self._drift_dyn_gen = H # ----- Not implemented yet ----- ?
        else:
            if not H.const:
                self._drift_dyn_gen = [matrice(H(t), dense=self.oper_dense)
                                       for t in self.time]
            else:
                self._drift_dyn_gen = np.ndarray(
                                    [matrice(H, dense=self.oper_dense)])
            for mat in self._drift_dyn_gen:
                mat.method = prop_method
                mat.fact_mat_round_prec = fact_mat_round_prec
                mat._mem_eigen_adj = self.cache_dyn_gen_eigenvectors_adj
                mat._mem_prop = self.cache_prop
                mat.epsilon = epsilon
                if self.cache_phased_dyn_gen:
                    mat = self._apply_phase(mat)

        if not self.cache_ctrl_at_T:
            self._ctrl_dyn_gen = ctrl # ----- Not implemented yet ----- ?
        elif all((ctr.const for ctr in ctrl)):
            self._ctrl_dyn_gen = np.ndarray(
                                    [[matrice(ctr(0), dense=self.oper_dense)]
                                    for ctr in ctrl])
        else:
            self._ctrl_dyn_gen = np.ndarray(
                                    [[matrice(ctr(t), dense=self.oper_dense)
                                   for t in self.time] for ctr in ctrl]

        if _tslotcomp is None:
            self.tslotcomp =  tslotcomp.TSlotCompUpdateAll(self)
        else:
            self.tslotcomp = _tslotcomp
        self.tslotcomp.set(self)

        if _fidcomp is None:
            self.costcomp =  fidcomp.FidCompTraceDiff(self)
        elif isinstance(_fidcomp, list):
            self.costcomp = _fidcomp
        else:
            self.costcomp = [_fidcomp]
        for cost in self.costcomp:
            cost.set(self)

        # computation objects
        self._dyn_gen = []         # S[t] = H_0[t] + u_i[t]*H_i
        self._prop = []            # U[t] exp(-i*S[t])
        self._prop_grad = [[]]     # d U[t] / du_i
        self._fwd_evo = []         # U[t]U[t-dt]...U[dt]U[0] /initial
        self._onwd_evo = []        # /target U[T]U[T-dt]...U[t+dt]U[t]


        # These internal attributes will be of the internal operator data type
        # used to compute the evolution
        # Note this maybe ndarray, Qobj or some other depending on oper_dtype

        # self._phased_ctrl_dyn_gen = None
        # self._dyn_gen_phase = None
        # self._phase_application = None
        # self._phased_dyn_gen = None
        # self._onto_evo_target = None
        # self._onto_evo = None

    def _set_memory_optimizations(self, memory_optimization=0,
                                  cache_dyn_gen_eigenvectors_adj=None,
                                  cache_phased_dyn_gen=None,
                                  sparse_eigen_decomp=None,
                                  cache_drift_at_T=None,
                                  cache_prop_grad=None,
                                  cache_ctrl_at_T=None,
                                  cache_prop=None,
                                  oper_dtype=None,
                                  **kwarg):
        """
        Set various memory optimisation attributes based on the
        memory_optimization attribute.
        """

        if oper_dtype is None:
            self._choose_oper_dtype()
        else self.oper_dtype = oper_dtype

        if cache_phased_dyn_gen is None:
            self.cache_phased_dyn_gen = memory_optimization == 0
        else:
            self.cache_phased_dyn_gen = cache_phased_dyn_gen

        if cache_prop_grad is None:
            iself.cache_prop_grad = memory_optimization == 0
        else:
            self.cache_prop_grad = cache_prop_grad

        if cache_dyn_gen_eigenvectors_adj is None:
            self.cache_dyn_gen_eigenvectors_adj = memory_optimization == 0
        else:
            self.cache_dyn_gen_eigenvectors_adj = cache_dyn_gen_eigenvectors_adj

        if sparse_eigen_decomp is None:
            self.sparse_eigen_decomp = memory_optimization > 1
        else:
            self.sparse_eigen_decomp = sparse_eigen_decomp

        # If the drift operator depends on time, precompute for each t
        if cache_drift_at_T is None:
            self.cache_drift_at_T = True # memory_optimization > 1
        else:
            self.cache_drift_at_T = cache_drift_at_T

        # If one of the ctrl operators depend on time, precompute for each t
        if cache_ctrl_at_T is None:
            self.cache_ctrl_at_T = True # memory_optimization > 1
        else:
            self.cache_ctrl_at_T = cache_drift_at_T

        if cache_prop is None:
            self.cache_prop = memory_optimization == 0
        else:
            self.cache_prop = cache_prop


    def clean(self):
        """Remove object saved but not used during computation."""
        pass


    ### -------------------------- Computation part ---------------------------
    def error(self, x):
        if not np.allclose(self.x_ == x):
            self._compute_state(x)
        return self.error

    def gradient(self, x):
        if not np.allclose(self.x_ == x):
            self._compute_state(x)
        return self.gradient_x

    def _compute_state(self, x):
        """For a state x compute the cost and grandient"""
        self.x_ = x
        self._ctrl_amps = self.filter(x)
        self._compute_gen()
        self.tslotcomp.compute_evolution()
        for costs in self.costcomp:
            error, gradient_u_cost = costs(forward, backward)
            self.error += error
            gradient_u += gradient_u_cost
        self.gradient_x = self.filter.reverse(gradient_u)

    def _apply_phase(self, dg):
        return self._prephase * dg * self._postphase









def _is_string(var):
    try:
        if isinstance(var, basestring):
            return True
    except NameError:
        try:
            if isinstance(var, str):
                return True
        except:
            return False
    except:
        return False

    return False

def _check_ctrls_container(ctrls):
    """
    Check through the controls container.
    Convert to an array if its a list of lists
    return the processed container
    raise type error if the container structure is invalid
    """
    if isinstance(ctrls, (list, tuple)):
        # Check to see if list of lists
        try:
            if isinstance(ctrls[0], (list, tuple)):
                ctrls = np.array(ctrls)
        except:
            pass

    if isinstance(ctrls, np.ndarray):
        if len(ctrls.shape) != 2:
            raise TypeError("Incorrect shape for ctrl dyn gen array")
        for k in range(ctrls.shape[0]):
            for j in range(ctrls.shape[1]):
                if not isinstance(ctrls[k, j], Qobj):
                    raise TypeError("All control dyn gen must be Qobj")
    elif isinstance(ctrls, (list, tuple)):
        for ctrl in ctrls:
            if not isinstance(ctrl, Qobj):
                raise TypeError("All control dyn gen must be Qobj")
    else:
        raise TypeError("Controls list or array not set correctly")

    return ctrls

def _check_drift_dyn_gen(drift):
    if not isinstance(drift, Qobj):
        if not isinstance(drift, (list, tuple)):
            raise TypeError("drift should be a Qobj or a list of Qobj")
        else:
            for d in drift:
                if not isinstance(d, Qobj):
                    raise TypeError(
                        "drift should be a Qobj or a list of Qobj")


class Dynamics(object):

    def __init__(self, optimconfig, params=None):
        "self.config = optimconfig"
        "self.params = params"
        "self.reset()"

    def reset(self):
        # Link to optimiser object if self is linked to one
        """self.parent = None"""
        # Main functional attributes
        """self.time = None
        self.initial = None
        self.target = None
        self.ctrl_amps = None
        self.initial_ctrl_scaling = 1.0
        self.initial_ctrl_offset = 0.0
        self.drift_dyn_gen = None
        self.ctrl_dyn_gen = None
        self._tau = None
        self._evo_time = None
        self._num_ctrls = None
        self._num_tslots = None"""
        # attributes used for processing evolution
        """self.memory_optimization = 0
        self.oper_dtype = None
        self.cache_phased_dyn_gen = None
        self.cache_prop_grad = None
        self.cache_dyn_gen_eigenvectors_adj = None
        self.sparse_eigen_decomp = None
        self.dyn_dims = None
        self.dyn_shape = None
        self.sys_dims = None
        self.sys_shape = None
        self.time_depend_drift = False
        self.time_depend_ctrl_dyn_gen = False"""
        # These internal attributes will be of the internal operator data type
        # used to compute the evolution
        # Note this maybe ndarray, Qobj or some other depending on oper_dtype
        """self._drift_dyn_gen = None
        self._ctrl_dyn_gen = None
        self._phased_ctrl_dyn_gen = None
        self._dyn_gen_phase = None
        self._phase_application = None
        self._initial = None
        self._target = None
        self._onto_evo_target = None
        self._dyn_gen = None
        self._phased_dyn_gen = None
        self._prop = None
        self._prop_grad = None
        self._fwd_evo = None
        self._onwd_evo = None
        self._onto_evo = None"""
        # The _qobj attribs are Qobj representations of the equivalent
        # internal attribute. They are only set when the extenal accessors
        # are used
        """self._onto_evo_target_qobj = None
        self._dyn_gen_qobj = None
        self._prop_qobj = None
        self._prop_grad_qobj = None
        self._fwd_evo_qobj = None
        self._onwd_evo_qobj = None
        self._onto_evo_qobj = None"""
        # Atrributes used in diagonalisation
        # again in internal operator data type (see above)
        """self._decomp_curr = None
        self._prop_eigen = None
        self._dyn_gen_eigenvectors = None
        self._dyn_gen_eigenvectors_adj = None
        self._dyn_gen_factormatrix = None
        self.fact_mat_round_prec = 1e-10"""

        # Debug and information attribs
        """self.stats = None
        self.id_text = 'DYN_BASE'
        self.def_amps_fname = "ctrl_amps.txt"
        self.log_level = self.config.log_level"""
        # Internal flags
        """self._dyn_gen_mapped = False
        self._evo_initialized = False
        self._timeslots_initialized = False
        self._ctrls_initialized = False
        self._ctrl_dyn_gen_checked = False
        self._drift_dyn_gen_checked = False"""
        # Unitary checking
        "self.unitarity_check_level = 0"
        "self.unitarity_tol = 1e-10"
        # Data dumping
        "self.dump = None"
        "self.dump_to_file = False"

        "self.apply_params()"

        # Create the computing objects
        "self._create_computers()"

        "self.clear()"

    """def apply_params(self, params=None):
        ""
        Set object attributes based on the dictionary (if any) passed in the
        instantiation, or passed as a parameter
        This is called during the instantiation automatically.
        The key value pairs are the attribute name and value
        Note: attributes are created if they do not exist already,
        and are overwritten if they do.
        ""
        if not params:
            params = self.params

        if isinstance(params, dict):
            self.params = params
            for key in params:
                setattr(self, key, params[key])"""

    """ ### Login tool
        @property
        def log_level(self):
            return logger.level

        @log_level.setter
        def log_level(self, lvl):
            ""
            Set the log_level attribute and set the level of the logger
            that is call logger.setLevel(lvl)
            ""
            logger.setLevel(lvl)


        @property
        def dumping(self):
            ""
            The level of data dumping that will occur during the time evolution
            calculation.
             - NONE : No processing data dumped (Default)
             - SUMMARY : A summary of each time evolution will be recorded
             - FULL : All operators used or created in the calculation dumped
             - CUSTOM : Some customised level of dumping
            When first set to CUSTOM this is equivalent to SUMMARY. It is then up
            to the user to specify which operators are dumped
            WARNING: FULL could consume a lot of memory!
            ""
            if self.dump is None:
                lvl = 'NONE'
            else:
                lvl = self.dump.level

            return lvl

        @dumping.setter
        def dumping(self, value):
            if value is None:
                self.dump = None
            else:
                if not _is_string(value):
                    raise TypeError("Value must be string value")
                lvl = value.upper()
                if lvl == 'NONE':
                    self.dump = None
                else:
                    if not isinstance(self.dump, qtrldump.DynamicsDump):
                        self.dump = qtrldump.DynamicsDump(self, level=lvl)
                    else:
                        self.dump.level = lvl

        @property
        def dump_dir(self):
            if self.dump:
                return self.dump.dump_dir
            else:
                return None

        @dump_dir.setter
        def dump_dir(self, value):
            if not self.dump:
                self.dumping = 'SUMMARY'
            self.dump.dump_dir = value"""

    def _choose_oper_dtype(self):
        """
        Attempt select most efficient internal operator data type
        """

        if self.memory_optimization > 0:
            self.oper_dtype = Qobj
        else:
            # Method taken from Qobj.expm()
            # if method is not explicitly given, try to make a good choice
            # between sparse and dense solvers by considering the size of the
            # system and the number of non-zero elements.
            if self.time_depend_drift:
                dg = self.drift_dyn_gen[0]
            else:
                dg = self.drift_dyn_gen
            if self.time_depend_ctrl_dyn_gen:
                ctrls = self.ctrl_dyn_gen[0, :]
            else:
                ctrls = self.ctrl_dyn_gen
            for c in ctrls:
               dg = dg + c

            N = dg.data.shape[0]
            n = dg.data.nnz

            if N ** 2 < 100 * n:
                # large number of nonzero elements, revert to dense solver
                self.oper_dtype = np.ndarray
            elif N > 400:
                # large system, and quite sparse -> qutips sparse method
                self.oper_dtype = Qobj
            else:
                # small system, but quite sparse -> qutips sparse/dense method
                self.oper_dtype = np.ndarray

        return self.oper_dtype

    def _create_computers(self):
        """
        Create the default timeslot, fidelity and propagator computers
        """
        # The time slot computer. By default it is set to UpdateAll
        # can be set to DynUpdate in the configuration
        # (see class file for details)
        if self.config.tslot_type == 'DYNAMIC':
            self.tslot_computer = tslotcomp.TSlotCompDynUpdate(self)
        else:
            self.tslot_computer = tslotcomp.TSlotCompUpdateAll(self)

        self.prop_computer = propcomp.PropCompFrechet(self)
        self.fid_computer = fidcomp.FidCompTraceDiff(self)


    def clear(self):
        self.ctrl_amps = None
        self.evo_current = False
        if self.fid_computer is not None:
            self.fid_computer.clear()


    """ ###Times
        @property
        def num_tslots(self):
            if not self._timeslots_initialized:
                self.init_timeslots()
            return self._num_tslots

        @num_tslots.setter
        def num_tslots(self, value):
            self._num_tslots = value
            if self._timeslots_initialized:
                self._tau = None
                self.init_timeslots()

        @property
        def evo_time(self):
            if not self._timeslots_initialized:
                self.init_timeslots()
            return self._evo_time

        @evo_time.setter
        def evo_time(self, value):
            self._evo_time = value
            if self._timeslots_initialized:
                self._tau = None
                self.init_timeslots()

        @property
        def tau(self):
            if not self._timeslots_initialized:
                self.init_timeslots()
            return self._tau

        @tau.setter
        def tau(self, value):
            self._tau = value
            self.init_timeslots()

        def init_timeslots(self):
            ""
            Generate the timeslot duration array 'tau' based on the evo_time
            and num_tslots attributes, unless the tau attribute is already set
            in which case this step in ignored
            Generate the cumulative time array 'time' based on the tau values
            ""
            # set the time intervals to be equal timeslices of the total if
            # the have not been set already (as part of user config)
            if self._num_tslots is None:
                self._num_tslots = DEF_NUM_TSLOTS
            if self._evo_time is None:
                self._evo_time = DEF_EVO_TIME

            if self._tau is None:
                self._tau = np.ones(self._num_tslots, dtype='f') * \
                    self._evo_time/self._num_tslots
            else:
                self._num_tslots = len(self._tau)
                self._evo_time = np.sum(self._tau)

            self.time = np.zeros(self._num_tslots+1, dtype=float)
            # set the cumulative time by summing the time intervals
            for t in range(self._num_tslots):
                self.time[t+1] = self.time[t] + self._tau[t]

            self._timeslots_initialized = True"""




    def _init_evo(self):
        """
        Create the container lists / arrays for the:
        dynamics generations, propagators, and evolutions etc
        Set the time slices and cumulative time
        """
        # check evolution operators
        if not self._drift_dyn_gen_checked:
            _check_drift_dyn_gen(self.drift_dyn_gen)
        if not self._ctrl_dyn_gen_checked:
            self.ctrl_dyn_gen = _check_ctrls_container(self.ctrl_dyn_gen)

        if not isinstance(self.initial, Qobj):
            raise TypeError("initial must be a Qobj")

        if not isinstance(self.target, Qobj):
            raise TypeError("target must be a Qobj")

        self.refresh_drift_attribs()
        self.sys_dims = self.initial.dims
        self.sys_shape = self.initial.shape
        # Set the phase application method
        self._init_phase()
        self._set_memory_optimizations()
        n_ts = self.num_tslots
        n_ctrls = self.num_ctrls
        if self.oper_dtype == Qobj:
            self._initial = self.initial
            self._target = self.target
            self._drift_dyn_gen = self.drift_dyn_gen
            self._ctrl_dyn_gen = self.ctrl_dyn_gen
        elif self.oper_dtype == np.ndarray:
            self._initial = self.initial.full()
            self._target = self.target.full()
            if self.time_depend_drift:
                self._drift_dyn_gen = [d.full() for d in self.drift_dyn_gen]
            else:
                self._drift_dyn_gen = self.drift_dyn_gen.full()
            if self.time_depend_ctrl_dyn_gen:
                self._ctrl_dyn_gen = np.empty([n_ts, n_ctrls], dtype=object)
                for k in range(n_ts):
                    for j in range(n_ctrls):
                        self._ctrl_dyn_gen[k, j] = \
                                    self.ctrl_dyn_gen[k, j].full()
            else:
                self._ctrl_dyn_gen = [ctrl.full()
                                        for ctrl in self.ctrl_dyn_gen]
        elif self.oper_dtype == sp.csr_matrix:
            self._initial = self.initial.data
            self._target = self.target.data
            if self.time_depend_drift:
                self._drift_dyn_gen = [d.data for d in self.drift_dyn_gen]
            else:
                self._drift_dyn_gen = self.drift_dyn_gen.data

            if self.time_depend_ctrl_dyn_gen:
                self._ctrl_dyn_gen = np.empty([n_ts, n_ctrls], dtype=object)
                for k in range(n_ts):
                    for j in range(n_ctrls):
                        self._ctrl_dyn_gen[k, j] = \
                                    self.ctrl_dyn_gen[k, j].data
            else:
                self._ctrl_dyn_gen = [ctrl.data for ctrl in self.ctrl_dyn_gen]
        else:
            logger.warn("Unknown option '{}' for oper_dtype. "
                "Assuming that internal drift, ctrls, initial and target "
                "have been set correctly".format(self.oper_dtype))

        if self.cache_phased_dyn_gen:
            if self.time_depend_ctrl_dyn_gen:
                self._phased_ctrl_dyn_gen = np.empty([n_ts, n_ctrls],
                                                     dtype=object)
                for k in range(n_ts):
                    for j in range(n_ctrls):
                        self._phased_ctrl_dyn_gen[k, j] = self._apply_phase(
                                    self._ctrl_dyn_gen[k, j])
            else:
                self._phased_ctrl_dyn_gen = [self._apply_phase(ctrl)
                                                for ctrl in self._ctrl_dyn_gen]

        self._dyn_gen = [object for x in range(self.num_tslots)]
        if self.cache_phased_dyn_gen:
            self._phased_dyn_gen = [object for x in range(self.num_tslots)]
        self._prop = [object for x in range(self.num_tslots)]
        if self.prop_computer.grad_exact and self.cache_prop_grad:
            self._prop_grad = np.empty([self.num_tslots, self.num_ctrls],
                                      dtype=object)
        # Time evolution operator (forward propagation)
        self._fwd_evo = [object for x in range(self.num_tslots+1)]
        self._fwd_evo[0] = self._initial
        if self.fid_computer.uses_onwd_evo:
            # Time evolution operator (onward propagation)
            self._onwd_evo = [object for x in range(self.num_tslots)]
        if self.fid_computer.uses_onto_evo:
            # Onward propagation overlap with inverse target
            self._onto_evo = [object for x in range(self.num_tslots+1)]
            self._onto_evo[self.num_tslots] = self._get_onto_evo_target()

        if isinstance(self.prop_computer, propcomp.PropCompDiag):
            self._create_decomp_lists()

        if (self.log_level <= logging.DEBUG
            and isinstance(self, DynamicsUnitary)):
                self.unitarity_check_level = 1

        if self.dump_to_file:
            if self.dump is None:
                self.dumping = 'SUMMARY'
            self.dump.write_to_file = True
            self.dump.create_dump_dir()
            logger.info("Dynamics dump will be written to:\n{}".format(
                            self.dump.dump_dir))

        self._evo_initialized = True


### ---------------------------------------------------------------------------
#  --------------------------------- Moved ------------------------------------
    """Phase should be cached,
        To save memory: not keeping the original,
            (which is not really used in the computation)
    """
    @property
    def dyn_gen_phase(self):
        """
        Some op that is applied to the dyn_gen before expontiating to
        get the propagator.
        See `phase_application` for how this is applied
        """
        # Note that if this returns None then _apply_phase will never be
        # called
        return self._dyn_gen_phase

    @dyn_gen_phase.setter
    def dyn_gen_phase(self, value):
        self._dyn_gen_phase = value

    @property
    def phase_application(self):
        """
        phase_application : scalar(string), default='preop'
        Determines how the phase is applied to the dynamics generators
         - 'preop'  : P = expm(phase*dyn_gen)
         - 'postop' : P = expm(dyn_gen*phase)
         - 'custom' : Customised phase application
        The 'custom' option assumes that the _apply_phase method has been
        set to a custom function
        """
        return self._phase_application

    @phase_application.setter
    def phase_application(self, value):
        self._set_phase_application(value)

    def _set_phase_application(self, value):
        self._config_phase_application(value)
        self._phase_application = value

    def _config_phase_application(self, ph_app=None):
        """
        Set the appropriate function for the phase application
        """
        err_msg = ("Invalid value '{}' for phase application. Must be either "
                   "'preop', 'postop' or 'custom'".format(ph_app))

        if ph_app is None:
            ph_app = self._phase_application

        try:
            ph_app = ph_app.lower()
        except:
            raise ValueError(err_msg)

        if ph_app == 'preop':
            self._apply_phase = self._apply_phase_preop
        elif ph_app == 'postop':
            self._apply_phase = self._apply_phase_postop
        elif ph_app == 'custom':
            # Do nothing, assume _apply_phase set elsewhere
            pass
        else:
            raise ValueError(err_msg)

    def _init_phase(self):
        if self.dyn_gen_phase is not None:
            self._config_phase_application()
        else:
            self.cache_phased_dyn_gen = False

    def _apply_phase(self, dg):
        """
        This default method does nothing.
        It will be set to another method automatically if `phase_application`
        is 'preop' or 'postop'. It should be overridden repointed if
        `phase_application` is 'custom'
        It will never be called if `dyn_gen_phase` is None
        """
        return dg

    def _apply_phase_preop(self, dg):
        """
        Apply phasing operator to dynamics generator.
        This called during the propagator calculation.
        In this case it will be applied as phase*dg
        """
        if hasattr(self.dyn_gen_phase, 'dot'):
            phased_dg = self._dyn_gen_phase.dot(dg)
        else:
            phased_dg = self._dyn_gen_phase*dg
        return phased_dg

    def _apply_phase_postop(self, dg):
        """
        Apply phasing operator to dynamics generator.
        This called during the propagator calculation.
        In this case it will be applied as dg*phase
        """
        if hasattr(self.dyn_gen_phase, 'dot'):
            phased_dg = dg.dot(self._dyn_gen_phase)
        else:
            phased_dg = dg*self._dyn_gen_phase
        return phased_dg


### ---------------------------------------------------------------------------
    # Nope
    def _create_decomp_lists(self):
        """
        Create lists that will hold the eigen decomposition
        used in calculating propagators and gradients
        Note: used with PropCompDiag propagator calcs
        """
        n_ts = self.num_tslots
        self._decomp_curr = [False for x in range(n_ts)]
        self._prop_eigen = [object for x in range(n_ts)]
        self._dyn_gen_eigenvectors = [object for x in range(n_ts)]
        if self.cache_dyn_gen_eigenvectors_adj:
            self._dyn_gen_eigenvectors_adj = [object for x in range(n_ts)]
        self._dyn_gen_factormatrix = [object for x in range(n_ts)]

    # Changed
    def initialize_controls(self, amps, init_tslots=True):
        """
        Set the initial control amplitudes and time slices
        Note this must be called after the configuration is complete
        before any dynamics can be calculated
        """
        if not isinstance(self.prop_computer, propcomp.PropagatorComputer):
            raise errors.UsageError(
                "No prop_computer (propagator computer) "
                "set. A default should be assigned by the Dynamics subclass")

        if not isinstance(self.tslot_computer, tslotcomp.TimeslotComputer):
            raise errors.UsageError(
                "No tslot_computer (Timeslot computer)"
                " set. A default should be assigned by the Dynamics class")

        if not isinstance(self.fid_computer, fidcomp.FidelityComputer):
            raise errors.UsageError(
                "No fid_computer (Fidelity computer)"
                " set. A default should be assigned by the Dynamics subclass")

        self.ctrl_amps = None
        if not self._timeslots_initialized:
            init_tslots = True  # Why?
        if init_tslots:
            self.init_timeslots()
        self._init_evo()
        self.tslot_computer.init_comp()
        self.fid_computer.init_comp()
        self._ctrls_initialized = True
        self.update_ctrl_amps(amps)

    # Nope
    def check_ctrls_initialized(self):
        if not self._ctrls_initialized:
            raise errors.UsageError(
                "Controls not initialised. "
                "Ensure Dynamics.initialize_controls has been "
                "executed with the initial control amplitudes.")

    # Nope
    def get_amp_times(self):
        return self.time[:self.num_tslots]

    # log/dump
    def save_amps(self, file_name=None, times=None, amps=None, verbose=False):
        """
        Save a file with the current control amplitudes in each timeslot
        The first column in the file will be the start time of the slot

        Parameters
        ----------
        file_name : string
            Name of the file
            If None given the def_amps_fname attribuite will be used

        times : List type (or string)
            List / array of the start times for each slot
            If None given this will be retrieved through get_amp_times()
            If 'exclude' then times will not be saved in the file, just
            the amplitudes

        amps : Array[num_tslots, num_ctrls]
            Amplitudes to be saved
            If None given the ctrl_amps attribute will be used

        verbose : Boolean
            If True then an info message will be logged
        """
        self.check_ctrls_initialized()

        inctimes = True
        if file_name is None:
            file_name = self.def_amps_fname
        if amps is None:
            amps = self.ctrl_amps
        if times is None:
            times = self.get_amp_times()
        else:
            if _is_string(times):
                if times.lower() == 'exclude':
                    inctimes = False
                else:
                    logger.warn("Unknown option for times '{}' "
                                "when saving amplitudes".format(times))
                    times = self.get_amp_times()

        try:
            if inctimes:
                shp = amps.shape
                data = np.empty([shp[0], shp[1] + 1], dtype=float)
                data[:, 0] = times
                data[:, 1:] = amps
            else:
                data = amps

            np.savetxt(file_name, data, delimiter='\t', fmt='%14.6g')

            if verbose:
                logger.info("Amplitudes saved to file: " + file_name)
        except Exception as e:
            logger.error("Failed to save amplitudes due to underling "
                         "error: {}".format(e))

    # Nope
    def update_ctrl_amps(self, new_amps):
        """
        Determine if any amplitudes have changed. If so, then mark the
        timeslots as needing recalculation
        The actual work is completed by the compare_amps method of the
        timeslot computer
        """

        if self.log_level <= logging.DEBUG_INTENSE:
            logger.log(logging.DEBUG_INTENSE, "Updating amplitudes...\n"
                       "Current control amplitudes:\n" + str(self.ctrl_amps) +
                       "\n(potenially) new amplitudes:\n" + str(new_amps))

        self.tslot_computer.compare_amps(new_amps)

    # Nope
    def flag_system_changed(self):
        """
        Flag evolution, fidelity and gradients as needing recalculation
        """
        self.evo_current = False
        self.fid_computer.flag_system_changed()

    # Nope
    def get_drift_dim(self):
        """
        Returns the size of the matrix that defines the drift dynamics
        that is assuming the drift is NxN, then this returns N
        """
        if self.dyn_shape is None:
            self.refresh_drift_attribs()
        return self.dyn_shape[0]

    # Nope
    def refresh_drift_attribs(self):
        """Reset the dyn_shape, dyn_dims and time_depend_drift attribs"""

        if isinstance(self.drift_dyn_gen, (list, tuple)):
            d0 = self.drift_dyn_gen[0]
            self.time_depend_drift = True
        else:
            d0 = self.drift_dyn_gen
            self.time_depend_drift = False

        if not isinstance(d0, Qobj):
            raise TypeError("Unable to determine drift attributes, "
                    "because drift_dyn_gen is not Qobj (nor list of)")

        self.dyn_shape = d0.shape
        self.dyn_dims = d0.dims

    # Nope
    def _get_num_ctrls(self):
        if not self._ctrl_dyn_gen_checked:
            self.ctrl_dyn_gen = _check_ctrls_container(self.ctrl_dyn_gen)
            self._ctrl_dyn_gen_checked = True
        if isinstance(self.ctrl_dyn_gen, np.ndarray):
            self._num_ctrls = self.ctrl_dyn_gen.shape[1]
            self.time_depend_ctrl_dyn_gen = True
        else:
            self._num_ctrls = len(self.ctrl_dyn_gen)

        return self._num_ctrls

    @property  # Nope
    def num_ctrls(self):
        ""
        calculate the of controls from the length of the control list
        sets the num_ctrls property, which can be used alternatively
        subsequently
        ""
        if self._num_ctrls is None:
            self._num_ctrls = self._get_num_ctrls()
        return self._num_ctrls

    # Nope
    @property
    def onto_evo_target(self):
        if self._onto_evo_target is None:
            self._get_onto_evo_target()

        if self._onto_evo_target_qobj is None:
            if isinstance(self._onto_evo_target, Qobj):
                self._onto_evo_target_qobj = self._onto_evo_target
            else:
                rev_dims = [self.sys_dims[1], self.sys_dims[0]]
                self._onto_evo_target_qobj = Qobj(self._onto_evo_target,
                                                  dims=rev_dims)

        return self._onto_evo_target_qobj

    # Nope
    def _get_onto_evo_target(self):
        """
        Get the inverse of the target.
        Used for calculating the 'onto target' evolution
        This is actually only relevant for unitary dynamics where
        the target.dag() is what is required
        However, for completeness, in general the inverse of the target
        operator is is required
        For state-to-state, the bra corresponding to the is required ket
        """
        if self.target.shape[0] == self.target.shape[1]:
            #Target is operator
            targ = la.inv(self.target.full())
            if self.oper_dtype == Qobj:
                self._onto_evo_target = Qobj(targ)
            elif self.oper_dtype == np.ndarray:
                self._onto_evo_target = targ
            elif self.oper_dtype == sp.csr_matrix:
                self._onto_evo_target = sp.csr_matrix(targ)
            else:
                targ_cls = self._target.__class__
                self._onto_evo_target = targ_cls(targ)
        else:
            if self.oper_dtype == Qobj:
                self._onto_evo_target = self.target.dag()
            elif self.oper_dtype == np.ndarray:
                self._onto_evo_target = self.target.dag().full()
            elif self.oper_dtype == sp.csr_matrix:
                self._onto_evo_target = self.target.dag().data
            else:
                targ_cls = self._target.__class__
                self._onto_evo_target = targ_cls(self.target.dag().full())

        return self._onto_evo_target

### ---------------------------------------------------------------------------
#  --------------------------------- Moved ------------------------------------
    def _combine_dyn_gen(self, k):
        """
        Computes the dynamics generator for a given timeslot
        The is the combined Hamiltion for unitary systems
        Also applies the phase (if any required by the propagation)
        """
        if self.time_depend_drift:
            dg = self._drift_dyn_gen[k]
        else:
            dg = self._drift_dyn_gen
        for j in range(self._num_ctrls):
            if self.time_depend_ctrl_dyn_gen:
                dg = dg + self.ctrl_amps[k, j]*self._ctrl_dyn_gen[k, j]
            else:
                dg = dg + self.ctrl_amps[k, j]*self._ctrl_dyn_gen[j]

        self._dyn_gen[k] = dg
        if self.cache_phased_dyn_gen:
            self._phased_dyn_gen[k] = self._apply_phase(dg)

    def _get_phased_dyn_gen(self, k):
        if self.dyn_gen_phase is None:
            return self._dyn_gen[k]
        else:
            if self._phased_dyn_gen is None:
                return self._apply_phase(self._dyn_gen[k])
            else:
                return self._phased_dyn_gen[k]

    def _get_phased_ctrl_dyn_gen(self, k, j):
        if self._phased_ctrl_dyn_gen is not None:
            if self.time_depend_ctrl_dyn_gen:
                return self._phased_ctrl_dyn_gen[k, j]
            else:
                return self._phased_ctrl_dyn_gen[j]
        else:
            if self.time_depend_ctrl_dyn_gen:
                if self.dyn_gen_phase is None:
                    return self._ctrl_dyn_gen[k, j]
                else:
                    return self._apply_phase(self._ctrl_dyn_gen[k, j])
            else:
                if self.dyn_gen_phase is None:
                    return self._ctrl_dyn_gen[j]
                else:
                    return self._apply_phase(self._ctrl_dyn_gen[j])

    @property
    def dyn_gen(self):
        ""
        List of combined dynamics generators (Qobj) for each timeslot
        ""
        if self._dyn_gen is not None:
            if self._dyn_gen_qobj is None:
                if self.oper_dtype == Qobj:
                    self._dyn_gen_qobj = self._dyn_gen
                else:
                    self._dyn_gen_qobj = [Qobj(dg, dims=self.dyn_dims)
                                            for dg in self._dyn_gen]
        return self._dyn_gen_qobj

### ---------------------------------------------------------------------------
    @property
    def prop(self):
        ""
        List of propagators (Qobj) for each timeslot
        ""
        if self._prop is not None:
            if self._prop_qobj is None:
                if self.oper_dtype == Qobj:
                    self._prop_qobj = self._prop
                else:
                    self._prop_qobj = [Qobj(dg, dims=self.dyn_dims)
                                            for dg in self._prop]
        return self._prop_qobj

    @property
    def prop_grad(self):
        ""
        Array of propagator gradients (Qobj) for each timeslot, control
        ""
        if self._prop_grad is not None:
            if self._prop_grad_qobj is None:
                if self.oper_dtype == Qobj:
                    self._prop_grad_qobj = self._prop_grad
                else:
                    self._prop_grad_qobj = np.empty(
                                    [self.num_tslots, self.num_ctrls],
                                    dtype=object)
                    for k in range(self.num_tslots):
                        for j in range(self.num_ctrls):
                            self._prop_grad_qobj[k, j] = Qobj(
                                                    self._prop_grad[k, j],
                                                    dims=self.dyn_dims)
        return self._prop_grad_qobj

    def _get_prop_grad(self, k, j):
        if self.cache_prop_grad:
            prop_grad = self._prop_grad[k, j]
        else:
            prop_grad = self.prop_computer._compute_prop_grad(k, j,
                                                       compute_prop = False)
        return prop_grad
### ---------------------------------------------------------------------------

    @property
    def fwd_evo(self):
        """
        List of evolution operators (Qobj) from the initial to the given
        timeslot
        """
        if self._fwd_evo is not None:
            if self._fwd_evo_qobj is None:
                if self.oper_dtype == Qobj:
                    self._fwd_evo_qobj = self._fwd_evo
                else:
                    self._fwd_evo_qobj = [self.initial]
                    for k in range(1, self.num_tslots+1):
                        self._fwd_evo_qobj.append(Qobj(self._fwd_evo[k],
                                                       dims=self.sys_dims))
        return self._fwd_evo_qobj

    def _get_full_evo(self): # Why
        return self._fwd_evo[self._num_tslots]

    @property  # Why double
    def full_evo(self):
        """Full evolution - time evolution at final time slot"""
        return self.fwd_evo[self.num_tslots]

    @property  # Why
    def onwd_evo(self):
        """
        List of evolution operators (Qobj) from the initial to the given
        timeslot
        """
        if self._onwd_evo is not None:
            if self._onwd_evo_qobj is None:
                if self.oper_dtype == Qobj:
                    self._onwd_evo_qobj = self._fwd_evo
                else:
                    self._onwd_evo_qobj = [Qobj(dg, dims=self.sys_dims)
                                            for dg in self._onwd_evo]
        return self._onwd_evo_qobj

    @property  # Why
    def onto_evo(self):
        """
        List of evolution operators (Qobj) from the initial to the given
        timeslot
        """
        if self._onto_evo is not None:
            if self._onto_evo_qobj is None:
                if self.oper_dtype == Qobj:
                    self._onto_evo_qobj = self._onto_evo
                else:
                    self._onto_evo_qobj = []
                    for k in range(0, self.num_tslots):
                        self._onto_evo_qobj.append(Qobj(self._onto_evo[k],
                                                       dims=self.sys_dims))
                    self._onto_evo_qobj.append(self.onto_evo_target)

        return self._onto_evo_qobj

    def compute_evolution(self):
        """
        Recalculate the time evolution operators
        Dynamics generators (e.g. Hamiltonian) and
        prop (propagators) are calculated as necessary
        Actual work is completed by the recompute_evolution method
        of the timeslot computer
        """

        # Check if values are already current, otherwise calculate all values
        if not self.evo_current:
            if self.log_level <= logging.DEBUG_VERBOSE:
                logger.log(logging.DEBUG_VERBOSE, "Computing evolution")
            self.tslot_computer.recompute_evolution()
            self.evo_current = True
            return True
        else:
            return False

### ---------------------------------------------------------------------------

    def _is_unitary(self, A):
        """
        Checks whether operator A is unitary
        A can be either Qobj or ndarray
        """
        if isinstance(A, Qobj):
            unitary = np.allclose(np.eye(A.shape[0]), A*A.dag().full(),
                        atol=self.unitarity_tol)
        else:
            unitary = np.allclose(np.eye(len(A)), A.dot(A.T.conj()),
                        atol=self.unitarity_tol)

        return unitary

    def _calc_unitary_err(self, A):
        if isinstance(A, Qobj):
            err = np.sum(abs(np.eye(A.shape[0]) - A*A.dag().full()))
        else:
            err = np.sum(abs(np.eye(len(A)) - A.dot(A.T.conj())))

        return err

    def unitarity_check(self):
        """
        Checks whether all propagators are unitary
        """
        for k in range(self.num_tslots):
            if not self._is_unitary(self._prop[k]):
                logger.warning(
                    "Progator of timeslot {} is not unitary".format(k))
