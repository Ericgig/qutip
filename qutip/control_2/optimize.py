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
# @email1: agp1@aber.ac.uk
# @email2: alex.pitchford@gmail.com
# @organization: Aberystwyth University
# @supervisor: Daniel Burgarth

"""
Classes here are expected to implement a run_optimization function
that will use some method for optimising the control pulse, as defined
by the control amplitudes. The system that the pulse acts upon are defined
by the Dynamics object that must be passed in the instantiation.

The methods are typically N dimensional function optimisers that
find the minima of a fidelity error function. Note the number of variables
for the fidelity function is the number of control timeslots,
i.e. n_ctrls x Ntimeslots
The methods will call functions on the Dynamics.fid_computer object,
one or many times per interation,
to get the fidelity error and gradient wrt to the amplitudes.
The optimisation will stop when one of the termination conditions are met,
for example: the fidelity aim has be reached, a local minima has been found,
the maximum time allowed has been exceeded

These function optimisation methods are so far from SciPy.optimize
The two methods implemented are:

    BFGS - Broyden–Fletcher–Goldfarb–Shanno algorithm

        This a quasi second order Newton method. It uses successive calls to
        the gradient function to make an estimation of the curvature (Hessian)
        and hence direct its search for the function minima
        The SciPy implementation is pure Python and hance is execution speed is
        not high
        use subclass: OptimizerBFGS

    L-BFGS-B - Bounded, limited memory BFGS

        This a version of the BFGS method where the Hessian approximation is
        only based on a set of the most recent gradient calls. It generally
        performs better where the are a large number of variables
        The SciPy implementation of L-BFGS-B is wrapper around a well
        established and actively maintained implementation in Fortran
        Its is therefore very fast.
        # See SciPy documentation for credit and details on the
        # scipy.optimize.fmin_l_bfgs_b function
        use subclass: OptimizerLBFGSB

The baseclass Optimizer implements the function wrappers to the
fidelity error, gradient, and iteration callback functions.
These are called from the within the SciPy optimisation functions.
The subclasses implement the algorithm specific pulse optimisation function.
"""


import numpy as np
import timeit
import scipy.optimize as spopt

# QuTiP
from qutip import Qobj


import importlib
import importlib.util
moduleName = "/home/eric/algo/qutip/qutip/qutip/control_2/stats.py"
spec = importlib.util.spec_from_file_location("stats", moduleName)
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)
Stats = stats.Stats



class solverEnd(Exception):
    pass

termination_conditions = {}
termination_conditions["fid_goal"] = None
termination_conditions["fid_err_targ"] = 1e-7
termination_conditions["min_gradient_norm"] = 1e-7
termination_conditions["max_wall_time"] = 10*60.0
termination_conditions["max_fid_func_calls"] = 1e6
termination_conditions["max_iterations"] = 10000

method_options = {}
method_options["ftol"] = 1e-5
method_options["disp"] = False

class Optimizer(object):
    """
    Base class for all control pulse optimisers. This class should not be
    instantiated, use its subclasses
    This class implements the fidelity, gradient and interation callback
    functions.
    All subclass objects must be initialised with a

        OptimConfig instance - various configuration options
        Dynamics instance - describes the dynamics of the (quantum) system
                            to be control optimised

    Attributes
    ----------
    log_level : integer
        level of messaging output from the logger.
        Options are attributes of qutip.logging_utils,
        in decreasing levels of messaging, are:
        DEBUG_INTENSE, DEBUG_VERBOSE, DEBUG, INFO, WARN, ERROR, CRITICAL
        Anything WARN or above is effectively 'quiet' execution,
        assuming everything runs as expected.
        The default NOTSET implies that the level will be taken from
        the QuTiP settings file, which by default is WARN

    params:  Dictionary
        The key value pairs are the attribute name and value
        Note: attributes are created if they do not exist already,
        and are overwritten if they do.

    alg : string
        Algorithm to use in pulse optimisation.
        Options are:
            'GRAPE' (default) - GRadient Ascent Pulse Engineering
            'CRAB' - Chopped RAndom Basis

    alg_params : Dictionary
        options that are specific to the pulse optim algorithm
        that is GRAPE or CRAB

    disp_conv_msg : bool
        Set true to display a convergence message
        (for scipy.optimize.minimize methods anyway)

    optim_method : string
        a scipy.optimize.minimize method that will be used to optimise
        the pulse for minimum fidelity error

    method_params : Dictionary
        Options for the optim_method.
        Note that where there is an equivalent attribute of this instance
        or the termination_conditions (for example maxiter)
        it will override an value in these options

    approx_grad : bool
        If set True then the method will approximate the gradient itself
        (if it has requirement and facility for this)
        This will mean that the fid_err_grad_wrapper will not get called
        Note it should be left False when using the Dynamics
        to calculate approximate gradients
        Note it is set True automatically when the alg is CRAB

    amp_lbound : float or list of floats
        lower boundaries for the control amplitudes
        Can be a scalar value applied to all controls
        or a list of bounds for each control

    amp_ubound : float or list of floats
        upper boundaries for the control amplitudes
        Can be a scalar value applied to all controls
        or a list of bounds for each control

    bounds : List of floats
        Bounds for the parameters.
        If not set before the run_optimization call then the list
        is built automatically based on the amp_lbound and amp_ubound
        attributes.
        Setting this attribute directly allows specific bounds to be set
        for individual parameters.
        Note: Only some methods use bounds

    dynamics : Dynamics (subclass instance)
        describes the dynamics of the (quantum) system to be control optimised
        (see Dynamics classes for details)

    config : OptimConfig instance
        various configuration options
        (see OptimConfig for details)

    termination_conditions : TerminationCondition instance
        attributes determine when the optimisation will end

    pulse_generator : PulseGen (subclass instance)
        (can be) used to create initial pulses
        not used by the class, but set by pulseoptim.create_pulse_optimizer

    stats : Stats
        attributes of which give performance stats for the optimisation
        set to None to reduce overhead of calculating stats.
        Note it is (usually) shared with the Dynamics instance

    dump : :class:`dump.OptimDump`
        Container for data dumped during the optimisation.
        Can be set by specifying the dumping level or set directly.
        Note this is mainly intended for user and a development debugging
        but could be used for status information during a long optimisation.

    dumping : string
        level of data dumping: NONE, SUMMARY, FULL or CUSTOM
        See property docstring for details

    dump_to_file : bool
        If set True then data will be dumped to file during the optimisation
        dumping will be set to SUMMARY during init_optim
        if dump_to_file is True and dumping not set.
        Default is False

    dump_dir : string
        Basically a link to dump.dump_dir. Exists so that it can be set through
        optim_params.
        If dump is None then will return None or will set dumping to SUMMARY
        when setting a path

    iter_summary : :class:`OptimIterSummary`
        Summary of the most recent iteration.
        Note this is only set if dummping is on

    """

    def __init__(self, error, grad=None, x0=np.zeros(0),
                 stats=None, dyn_stats=None):
        self.errorFunc = error
        self.gradFunc = grad
        self.x_shape = x0.shape
        self.x0 = x0.flatten()
        if stats is not None:
            self.stats = stats
        else:
            self.stats = Stats(1,1)
        self.dyn_stats = dyn_stats
        self.reset()

    def reset(self):
        self.alg = 'GRAPE'
        self.alg_params = None

        self.method = 'L-BFGS-B'
        self.method_params = None

        self.bounds = None

        # termination conditions
        self.termination_conditions = {}
        self.termination_conditions["fid_goal"] = None
        self.termination_conditions["fid_err_targ"] = 1e-7
        self.termination_conditions["min_gradient_norm"] = 1e-7
        self.termination_conditions["max_wall_time"] = 10*60.0
        self.termination_conditions["max_fid_func_calls"] = 1e6
        self.termination_conditions["max_iterations"] = 10000
        self.termination_conditions.update(termination_conditions)

        self.method_options = {}
        self.method_options["ftol"] = 1e-5
        self.method_options["disp"] = False
        self.method_options.update(method_options)
        self.method_options["maxfun"] = \
                self.termination_conditions["max_fid_func_calls"]
        self.method_options["gtol"] = \
                self.termination_conditions["min_gradient_norm"]
        self.method_options["maxiter"] = \
                self.termination_conditions["max_iterations"]

        self.wall_time_optim_start = 0.0
        self.num_iter = 0
        self.num_fid_func_calls = 0
        self.num_grad_func_calls = 0
        self.termination_signal = ""

        # Stats
        # self.stats = None
        # self.iter_summary = None
        # self.disp_conv_msg = False
        # self.apply_params()

    def apply_method_params(self, params):
        """
        Loops through all the method_params
        (either passed here or the method_params attribute)
        If the name matches an attribute of this object or the
        termination conditions object, then the value of this attribute
        is set. Otherwise it is assumed to a method_option for the
        scipy.optimize.minimize function
        """
        if isinstance(params, dict):
            for key, val in params.items:
                if key in self.termination_conditions:
                    self.termination_conditions[key] = val
                else:
                    self.method_options[key] = val

    def set_bounds(self, l_bound, u_bound):
        self.bounds = []
        for t in range(self.x_shape[0]):
            for c in range(self.x_shape[1]):
                if isinstance(l_bound, list):
                    lb = l_bound[c]
                else:
                    lb = l_bound
                if isinstance(self.amp_ubound, list):
                    ub = u_bound[c]
                else:
                    ub = u_bound

                if not lb is None and np.isinf(lb):
                    lb = None
                if not ub is None and np.isinf(ub):
                    ub = None

                self.bounds.append((lb, ub))

    def run_optimization(self, result):
        """
        This default function optimisation method is a wrapper to the
        scipy.optimize.minimize function.

        It will attempt to minimise the fidelity error with respect to some
        parameters, which are determined by _get_optim_var_vals (see below)

        The optimisation end when one of the passed termination conditions
        has been met, e.g. target achieved, wall time, or
        function call or iteration count exceeded. Note these
        conditions include gradient minimum met (local minima) for
        methods that use a gradient.

        The function minimisation method is taken from the optim_method
        attribute. Note that not all of these methods have been tested.
        Note that some of these use a gradient and some do not.
        See the scipy documentation for details. Options specific to the
        method can be passed setting the method_params attribute.

        If the parameter term_conds=None, then the termination_conditions
        attribute must already be set. It will be overwritten if the
        parameter is not None

        The result is returned in an OptimResult object, which includes
        the final fidelity, time evolution, reason for termination etc

        """
        # self.init_optim(term_conds)
        # term_conds = self.termination_conditions
        # dyn = self.dynamics
        # cfg = self.config
        # self.optim_var_vals = x0  # self._get_optim_var_vals()

        """if self.stats is not None:
            self.stats.wall_time_optim_start = st_time
            self.stats.wall_time_optim_end = 0.0
            self.stats.num_iter = 0"""

        #self._build_method_options()
        #result = self._create_result()

        st_time = timeit.default_timer()
        self.wall_time_optimize_start = st_time
        if True: #self.stats is not None and self.stats.timings:
            self.stats.wall_time_optim_start = st_time
            self.stats.num_grad_func_calls_per_iter = [0]
            self.stats.num_fidelity_func_calls_per_iter = [0]
            self.stats.wall_time_per_iter = [0.]

        """result.evo_full_initial = self.dynamics.full_evo.copy()
        result.initial_fid_err = self.fid_err_func_wrapper(self.x0)
        result.initial_amps = self.x0.reshape(self.x_shape).copy()"""

        result.optimizer = self.method

        if self.alg == 'CRAB':
            jac=None
        else:
            jac=self.fid_err_grad_wrapper

        try:
            opt_res = spopt.minimize(
                self.fid_err_func_wrapper, self.x0,
                method=self.method,
                jac=jac,
                bounds=self.bounds,
                options=self.method_options,
                callback=self.iter_step_callback_func)
            result.final_x = opt_res.x.reshape(self.x_shape)
            result.termination_reason = opt_res.message
            result.num_iter = opt_res.nit

        except solverEnd as except_term:
            result.final_x = self.x1.reshape(self.x_shape)
            result.termination_reason = self.termination_signal
            result.num_iter = self.num_iter

        result.fid_err = self.err
        result.grad_norm_final = self.gradnorm
        result.num_fid_func_calls = self.num_fid_func_calls
        result.wall_time = timeit.default_timer() - st_time

        if True: #self.stats is not None and self.stats.timings:
            self.stats.wall_time_optim_end = timeit.default_timer()
            self.stats.wall_time_optim = timeit.default_timer() - st_time
            self.stats.wall_time_per_iter = \
                            np.diff(np.array(self.stats.wall_time_per_iter))

        return result

    def fid_err_func_wrapper(self, x):
        """
        Get the fidelity error achieved using the ctrl amplitudes passed
        in as the first argument.

        This is called by generic optimisation algorithm as the
        func to the minimised. The argument is the current
        variable values, i.e. control amplitudes, passed as
        a flat array. Hence these are reshaped as [nTimeslots, n_ctrls]
        and then used to update the stored ctrl values (if they have changed)

        The error is checked against the target, and the optimisation is
        terminated if the target has been achieved.
        """
        self.num_fid_func_calls += 1
        if True: #self.stats is not None and self.stats.timings:
            self.stats.num_fidelity_func_calls += 1
            self.stats.num_fidelity_func_calls_per_iter[self.num_iter] += 1
            t_start = timeit.default_timer()

        x_2d = x.reshape(self.x_shape)
        self.err = self.errorFunc(x_2d)

        if True: #if self.stats is not None and self.stats.timings:
            self.stats.wall_time_fidelity_func += timeit.default_timer() - t_start

        #if self.err <= self.termination_conditions["fid_err_targ"]:
        #    self.termination_signal = "fid_err_targ"
        #    raise solverEnd()
        #    raise errors.GoalAchievedTerminate(err)

        if self.num_fid_func_calls > self.termination_conditions["max_fid_func_calls"]:
            self.termination_signal = "max_fid_func_calls"
            raise solverEnd()
            raise errors.MaxFidFuncCallTerminate()

        return self.err

    def fid_err_grad_wrapper(self, x):
        """
        Get the gradient of the fidelity error with respect to all of the
        variables, i.e. the ctrl amplidutes in each timeslot

        This is called by generic optimisation algorithm as the gradients of
        func to the minimised wrt the variables. The argument is the current
        variable values, i.e. control amplitudes, passed as
        a flat array. Hence these are reshaped as [nTimeslots, n_ctrls]
        and then used to update the stored ctrl values (if they have changed)

        Although the optimisation algorithms have a check within them for
        function convergence, i.e. local minima, the sum of the squares
        of the normalised gradient is checked explicitly, and the
        optimisation is terminated if this is below the min_gradient_norm
        condition
        """
        self.num_grad_func_calls += 1
        if True: #if self.stats is not None and self.stats.timings:
            self.stats.num_grad_func_calls += 1
            self.stats.num_grad_func_calls_per_iter[self.num_iter] += 1
            t_start = timeit.default_timer()

        x_2d = x.reshape(self.x_shape)
        grad = self.gradFunc(x_2d)

        if True: #if self.stats is not None and self.stats.timings:
            self.stats.wall_time_grad_func += timeit.default_timer() - t_start

        self.gradnorm = np.sum(grad*grad.conj())
        #if self.gradnorm < self.termination_conditions['min_gradient_norm']:
        #    self.termination_signal = "min_gradient_norm"
        #    raise solverEnd()

        return grad.flatten()

    def iter_step_callback_func(self, x):
        """
        Check the elapsed wall time for the optimisation run so far.
        Terminate if this has exceeded the maximum allowed time
        """
        self.num_iter += 1
        wall_time = timeit.default_timer() - self.wall_time_optimize_start
        if True: #if self.stats is not None and self.stats.timings:
            self.stats.num_iter += 1
            #self.stats.num_grad_func_calls += 1
            self.stats.num_grad_func_calls_per_iter += [0]
            self.stats.num_fidelity_func_calls_per_iter += [0]
            self.stats.wall_time_per_iter += [wall_time]

        self.x1 = x.copy()

        if self.stats.states:
            x_2d = x.reshape(self.x_shape)
            self.dyn_stats(x_2d)
            self.err = self.errorFunc(x_2d)
            #grad = self.gradFunc(x_2d, stats=True)
            #self.stats.grad_norm += [np.sum(grad*grad.conj())]
            print("step: ", self.num_iter, " cost: ",self.err)

        if wall_time > self.termination_conditions["max_wall_time"]:
            self.termination_signal = "max_wall_time"
            raise solverEnd()
        if self.num_iter > self.termination_conditions['max_iterations']:
            self.termination_signal = "max_iterations"
            raise solverEnd()
