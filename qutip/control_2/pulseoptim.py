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
Wrapper functions that will manage the creation of the objects,
build the configuration, and execute the algorithm required to optimise
a set of ctrl pulses for a given (quantum) system.
The fidelity error is some measure of distance of the system evolution
from the given target evolution in the time allowed for the evolution.
The functions minimise this fidelity error wrt the piecewise control
amplitudes in the timeslots

There are currently two quantum control pulse optmisations algorithms
implemented in this library. There are accessible through the methods
in this module. Both the algorithms use the scipy.optimize methods
to minimise the fidelity error with respect to to variables that define
the pulse.

GRAPE
-----
The default algorithm (as it was implemented here first) is GRAPE
GRadient Ascent Pulse Engineering [1][2]. It uses a gradient based method such
as BFGS to minimise the fidelity error. This makes convergence very quick
when an exact gradient can be calculated, but this limits the factors that can
taken into account in the fidelity.

CRAB
----
The CRAB [3][4] algorithm was developed at the University of Ulm.
In full it is the Chopped RAndom Basis algorithm.
The main difference is that it reduces the number of optimisation variables
by defining the control pulses by expansions of basis functions,
where the variables are the coefficients. Typically a Fourier series is chosen,
i.e. the variables are the Fourier coefficients.
Therefore it does not need to compute an explicit gradient.
By default it uses the Nelder-Mead method for fidelity error minimisation.

References
----------
1.  N Khaneja et. al.
    Optimal control of coupled spin dynamics: Design of NMR pulse sequences
    by gradient ascent algorithms. J. Magn. Reson. 172, 296–305 (2005).
2.  Shai Machnes et.al
    DYNAMO - Dynamic Framework for Quantum Optimal Control
    arXiv.1011.4874
3.  Doria, P., Calarco, T. & Montangero, S.
    Optimal Control Technique for Many-Body Quantum Dynamics.
    Phys. Rev. Lett. 106, 1–4 (2011).
4.  Caneva, T., Calarco, T. & Montangero, S.
    Chopped random-basis quantum optimization.
    Phys. Rev. A - At. Mol. Opt. Phys. 84, (2011).

"""
import numpy as np
import warnings

# QuTiP
from qutip import Qobj
"""import qutip.logging_utils as logging
logger = logging.get_logger()
# QuTiP control modules
import qutip.control.optimconfig as optimconfig
import qutip.control.dynamics as dynamics
import qutip.control.termcond as termcond
import qutip.control.optimizer as optimizer
import qutip.control.stats as stats
import qutip.control.errors as errors
import qutip.control.fidcomp as fidcomp
import qutip.control.propcomp as propcomp
import qutip.control.pulsegen as pulsegen
#import qutip.control.pulsegencrab as pulsegencrab"""

dev_import = False

if dev_import:
    import importlib
    import importlib.util

    moduleName = "/home/eric/algo/qutip/qutip/qutip/control_2/pulsegen.py"
    spec = importlib.util.spec_from_file_location("pulsegen", moduleName)
    pulsegen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pulsegen)

    moduleName = "/home/eric/algo/qutip/qutip/qutip/control_2/transfer_functions.py"
    spec = importlib.util.spec_from_file_location("transfer_functions", moduleName)
    transfer_functions = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transfer_functions)
else:
    import qutip.control_2.pulsegen as pulsegen
    import qutip.control_2.transfer_function as transfer_function


warnings.simplefilter('always', DeprecationWarning) #turn off filter
def _param_deprecation(message, stacklevel=3):
    """
    Issue deprecation warning
    Using stacklevel=3 will ensure message refers the function
    calling with the deprecated parameter,
    """
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)

def _upper_safe(s):
    try:
        s = s.upper()
    except:
        pass
    return s

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
    #Check through the controls container.
    #Convert to an array if its a list of lists
    #return the processed container
    #raise type error if the container structure is invalid
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

def calc_omega(n):
    """
    Calculate the 2n x 2n Omega matrix
    Used as dynamics generator phase to calculate symplectic propagators

    Parameters
    ----------
    n : scalar(int)
        number of modes in oscillator system

    Returns
    -------
    array(float)
        Symplectic phase Omega
    """
    omg = np.zeros((2*n, 2*n))
    for i in range(0,2*n,2):
        omg[i+1,i] = -1
        omg[i,i+1] = 1
    return omg



def optimize_pulse(
        drift, ctrls, initial, target,
        num_tslots=None, evo_time=None, tau=None,
        amp_lbound=None, amp_ubound=None,
        fid_err_targ=1e-10, min_grad=1e-10, max_iter=500, max_wall_time=180,

        optim_method='DEF', method_params=None,
        transfer_function_type='DEF', transfer_function_params={},
        prop_type='DEF', mat_type='DEF', mat_params={},
        fid_type='DEF', fid_params=None,
        tslot_type='DEF', tslot_params=None,
        init_pulse_type='DEF', init_pulse_params=None,
        log_level=logging.NOTSET, out_file_ext=None, gen_stats=False,
        alg='GRAPE', symplectic="",
        # old
        optim_params=None, prop_params=None,
        alg_params=None, dyn_type='GEN_MAT', dyn_params=None,
        pulse_scaling=1.0, pulse_offset=0.0,
        ramping_pulse_type=None, ramping_pulse_params=None
        #already old
        # optim_alg=None, max_metric_corr=None, accuracy_factor=None,
        # phase_option=None, amp_update_mode=None,
        # fid_err_scale_factor=None,
        ):
    """
    Optimise a control pulse to minimise the fidelity error.
    The dynamics of the system in any given timeslot are governed
    by the combined dynamics generator,
    i.e. the sum of the drift+ctrl_amp[j]*ctrls[j]
    The control pulse is an [n_ts, n_ctrls)] array of piecewise amplitudes
    Starting from an intital (typically random) pulse,
    a multivariable optimisation algorithm attempts to determines the
    optimal values for the control pulse to minimise the fidelity error
    The fidelity error is some measure of distance of the system evolution
    from the given target evolution in the time allowed for the evolution.

    Parameters
    ----------
    drift : Qobj or list of Qobj
        the underlying dynamics generator of the system
        can provide list (of length num_tslots) for time dependent drift

    ctrls : List of Qobj or array like [num_tslots, evo_time]
        a list of control dynamics generators. These are scaled by
        the amplitudes to alter the overall dynamics
        Array like imput can be provided for time dependent control generators

    initial : Qobj
        starting point for the evolution.
        Typically the identity matrix

    target : Qobj
        target transformation, e.g. gate or state, for the time evolution

    num_tslots : integer or None
        number of timeslots.
        None implies that timeslots will be given in the tau array

    evo_time : float or None
        total time for the evolution
        None implies that timeslots will be given in the tau array

    tau : array[num_tslots] of floats or None
        durations for the timeslots.
        if this is given then num_tslots and evo_time are dervived
        from it
        None implies that timeslot durations will be equal and
        calculated as evo_time/num_tslots

    amp_lbound : float or list of floats
        lower boundaries for the control amplitudes
        Can be a scalar value applied to all controls
        or a list of bounds for each control

    amp_ubound : float or list of floats
        upper boundaries for the control amplitudes
        Can be a scalar value applied to all controls
        or a list of bounds for each control

    fid_err_targ : float
        Fidelity error target. Pulse optimisation will
        terminate when the fidelity error falls below this value

    mim_grad : float
        Minimum gradient. When the sum of the squares of the
        gradients wrt to the control amplitudes falls below this
        value, the optimisation terminates, assuming local minima

    max_iter : integer
        Maximum number of iterations of the optimisation algorithm

    max_wall_time : float
        Maximum allowed elapsed time for the  optimisation algorithm

    alg : string
        Algorithm to use in pulse optimisation.
        Options are:

            'GRAPE' (default) - GRadient Ascent Pulse Engineering
            'CRAB' - Chopped RAndom Basis

    alg_params : Dictionary
        options that are specific to the algorithm see above

    optim_params : Dictionary
        The key value pairs are the attribute name and value
        used to set attribute values
        Note: attributes are created if they do not exist already,
        and are overwritten if they do.
        Note: method_params are applied afterwards and so may override these

    optim_method : string
        a scipy.optimize.minimize method that will be used to optimise
        the pulse for minimum fidelity error
        Note that FMIN, FMIN_BFGS & FMIN_L_BFGS_B will all result
        in calling these specific scipy.optimize methods
        Note the LBFGSB is equivalent to FMIN_L_BFGS_B for backwards
        capatibility reasons.
        Supplying DEF will given alg dependent result:
            GRAPE - Default optim_method is FMIN_L_BFGS_B
            CRAB - Default optim_method is FMIN

    method_params : dict
        Parameters for the optim_method.
        Note that where there is an attribute of the
        Optimizer object or the termination_conditions matching the key
        that attribute. Otherwise, and in some case also,
        they are assumed to be method_options
        for the scipy.optimize.minimize method.

    optim_alg : string
        Deprecated. Use optim_method.

    max_metric_corr : integer
        Deprecated. Use method_params instead

    accuracy_factor : float
        Deprecated. Use method_params instead

    dyn_type : string
        Dynamics type, i.e. the type of matrix used to describe
        the dynamics. Options are UNIT, GEN_MAT, SYMPL
        (see Dynamics classes for details)

    dyn_params : dict
        Parameters for the Dynamics object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    prop_type : string
        Propagator type i.e. the method used to calculate the
        propagtors and propagtor gradient for each timeslot
        options are DEF, APPROX, DIAG, FRECHET, AUG_MAT
        DEF will use the default for the specific dyn_type
        (see PropagatorComputer classes for details)

    prop_params : dict
        Parameters for the PropagatorComputer object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    fid_type : string
        Fidelity error (and fidelity error gradient) computation method
        Options are DEF, UNIT, TRACEDIFF, TD_APPROX
        DEF will use the default for the specific dyn_type
        (See FidelityComputer classes for details)

    fid_params : dict
        Parameters for the FidelityComputer object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    phase_option : string
        Deprecated. Pass in fid_params instead.

    fid_err_scale_factor : float
        Deprecated. Use scale_factor key in fid_params instead.

    tslot_type : string
        Method for computing the dynamics generators, propagators and
        evolution in the timeslots.
        Options: DEF, UPDATE_ALL, DYNAMIC
        UPDATE_ALL is the only one that currently works
        (See TimeslotComputer classes for details)

    tslot_params : dict
        Parameters for the TimeslotComputer object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    amp_update_mode : string
        Deprecated. Use tslot_type instead.

    init_pulse_type : string
        type / shape of pulse(s) used to initialise the
        the control amplitudes.
        Options (GRAPE) include:
            RND, LIN, ZERO, SINE, SQUARE, TRIANGLE, SAW
        DEF is RND
        (see PulseGen classes for details)
        For the CRAB the this the guess_pulse_type.

    init_pulse_params : dict
        Parameters for the initial / guess pulse generator object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    pulse_scaling : float
        Linear scale factor for generated initial / guess pulses
        By default initial pulses are generated with amplitudes in the
        range (-1.0, 1.0). These will be scaled by this parameter

    pulse_offset : float
        Linear offset for the pulse. That is this value will be added
        to any initial / guess pulses generated.

    ramping_pulse_type : string
        Type of pulse used to modulate the control pulse.
        It's intended use for a ramping modulation, which is often required in
        experimental setups.
        This is only currently implemented in CRAB.
        GAUSSIAN_EDGE was added for this purpose.

    ramping_pulse_params : dict
        Parameters for the ramping pulse generator object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    log_level : integer
        level of messaging output from the logger.
        Options are attributes of qutip.logging_utils,
        in decreasing levels of messaging, are:
        DEBUG_INTENSE, DEBUG_VERBOSE, DEBUG, INFO, WARN, ERROR, CRITICAL
        Anything WARN or above is effectively 'quiet' execution,
        assuming everything runs as expected.
        The default NOTSET implies that the level will be taken from
        the QuTiP settings file, which by default is WARN

    out_file_ext : string or None
        files containing the initial and final control pulse
        amplitudes are saved to the current directory.
        The default name will be postfixed with this extension
        Setting this to None will suppress the output of files

    gen_stats : boolean
        if set to True then statistics for the optimisation
        run will be generated - accessible through attributes
        of the stats object

    Returns
    -------
    opt : OptimResult
        Returns instance of OptimResult, which has attributes giving the
        reason for termination, final fidelity error, final evolution
        final amplitudes, statistics etc
    '
    """
    if log_level == logging.NOTSET:
        log_level = logger.getEffectiveLevel()
    else:
        logger.setLevel(log_level)

    # The parameters types are checked in create_pulse_optimizer
    # so no need to do so here
    # However, the deprecation management is repeated here
    # so that the stack level is correct

    if alg is not 'GRAPE':
        _param_deprecation("Use opt_pulse_crab_unitary for CRAB algorithm.")

    if alg_params is not None:
        _param_deprecation("alg_params is deprecated.")

    if optim_params is not None:
        _param_deprecation("optim_params is deprecated.")

    if prop_params is not None:
        mat_params.update(prop_params)
        _param_deprecation("prop_params is deprecated, "
                           "mat_params replace it for most option.")

    if fid_type is not None:
        mat_params.update(fid_type)
        _param_deprecation("fid_type is deprecated.")

    if ramping_pulse_type is not None:
        _param_deprecation("ramping_pulse_type is only used for CRAB.")

    if ramping_pulse_params is not None:
        _param_deprecation("ramping_pulse_params is only used for CRAB.")

    if pulse_scaling is not None:
        init_pulse_params["scaling"] = pulse_scaling
        _param_deprecation("pulse_scaling is now part of init_pulse_params.")

    if pulse_offset is not None:
        init_pulse_params["offset"] = pulse_offset
        _param_deprecation("pulse_offset is now part of init_pulse_params.")

    if dyn_type is not None:
        if dyn_type == "SYMPL":
            symplectic = "def"
        _param_deprecation("dyn_type is deprecated, "
                           "use symplectic='preop'/'postop' "
                           "or optimize_pulse_unitary if needed.")

    if dyn_params is not None:
        if symplectic =="def" and '_phase_application' in dyn_params:
            symplectic = dyn_params['_phase_application']
        _param_deprecation("dyn_params is deprecated.")

    if symplectic:
        try:
            H = td_Qobj(drift)
            N = H.cte.shape[0]//2
            H_td = True
        except:
            _check_drift_dyn_gen(drift)
            N = H_d[0].shape[0]//2
            H_td = False
        omg = calc_omega(N)
        if symplectic == 'preop':
            if H_td:
                drift = omg*H
            else:
                drift = [omg*H_ for H_ in drift]
            try:
                ctrls = [omg*td_Qobj(H_) for H_ in ctrls ]
            except:
                ctrls = np.array([[omg * H_ for H_ in H__]
                                    for H__ in _check_ctrls_container(ctrls)])
        else: #if symplectic == 'preop':
            omg *= -1
            if H_td:
                drift = H*omg
            else:
                drift = [H_*omg for H_ in drift]
            try:
                ctrls = [td_Qobj(H_)*omg for H_ in ctrls ]
            except:
                ctrls = np.array([[ H_*omg for H_ in H__]
                                    for H__ in _check_ctrls_container(ctrls)])

    """optim = create_pulse_optimizer(
        drift, ctrls, initial, target,
        num_tslots=num_tslots, evo_time=evo_time, tau=tau,
        amp_lbound=amp_lbound, amp_ubound=amp_ubound,
        fid_err_targ=fid_err_targ, min_grad=min_grad,
        max_iter=max_iter, max_wall_time=max_wall_time,
        alg=alg, alg_params=alg_params, optim_params=optim_params,
        optim_method=optim_method, method_params=method_params,
        dyn_type=dyn_type, dyn_params=dyn_params,
        prop_type=prop_type, prop_params=prop_params,
        fid_type=fid_type, fid_params=fid_params,
        init_pulse_type=init_pulse_type, init_pulse_params=init_pulse_params,
        pulse_scaling=pulse_scaling, pulse_offset=pulse_offset,
        ramping_pulse_type=ramping_pulse_type,
        ramping_pulse_params=ramping_pulse_params,
        log_level=log_level, gen_stats=gen_stats)"""

    dyn = create_pulse_optimizer(
          drift, ctrls, initial, target,
          num_tslots=num_tslots, evo_time=evo_time, tau=tau,
          amp_lbound=amp_lbound, amp_ubound=amp_ubound,
          fid_err_targ=fid_err_targ, min_grad=min_grad,
          max_iter=max_iter, max_wall_time=max_wall_time,
          alg='GRAPE',

          optim_method=optim_method, method_params=method_params,
          transfer_function_type=transfer_function_type,
          transfer_function_params=transfer_function_params,
          fid_params=fid_params,
          tslot_type=tslot_type, tslot_params=tslot_params,
          prop_type=prop_type, mat_type=mat_type, mat_params=mat_params,
          init_pulse_type=init_pulse_type, init_pulse_params=init_pulse_params,

          #log_level=logging.NOTSET,
          gen_stats=False):

    dyn = optim.dynamics

    dyn.init_timeslots()
    # Generate initial pulses for each control
    init_amps = np.zeros([dyn.num_tslots, dyn.num_ctrls])

    if alg == 'CRAB':
        for j in range(dyn.num_ctrls):
            pgen = optim.pulse_generator[j]
            pgen.init_pulse()
            init_amps[:, j] = pgen.gen_pulse()
    else:
        pgen = optim.pulse_generator
        for j in range(dyn.num_ctrls):
            init_amps[:, j] = pgen.gen_pulse()

    # Initialise the starting amplitudes
    dyn.initialize_controls(init_amps)

    if log_level <= logging.INFO:
        msg = "System configuration:\n"
        dg_name = "dynamics generator"
        if dyn_type == 'UNIT':
            dg_name = "Hamiltonian"
        if dyn.time_depend_drift:
            msg += "Initial drift {}:\n".format(dg_name)
            msg += str(dyn.drift_dyn_gen[0])
        else:
            msg += "Drift {}:\n".format(dg_name)
            msg += str(dyn.drift_dyn_gen)
        for j in range(dyn.num_ctrls):
            msg += "\nControl {} {}:\n".format(j+1, dg_name)
            msg += str(dyn.ctrl_dyn_gen[j])
        msg += "\nInitial state / operator:\n"
        msg += str(dyn.initial)
        msg += "\nTarget state / operator:\n"
        msg += str(dyn.target)
        logger.info(msg)

    if out_file_ext is not None:
        # Save initial amplitudes to a text file
        pulsefile = "ctrl_amps_initial_" + out_file_ext
        dyn.save_amps(pulsefile)
        if log_level <= logging.INFO:
            logger.info("Initial amplitudes output to file: " + pulsefile)

    # Start the optimisation
    result = optim.run_optimization()

    if out_file_ext is not None:
        # Save final amplitudes to a text file
        pulsefile = "ctrl_amps_final_" + out_file_ext
        dyn.save_amps(pulsefile)
        if log_level <= logging.INFO:
            logger.info("Final amplitudes output to file: " + pulsefile)

    return result

def optimize_pulse_unitary(
        H_d, H_c, U_0, U_targ,
        num_tslots=None, evo_time=None, tau=None,
        amp_lbound=None, amp_ubound=None,
        fid_err_targ=1e-10, min_grad=1e-10, max_iter=500, max_wall_time=180,

        optim_method='DEF', method_params=None,
        prop_type='DEF', mat_type='DEF', mat_params={},
        fid_params=None,
        tslot_type='DEF', tslot_params=None,
        init_pulse_type='DEF', init_pulse_params=None,
        transfer_function_type='DEF', transfer_function_params={},

        log_level=logging.NOTSET, out_file_ext=None, gen_stats=False,
        phase_option='PSU', alg='GRAPE',
        #old
        alg_params=None, optim_params=None,
        dyn_params=None, prop_params=None,
        ramping_pulse_type=None, ramping_pulse_params=None,
        pulse_scaling=1.0, pulse_offset=0.0
        # already old
        # optim_alg=None, max_metric_corr=None, accuracy_factor=None,
        # amp_update_mode=None,
        ):

    """
    Optimise a control pulse to minimise the fidelity error, assuming that
    the dynamics of the system are generated by unitary operators.
    This function is simply a wrapper for optimize_pulse, where the
    appropriate options for unitary dynamics are chosen and the parameter
    names are in the format familiar to unitary dynamics
    The dynamics of the system  in any given timeslot are governed
    by the combined Hamiltonian,
    i.e. the sum of the H_d + ctrl_amp[j]*H_c[j]
    The control pulse is an [n_ts, n_ctrls] array of piecewise amplitudes
    Starting from an intital (typically random) pulse,
    a multivariable optimisation algorithm attempts to determines the
    optimal values for the control pulse to minimise the fidelity error
    The maximum fidelity for a unitary system is 1, i.e. when the
    time evolution resulting from the pulse is equivalent to the target.
    And therefore the fidelity error is 1 - fidelity

    Parameters
    ----------
    H_d : Qobj or list of Qobj
        Drift (aka system) the underlying Hamiltonian of the system
        can provide list (of length num_tslots) for time dependent drift

    H_c : List of Qobj or array like [num_tslots, evo_time]
        a list of control Hamiltonians. These are scaled by
        the amplitudes to alter the overall dynamics
        Array like imput can be provided for time dependent control generators

    U_0 : Qobj
        starting point for the evolution.
        Typically the identity matrix

    U_targ : Qobj
        target transformation, e.g. gate or state, for the time evolution

    num_tslots : integer or None
        number of timeslots.
        None implies that timeslots will be given in the tau array

    evo_time : float or None
        total time for the evolution
        None implies that timeslots will be given in the tau array

    tau : array[num_tslots] of floats or None
        durations for the timeslots.
        if this is given then num_tslots and evo_time are dervived
        from it
        None implies that timeslot durations will be equal and
        calculated as evo_time/num_tslots

    amp_lbound : float or list of floats
        lower boundaries for the control amplitudes
        Can be a scalar value applied to all controls
        or a list of bounds for each control

    amp_ubound : float or list of floats
        upper boundaries for the control amplitudes
        Can be a scalar value applied to all controls
        or a list of bounds for each control

    fid_err_targ : float
        Fidelity error target. Pulse optimisation will
        terminate when the fidelity error falls below this value

    mim_grad : float
        Minimum gradient. When the sum of the squares of the
        gradients wrt to the control amplitudes falls below this
        value, the optimisation terminates, assuming local minima

    max_iter : integer
        Maximum number of iterations of the optimisation algorithm

    max_wall_time : float
        Maximum allowed elapsed time for the  optimisation algorithm

    alg : string
        Algorithm to use in pulse optimisation.
        Options are:
            'GRAPE' (default) - GRadient Ascent Pulse Engineering
            'CRAB' - Chopped RAndom Basis

    alg_params : Dictionary
        options that are specific to the algorithm see above

    optim_params : Dictionary
        The key value pairs are the attribute name and value
        used to set attribute values
        Note: attributes are created if they do not exist already,
        and are overwritten if they do.
        Note: method_params are applied afterwards and so may override these

    optim_method : string
        a scipy.optimize.minimize method that will be used to optimise
        the pulse for minimum fidelity error
        Note that FMIN, FMIN_BFGS & FMIN_L_BFGS_B will all result
        in calling these specific scipy.optimize methods
        Note the LBFGSB is equivalent to FMIN_L_BFGS_B for backwards
        capatibility reasons.
        Supplying DEF will given alg dependent result:

            GRAPE - Default optim_method is FMIN_L_BFGS_B
            CRAB - Default optim_method is FMIN

    method_params : dict
        Parameters for the optim_method.
        Note that where there is an attribute of the
        Optimizer object or the termination_conditions matching the key
        that attribute. Otherwise, and in some case also,
        they are assumed to be method_options
        for the scipy.optimize.minimize method.

    optim_alg : string
        Deprecated. Use optim_method.

    max_metric_corr : integer
        Deprecated. Use method_params instead

    accuracy_factor : float
        Deprecated. Use method_params instead

    phase_option : string
        determines how global phase is treated in fidelity
        calculations (fid_type='UNIT' only). Options:

            PSU - global phase ignored
            SU - global phase included

    dyn_params : dict
        Parameters for the Dynamics object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    prop_params : dict
        Parameters for the PropagatorComputer object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    fid_params : dict
        Parameters for the FidelityComputer object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    tslot_type : string
        Method for computing the dynamics generators, propagators and
        evolution in the timeslots.
        Options: DEF, UPDATE_ALL, DYNAMIC
        UPDATE_ALL is the only one that currently works
        (See TimeslotComputer classes for details)

    tslot_params : dict
        Parameters for the TimeslotComputer object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    amp_update_mode : string
        Deprecated. Use tslot_type instead.

    init_pulse_type : string
        type / shape of pulse(s) used to initialise the
        the control amplitudes.
        Options (GRAPE) include:

            RND, LIN, ZERO, SINE, SQUARE, TRIANGLE, SAW
            DEF is RND

        (see PulseGen classes for details)
        For the CRAB the this the guess_pulse_type.

    init_pulse_params : dict
        Parameters for the initial / guess pulse generator object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    pulse_scaling : float
        Linear scale factor for generated initial / guess pulses
        By default initial pulses are generated with amplitudes in the
        range (-1.0, 1.0). These will be scaled by this parameter

    pulse_offset : float
        Linear offset for the pulse. That is this value will be added
        to any initial / guess pulses generated.

    ramping_pulse_type : string
        Type of pulse used to modulate the control pulse.
        It's intended use for a ramping modulation, which is often required in
        experimental setups.
        This is only currently implemented in CRAB.
        GAUSSIAN_EDGE was added for this purpose.

    ramping_pulse_params : dict
        Parameters for the ramping pulse generator object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    log_level : integer
        level of messaging output from the logger.
        Options are attributes of qutip.logging_utils,
        in decreasing levels of messaging, are:
        DEBUG_INTENSE, DEBUG_VERBOSE, DEBUG, INFO, WARN, ERROR, CRITICAL
        Anything WARN or above is effectively 'quiet' execution,
        assuming everything runs as expected.
        The default NOTSET implies that the level will be taken from
        the QuTiP settings file, which by default is WARN

    out_file_ext : string or None
        files containing the initial and final control pulse
        amplitudes are saved to the current directory.
        The default name will be postfixed with this extension
        Setting this to None will suppress the output of files

    gen_stats : boolean
        if set to True then statistics for the optimisation
        run will be generated - accessible through attributes
        of the stats object

    Returns
    -------
    opt : OptimResult
        Returns instance of OptimResult, which has attributes giving the
        reason for termination, final fidelity error, final evolution
        final amplitudes, statistics etc
    '
    """

    # parameters are checked in create pulse optimiser

    # The deprecation management is repeated here
    # so that the stack level is correct
    if alg is not 'GRAPE':
        _param_deprecation("Use opt_pulse_crab_unitary for CRAB algorithm.")

    if alg_params is not None:
        _param_deprecation("alg_params is deprecated.")

    if optim_params is not None:
        _param_deprecation("optim_params is deprecated.")

    if dyn_params is not None:
        _param_deprecation("dyn_params is deprecated.")

    if prop_params is not None:
        mat_params.update(prop_params)
        _param_deprecation("prop_params is deprecated, "
                           "mat_params replace it for most option.")

    if ramping_pulse_type is not None:
        _param_deprecation("ramping_pulse_type is only used for CRAB.")

    if ramping_pulse_params is not None:
        _param_deprecation("ramping_pulse_params is only used for CRAB.")

    if pulse_scaling is not None:
        init_pulse_params["scaling"] = pulse_scaling
        _param_deprecation("pulse_scaling is now part of init_pulse_params.")

    if pulse_offset is not None:
        init_pulse_params["offset"] = pulse_offset
        _param_deprecation("pulse_offset is now part of init_pulse_params.")

    # phase_option is still valid for this method
    # pass it via the fid_params
    if not phase_option is None:
        if fid_params is None:
            fid_params = {'phase_option':phase_option}
        else:
            if not 'phase_option' in fid_params:
                fid_params['phase_option'] = phase_option

    try:
        H_d = td_Qobj(H_d)
        H_d *= -1j
    except:
        _check_drift_dyn_gen(H_d)
        for H_i in H_d:
            H_i *= -1j

    try:
        ctrls = [td_Qobj(H_)* -1j for H_ in H_c ]
    except:
        ctrls = _check_ctrls_container(H_c) * -1j

    dyn =  create_pulse_optimizer(
            drift=H_d, ctrls=ctrls, initial=U_0, target=U_targ,
            num_tslots=num_tslots, evo_time=evo_time, tau=tau,
            amp_lbound=amp_lbound, amp_ubound=amp_ubound,
            fid_err_targ=fid_err_targ, min_grad=min_grad,
            max_iter=max_iter, max_wall_time=max_wall_time,
            alg=alg,
            optim_method=optim_method, method_params=method_params,
            transfer_function_type=transfer_function_type,
            transfer_function_params=transfer_function_params,
            prop_type=prop_type, mat_type=mat_type, mat_params=mat_params,
            fid_params=fid_params,
            init_pulse_type=init_pulse_type, init_pulse_params=init_pulse_params,
            #log_level=log_level, out_file_ext=out_file_ext,
            gen_stats=gen_stats)

def opt_pulse_crab(
        drift, ctrls, initial, target,
        num_tslots=None, evo_time=None, tau=None,
        amp_lbound=None, amp_ubound=None,
        fid_err_targ=1e-5,
        max_iter=500, max_wall_time=180,
        alg_params=None,
        num_coeffs=None, init_coeff_scaling=1.0,
        optim_params=None, optim_method='fmin', method_params=None,
        dyn_type='GEN_MAT', dyn_params=None,
        prop_type='DEF', prop_params=None,
        fid_type='DEF', fid_params=None,
        tslot_type='DEF', tslot_params=None,
        guess_pulse_type=None, guess_pulse_params=None,
        guess_pulse_scaling=1.0, guess_pulse_offset=0.0,
        guess_pulse_action='MODULATE',
        ramping_pulse_type=None, ramping_pulse_params=None,
        log_level=logging.NOTSET, out_file_ext=None, gen_stats=False):
    """
    Optimise a control pulse to minimise the fidelity error.
    The dynamics of the system in any given timeslot are governed
    by the combined dynamics generator,
    i.e. the sum of the drift+ctrl_amp[j]*ctrls[j]
    The control pulse is an [n_ts, n_ctrls] array of piecewise amplitudes.
    The CRAB algorithm uses basis function coefficents as the variables to
    optimise. It does NOT use any gradient function.
    A multivariable optimisation algorithm attempts to determines the
    optimal values for the control pulse to minimise the fidelity error
    The fidelity error is some measure of distance of the system evolution
    from the given target evolution in the time allowed for the evolution.

    Parameters
    ----------
    drift : Qobj or list of Qobj
        the underlying dynamics generator of the system
        can provide list (of length num_tslots) for time dependent drift

    ctrls : List of Qobj or array like [num_tslots, evo_time]
        a list of control dynamics generators. These are scaled by
        the amplitudes to alter the overall dynamics
        Array like imput can be provided for time dependent control generators

    initial : Qobj
        starting point for the evolution.
        Typically the identity matrix

    target : Qobj
        target transformation, e.g. gate or state, for the time evolution

    num_tslots : integer or None
        number of timeslots.
        None implies that timeslots will be given in the tau array

    evo_time : float or None
        total time for the evolution
        None implies that timeslots will be given in the tau array

    tau : array[num_tslots] of floats or None
        durations for the timeslots.
        if this is given then num_tslots and evo_time are dervived
        from it
        None implies that timeslot durations will be equal and
        calculated as evo_time/num_tslots

    amp_lbound : float or list of floats
        lower boundaries for the control amplitudes
        Can be a scalar value applied to all controls
        or a list of bounds for each control

    amp_ubound : float or list of floats
        upper boundaries for the control amplitudes
        Can be a scalar value applied to all controls
        or a list of bounds for each control

    fid_err_targ : float
        Fidelity error target. Pulse optimisation will
        terminate when the fidelity error falls below this value

    max_iter : integer
        Maximum number of iterations of the optimisation algorithm

    max_wall_time : float
        Maximum allowed elapsed time for the  optimisation algorithm

    alg_params : Dictionary
        options that are specific to the algorithm see above

    optim_params : Dictionary
        The key value pairs are the attribute name and value
        used to set attribute values
        Note: attributes are created if they do not exist already,
        and are overwritten if they do.
        Note: method_params are applied afterwards and so may override these

    coeff_scaling : float
        Linear scale factor for the random basis coefficients
        By default these range from -1.0 to 1.0
        Note this is overridden by alg_params (if given there)

    num_coeffs : integer
        Number of coefficients used for each basis function
        Note this is calculated automatically based on the dimension of the
        dynamics if not given. It is crucial to the performane of the
        algorithm that it is set as low as possible, while still giving
        high enough frequencies.
        Note this is overridden by alg_params (if given there)

    optim_method : string
        Multi-variable optimisation method
        The only tested options are 'fmin' and 'Nelder-mead'
        In theory any non-gradient method implemented in
        scipy.optimize.mininize could be used.

    method_params : dict
        Parameters for the optim_method.
        Note that where there is an attribute of the
        Optimizer object or the termination_conditions matching the key
        that attribute. Otherwise, and in some case also,
        they are assumed to be method_options
        for the scipy.optimize.minimize method.
        The commonly used parameter are:
            xtol - limit on variable change for convergence
            ftol - limit on fidelity error change for convergence

    dyn_type : string
        Dynamics type, i.e. the type of matrix used to describe
        the dynamics. Options are UNIT, GEN_MAT, SYMPL
        (see Dynamics classes for details)

    dyn_params : dict
        Parameters for the Dynamics object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    prop_type : string
        Propagator type i.e. the method used to calculate the
        propagtors and propagtor gradient for each timeslot
        options are DEF, APPROX, DIAG, FRECHET, AUG_MAT
        DEF will use the default for the specific dyn_type
        (see PropagatorComputer classes for details)

    prop_params : dict
        Parameters for the PropagatorComputer object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    fid_type : string
        Fidelity error (and fidelity error gradient) computation method
        Options are DEF, UNIT, TRACEDIFF, TD_APPROX
        DEF will use the default for the specific dyn_type
        (See FidelityComputer classes for details)

    fid_params : dict
        Parameters for the FidelityComputer object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    tslot_type : string
        Method for computing the dynamics generators, propagators and
        evolution in the timeslots.
        Options: DEF, UPDATE_ALL, DYNAMIC
        UPDATE_ALL is the only one that currently works
        (See TimeslotComputer classes for details)

    tslot_params : dict
        Parameters for the TimeslotComputer object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    guess_pulse_type : string
        type / shape of pulse(s) used modulate the control amplitudes.
        Options include:
            RND, LIN, ZERO, SINE, SQUARE, TRIANGLE, SAW, GAUSSIAN
        Default is None

    guess_pulse_params : dict
        Parameters for the guess pulse generator object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    guess_pulse_action : string
        Determines how the guess pulse is applied to the pulse generated
        by the basis expansion.
        Options are: MODULATE, ADD
        Default is MODULATE

    pulse_scaling : float
        Linear scale factor for generated guess pulses
        By default initial pulses are generated with amplitudes in the
        range (-1.0, 1.0). These will be scaled by this parameter

    pulse_offset : float
        Linear offset for the pulse. That is this value will be added
        to any guess pulses generated.

    ramping_pulse_type : string
        Type of pulse used to modulate the control pulse.
        It's intended use for a ramping modulation, which is often required in
        experimental setups.
        This is only currently implemented in CRAB.
        GAUSSIAN_EDGE was added for this purpose.

    ramping_pulse_params : dict
        Parameters for the ramping pulse generator object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    log_level : integer
        level of messaging output from the logger.
        Options are attributes of qutip.logging_utils,
        in decreasing levels of messaging, are:
        DEBUG_INTENSE, DEBUG_VERBOSE, DEBUG, INFO, WARN, ERROR, CRITICAL
        Anything WARN or above is effectively 'quiet' execution,
        assuming everything runs as expected.
        The default NOTSET implies that the level will be taken from
        the QuTiP settings file, which by default is WARN

    out_file_ext : string or None
        files containing the initial and final control pulse
        amplitudes are saved to the current directory.
        The default name will be postfixed with this extension
        Setting this to None will suppress the output of files

    gen_stats : boolean
        if set to True then statistics for the optimisation
        run will be generated - accessible through attributes
        of the stats object

    Returns
    -------
    opt : OptimResult
        Returns instance of OptimResult, which has attributes giving the
        reason for termination, final fidelity error, final evolution
        final amplitudes, statistics etc
    '
    """

    # The parameters are checked in create_pulse_optimizer
    # so no need to do so here

    if log_level == logging.NOTSET:
        log_level = logger.getEffectiveLevel()
    else:
        logger.setLevel(log_level)

    # build the algorithm options
    if not isinstance(alg_params, dict):
        alg_params = {'num_coeffs':num_coeffs,
                       'init_coeff_scaling':init_coeff_scaling}
    else:
        if (num_coeffs is not None and
            not 'num_coeffs' in alg_params):
            alg_params['num_coeffs'] = num_coeffs
        if (init_coeff_scaling is not None and
            not 'init_coeff_scaling' in alg_params):
            alg_params['init_coeff_scaling'] = init_coeff_scaling

    # Build the guess pulse options
    # Any options passed in the guess_pulse_params take precedence
    # over the parameter values.
    if guess_pulse_type:
        if not isinstance(guess_pulse_params, dict):
            guess_pulse_params = {}
        if (guess_pulse_scaling is not None and
            not 'scaling' in guess_pulse_params):
            guess_pulse_params['scaling'] = guess_pulse_scaling
        if (guess_pulse_offset is not None and
            not 'offset' in guess_pulse_params):
            guess_pulse_params['offset'] = guess_pulse_offset
        if (guess_pulse_action is not None and
            not 'pulse_action' in guess_pulse_params):
            guess_pulse_params['pulse_action'] = guess_pulse_action

    return optimize_pulse(
        drift, ctrls, initial, target,
        num_tslots=num_tslots, evo_time=evo_time, tau=tau,
        amp_lbound=amp_lbound, amp_ubound=amp_ubound,
        fid_err_targ=fid_err_targ, min_grad=0.0,
        max_iter=max_iter, max_wall_time=max_wall_time,
        alg='CRAB', alg_params=alg_params, optim_params=optim_params,
        optim_method=optim_method, method_params=method_params,
        dyn_type=dyn_type, dyn_params=dyn_params,
        prop_type=prop_type, prop_params=prop_params,
        fid_type=fid_type, fid_params=fid_params,
        tslot_type=tslot_type, tslot_params=tslot_params,
        init_pulse_type=guess_pulse_type,
        init_pulse_params=guess_pulse_params,
        ramping_pulse_type=ramping_pulse_type,
        ramping_pulse_params=ramping_pulse_params,
        log_level=log_level, out_file_ext=out_file_ext, gen_stats=gen_stats)

def opt_pulse_crab_unitary(
        H_d, H_c, U_0, U_targ,
        num_tslots=None, evo_time=None, tau=None,
        amp_lbound=None, amp_ubound=None,
        fid_err_targ=1e-5,
        max_iter=500, max_wall_time=180,
        alg_params=None,
        num_coeffs=None, init_coeff_scaling=1.0,
        optim_params=None, optim_method='fmin', method_params=None,
        phase_option='PSU',
        dyn_params=None, prop_params=None, fid_params=None,
        tslot_type='DEF', tslot_params=None,
        guess_pulse_type=None, guess_pulse_params=None,
        guess_pulse_scaling=1.0, guess_pulse_offset=0.0,
        guess_pulse_action='MODULATE',
        ramping_pulse_type=None, ramping_pulse_params=None,
        log_level=logging.NOTSET, out_file_ext=None, gen_stats=False):
    """
    Optimise a control pulse to minimise the fidelity error, assuming that
    the dynamics of the system are generated by unitary operators.
    This function is simply a wrapper for optimize_pulse, where the
    appropriate options for unitary dynamics are chosen and the parameter
    names are in the format familiar to unitary dynamics
    The dynamics of the system  in any given timeslot are governed
    by the combined Hamiltonian,
    i.e. the sum of the H_d + ctrl_amp[j]*H_c[j]
    The control pulse is an [n_ts, n_ctrls] array of piecewise amplitudes

    The CRAB algorithm uses basis function coefficents as the variables to
    optimise. It does NOT use any gradient function.
    A multivariable optimisation algorithm attempts to determines the
    optimal values for the control pulse to minimise the fidelity error
    The fidelity error is some measure of distance of the system evolution
    from the given target evolution in the time allowed for the evolution.

    Parameters
    ----------

    H_d : Qobj or list of Qobj
        Drift (aka system) the underlying Hamiltonian of the system
        can provide list (of length num_tslots) for time dependent drift

    H_c : List of Qobj or array like [num_tslots, evo_time]
        a list of control Hamiltonians. These are scaled by
        the amplitudes to alter the overall dynamics
        Array like imput can be provided for time dependent control generators

    U_0 : Qobj
        starting point for the evolution.
        Typically the identity matrix

    U_targ : Qobj
        target transformation, e.g. gate or state, for the time evolution

    num_tslots : integer or None
        number of timeslots.
        None implies that timeslots will be given in the tau array

    evo_time : float or None
        total time for the evolution
        None implies that timeslots will be given in the tau array

    tau : array[num_tslots] of floats or None
        durations for the timeslots.
        if this is given then num_tslots and evo_time are dervived
        from it
        None implies that timeslot durations will be equal and
        calculated as evo_time/num_tslots

    amp_lbound : float or list of floats
        lower boundaries for the control amplitudes
        Can be a scalar value applied to all controls
        or a list of bounds for each control

    amp_ubound : float or list of floats
        upper boundaries for the control amplitudes
        Can be a scalar value applied to all controls
        or a list of bounds for each control

    fid_err_targ : float
        Fidelity error target. Pulse optimisation will
        terminate when the fidelity error falls below this value

    max_iter : integer
        Maximum number of iterations of the optimisation algorithm

    max_wall_time : float
        Maximum allowed elapsed time for the  optimisation algorithm

    alg_params : Dictionary
        options that are specific to the algorithm see above

    optim_params : Dictionary
        The key value pairs are the attribute name and value
        used to set attribute values
        Note: attributes are created if they do not exist already,
        and are overwritten if they do.
        Note: method_params are applied afterwards and so may override these

    coeff_scaling : float
        Linear scale factor for the random basis coefficients
        By default these range from -1.0 to 1.0
        Note this is overridden by alg_params (if given there)

    num_coeffs : integer
        Number of coefficients used for each basis function
        Note this is calculated automatically based on the dimension of the
        dynamics if not given. It is crucial to the performane of the
        algorithm that it is set as low as possible, while still giving
        high enough frequencies.
        Note this is overridden by alg_params (if given there)

    optim_method : string
        Multi-variable optimisation method
        The only tested options are 'fmin' and 'Nelder-mead'
        In theory any non-gradient method implemented in
        scipy.optimize.mininize could be used.

    method_params : dict
        Parameters for the optim_method.
        Note that where there is an attribute of the
        Optimizer object or the termination_conditions matching the key
        that attribute. Otherwise, and in some case also,
        they are assumed to be method_options
        for the scipy.optimize.minimize method.
        The commonly used parameter are:
            xtol - limit on variable change for convergence
            ftol - limit on fidelity error change for convergence

    phase_option : string
        determines how global phase is treated in fidelity
        calculations (fid_type='UNIT' only). Options:
            PSU - global phase ignored
            SU - global phase included

    dyn_params : dict
        Parameters for the Dynamics object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    prop_params : dict
        Parameters for the PropagatorComputer object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    fid_params : dict
        Parameters for the FidelityComputer object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    tslot_type : string
        Method for computing the dynamics generators, propagators and
        evolution in the timeslots.
        Options: DEF, UPDATE_ALL, DYNAMIC
        UPDATE_ALL is the only one that currently works
        (See TimeslotComputer classes for details)

    tslot_params : dict
        Parameters for the TimeslotComputer object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    guess_pulse_type : string
        type / shape of pulse(s) used modulate the control amplitudes.
        Options include:
            RND, LIN, ZERO, SINE, SQUARE, TRIANGLE, SAW, GAUSSIAN
        Default is None

    guess_pulse_params : dict
        Parameters for the guess pulse generator object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    guess_pulse_action : string
        Determines how the guess pulse is applied to the pulse generated
        by the basis expansion.
        Options are: MODULATE, ADD
        Default is MODULATE

    pulse_scaling : float
        Linear scale factor for generated guess pulses
        By default initial pulses are generated with amplitudes in the
        range (-1.0, 1.0). These will be scaled by this parameter

    pulse_offset : float
        Linear offset for the pulse. That is this value will be added
        to any guess pulses generated.

    ramping_pulse_type : string
        Type of pulse used to modulate the control pulse.
        It's intended use for a ramping modulation, which is often required in
        experimental setups.
        This is only currently implemented in CRAB.
        GAUSSIAN_EDGE was added for this purpose.

    ramping_pulse_params : dict
        Parameters for the ramping pulse generator object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    log_level : integer
        level of messaging output from the logger.
        Options are attributes of qutip.logging_utils,
        in decreasing levels of messaging, are:
        DEBUG_INTENSE, DEBUG_VERBOSE, DEBUG, INFO, WARN, ERROR, CRITICAL
        Anything WARN or above is effectively 'quiet' execution,
        assuming everything runs as expected.
        The default NOTSET implies that the level will be taken from
        the QuTiP settings file, which by default is WARN

    out_file_ext : string or None
        files containing the initial and final control pulse
        amplitudes are saved to the current directory.
        The default name will be postfixed with this extension
        Setting this to None will suppress the output of files

    gen_stats : boolean
        if set to True then statistics for the optimisation
        run will be generated - accessible through attributes
        of the stats object

    Returns
    -------
    opt : OptimResult
        Returns instance of OptimResult, which has attributes giving the
        reason for termination, final fidelity error, final evolution
        final amplitudes, statistics etc
    '
    """

    # The parameters are checked in create_pulse_optimizer
    # so no need to do so here

    if log_level == logging.NOTSET:
        log_level = logger.getEffectiveLevel()
    else:
        logger.setLevel(log_level)

    # build the algorithm options
    if not isinstance(alg_params, dict):
        alg_params = {'num_coeffs':num_coeffs,
                       'init_coeff_scaling':init_coeff_scaling}
    else:
        if (num_coeffs is not None and
            not 'num_coeffs' in alg_params):
            alg_params['num_coeffs'] = num_coeffs
        if (init_coeff_scaling is not None and
            not 'init_coeff_scaling' in alg_params):
            alg_params['init_coeff_scaling'] = init_coeff_scaling

    # Build the guess pulse options
    # Any options passed in the guess_pulse_params take precedence
    # over the parameter values.
    if guess_pulse_type:
        if not isinstance(guess_pulse_params, dict):
            guess_pulse_params = {}
        if (guess_pulse_scaling is not None and
            not 'scaling' in guess_pulse_params):
            guess_pulse_params['scaling'] = guess_pulse_scaling
        if (guess_pulse_offset is not None and
            not 'offset' in guess_pulse_params):
            guess_pulse_params['offset'] = guess_pulse_offset
        if (guess_pulse_action is not None and
            not 'pulse_action' in guess_pulse_params):
            guess_pulse_params['pulse_action'] = guess_pulse_action

    return optimize_pulse_unitary(
        H_d, H_c, U_0, U_targ,
        num_tslots=num_tslots, evo_time=evo_time, tau=tau,
        amp_lbound=amp_lbound, amp_ubound=amp_ubound,
        fid_err_targ=fid_err_targ, min_grad=0.0,
        max_iter=max_iter, max_wall_time=max_wall_time,
        alg='CRAB', alg_params=alg_params, optim_params=optim_params,
        optim_method=optim_method, method_params=method_params,
        phase_option=phase_option,
        dyn_params=dyn_params, prop_params=prop_params, fid_params=fid_params,
        tslot_type=tslot_type, tslot_params=tslot_params,
        init_pulse_type=guess_pulse_type,
        init_pulse_params=guess_pulse_params,
        ramping_pulse_type=ramping_pulse_type,
        ramping_pulse_params=ramping_pulse_params,
        log_level=log_level, out_file_ext=out_file_ext, gen_stats=gen_stats)



def create_pulse_optimizer(
        drift, ctrls, initial, target,
        num_tslots=None, evo_time=None, tau=None,
        amp_lbound=None, amp_ubound=None,
        fid_err_targ=1e-10, min_grad=1e-10, max_iter=500, max_wall_time=180,
        alg='GRAPE', crab_params={},

        optim_method='DEF', method_params={},
        transfer_function_type='DEF', transfer_function_params={},
        fid_type='DEF', fid_params={},
        tslot_type='DEF', tslot_params={},
        prop_type='DEF', mat_type='DEF', mat_params={},

        init_pulse='DEF', init_pulse_params=None,
        ramping_pulse='DEF', ramping_pulse_params=None,

        #log_level=logging.NOTSET,
        gen_stats=False):

    """
    Generate the objects of the appropriate subclasses
    required for the pulse optmisation based on the parameters given
    Note this method may be preferable to calling optimize_pulse
    if more detailed configuration is required before running the
    optmisation algorthim, or the algorithm will be run many times,
    for instances when trying to finding global the optimum or
    minimum time optimisation

    Parameters
    ----------
    drift : Qobj or list of Qobj
        the underlying dynamics generator of the system
        can provide list (of length num_tslots) for time dependent drift

    ctrls : List of Qobj or array like [num_tslots, evo_time]
        a list of control dynamics generators. These are scaled by
        the amplitudes to alter the overall dynamics
        Array like imput can be provided for time dependent control generators

    initial : Qobj
        starting point for the evolution.
        Typically the identity matrix

    target : Qobj
        target transformation, e.g. gate or state, for the time evolution

    num_tslots : integer or None
        number of timeslots.
        None implies that timeslots will be given in the tau array

    evo_time : float or None
        total time for the evolution
        None implies that timeslots will be given in the tau array

    tau : array[num_tslots] of floats or None
        durations for the timeslots.
        if this is given then num_tslots and evo_time are dervived
        from it
        None implies that timeslot durations will be equal and
        calculated as evo_time/num_tslots

    amp_lbound : float or list of floats
        lower boundaries for the control amplitudes
        Can be a scalar value applied to all controls
        or a list of bounds for each control

    amp_ubound : float or list of floats
        upper boundaries for the control amplitudes
        Can be a scalar value applied to all controls
        or a list of bounds for each control

    fid_err_targ : float
        Fidelity error target. Pulse optimisation will
        terminate when the fidelity error falls below this value

    mim_grad : float
        Minimum gradient. When the sum of the squares of the
        gradients wrt to the control amplitudes falls below this
        value, the optimisation terminates, assuming local minima

    max_iter : integer
        Maximum number of iterations of the optimisation algorithm

    max_wall_time : float
        Maximum allowed elapsed time for the optimisation algorithm

    alg : string
        Algorithm to use in pulse optimisation.
        Options are:
            'GRAPE' (default) - GRadient Ascent Pulse Engineering
            'CRAB' - Chopped RAndom Basis

    crab_params : Dictionary
        options that are specific to the algorithm see above

    optim_method : string
        a scipy.optimize.minimize method that will be used to optimise
        the pulse for minimum fidelity error
        Note that FMIN, FMIN_BFGS & FMIN_L_BFGS_B will all result
        in calling these specific scipy.optimize methods
        Note the LBFGSB is equivalent to FMIN_L_BFGS_B for backwards
        capatibility reasons.
        Supplying DEF will given alg dependent result:
            - GRAPE - Default optim_method is FMIN_L_BFGS_B
            - CRAB - Default optim_method is Nelder-Mead

    method_params : dict
        Parameters for the optim_method.
        Note that where there is an attribute of the
        Optimizer object or the termination_conditions matching the key
        that attribute. Otherwise, and in some case also,
        they are assumed to be method_options
        for the scipy.optimize.minimize method.

    -----------dyn_params : dict
        Parameters for the Dynamics object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    prop_type : string
        Propagator type i.e. the method used to calculate the
        propagtors and propagtor gradient for each timeslot
        options are DEF, APPROX, DIAG, FRECHET, AUG_MAT
        DEF will use the default for the specific dyn_type
        (see PropagatorComputer classes for details)

    prop_params : dict
        Parameters for the PropagatorComputer object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    fid_type : string
        Fidelity error (and fidelity error gradient) computation method
        Options are DEF, UNIT, TRACEDIFF, TD_APPROX
        DEF will use the default for the specific dyn_type
        (See FidelityComputer classes for details)

    fid_params : dict
        Parameters for the FidelityComputer object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    transfer_function_type :

    transfer_function_params :

    tslot_type : string
        Method for computing the dynamics generators, propagators and
        evolution in the timeslots.
        Options: DEF, UPDATE_ALL, DYNAMIC
        UPDATE_ALL is the only one that currently works
        (See TimeslotComputer classes for details)

    tslot_params : dict
        Parameters for the TimeslotComputer object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    init_pulse_type : string
        type / shape of pulse(s) used to initialise the
        the control amplitudes.
        Options (GRAPE) include:

            RND, LIN, ZERO, SINE, SQUARE, TRIANGLE, SAW
            DEF is RND

        (see PulseGen classes for details)
        For the CRAB the this the guess_pulse_type.

    init_pulse_params : dict
        Parameters for the initial / guess pulse generator object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

        pulse_scaling : float
            Linear scale factor for generated initial / guess pulses
            By default initial pulses are generated with amplitudes in the
            range (-1.0, 1.0). These will be scaled by this parameter

        pulse_offset : float
            Linear offset for the pulse. That is this value will be added
            to any initial / guess pulses generated.

    ramping_pulse_type : string
        Type of pulse used to modulate the control pulse.
        It's intended use for a ramping modulation, which is often required in
        experimental setups.
        This is only currently implemented in CRAB.
        GAUSSIAN_EDGE was added for this purpose.

    ramping_pulse_params : dict
        Parameters for the ramping pulse generator object
        The key value pairs are assumed to be attribute name value pairs
        They applied after the object is created

    log_level : integer
        level of messaging output from the logger.
        Options are attributes of qutip.logging_utils,
        in decreasing levels of messaging, are:
        DEBUG_INTENSE, DEBUG_VERBOSE, DEBUG, INFO, WARN, ERROR, CRITICAL
        Anything WARN or above is effectively 'quiet' execution,
        assuming everything runs as expected.
        The default NOTSET implies that the level will be taken from
        the QuTiP settings file, which by default is WARN

    gen_stats : boolean
        if set to True then statistics for the optimisation
        run will be generated - accessible through attributes
        of the stats object

    Returns
    -------
    dyn : Dynamics
        Instance of a Dynamics, can be ran with the 'run' method.'
    """

    alg = _upper_safe(alg)
    if alg is None:
        raise errors.UsageError(
            "Optimisation algorithm must be specified through 'alg' parameter")
    elif alg == 'GRAPE':
        dyn = dynamic.dynamics()
    elif alg == 'CRAB':
        dyn = dynamic.dynamicsCRAB()
    else:
        raise errors.UsageError(
            "No option for pulse optimisation algorithm alg={}".format(alg))

    if True: # physics, times, bound and cost section

        dyn.set_physic(H=drift, ctrls=ctrls, target=rho_t, initial=initial)

        # Set the FidelityComputer options
        # The default will be typically be the best option
        # fid_type is no longer used as the type is set by target.
        if "mode" in fid_params:
            mode = fid_params["mode"]
        else:
            mode = None
        if "early" in fid_params:
            early = fid_params["early"]
        else:
            early = False
        if "weight" in fid_params:
            weight = fid_params["weight"]
        else:
            weight = None
        dyn.set_cost(self, mode=mode, early=early, weight=weight)

        num_tslots=None, evo_time=None, tau=None,
        if num_tslots is not None and evo_time is not None:
            times = np.linspace(0,evo_time,num_tslots+1)
        elif:
            times = np.cumsum(np.insert(tau,0,0))
        else:
            raise errors.UsageError(
                "Either the timeslot durations should be supplied as an "
                "array 'tau' or the number of timeslots 'num_tslots' "
                "and the evolution time 'evo_time' must be given.")
        dyn.set_times(times)
        dyn.set_amp_bound(amp_lbound, amp_ubound)

    if True: # matrix prop tlots section
        # Set the matrix type
        mat_type = _upper_safe(mat_type)
        if mat_type == 'DEF' or mat_type is None or mat_type == '':
            # Do nothing use the default for the Dynamics
            mat_mode = ""
        elif mat_type == 'SPARSE':
            mat_mode = "sparse"
        elif mat_type == 'DENSE':
            mat_mode = "dense"
        elif mat_type == 'DENSE':
            mat_mode = "mixed"
        else:
            raise Exception("No option for mat_type mat_type={}".format(mat_type))

        # Set the tslot computer option
        tslot_type = _upper_safe(tslot_type)
        if tslot_type == 'DEF' or tslot_type is None or tslot_type == '':
            # Do nothing use the default for the Dynamics
            tslot_mode = ""
        elif tslot_type in ['FULL', 'SAVE']:
            tslot_mode = "full"
        elif tslot_type in ['INT', "INTEGRATION"]:
            tslot_mode = "int"
        elif tslot_type in ['POWER', 'RECOMPUTE', 'MEM']:
            tslot_mode = "power"
        else:
            raise Exception("No option for mat_type mat_type={}".format(mat_type))

        # Set the Propagator Computation option, now computed in the control_matrix
        prop_type = _upper_safe(prop_type)
        if prop_type == 'DEF' or prop_type is None or prop_type == '':
            # Do nothing use the default for the Dynamics
            pass
        elif prop_type == 'APPROX':
            dyn.matrix_options["method"] = "approx"
            # dyn.matrix_options["epsilon"]
            dyn.matrix_options["_mem_prop"] = True
            dyn.matrix_options["sparse_exp"] = True
        elif prop_type == 'DIAG':
            dyn.matrix_options["method"] = "spectral"
            dyn.matrix_options["sparse2dense"] = True
            # dyn.matrix_options["_mem_eigen_adj"]
            # dyn.matrix_options["fact_mat_round_prec"]
        elif prop_type == 'FRECHET':
            dyn.matrix_options["method"] = "Frechet"
            dyn.matrix_options["sparse_exp"] = False
            dyn.matrix_options["sparse2dense"] = True
        elif prop_type in ['first_order', 1]:
            dyn.matrix_options["method"] = "first_order"
            dyn.matrix_options["sparse_exp"] = False
        elif prop_type in ['second_order', 2]:
            dyn.matrix_options["method"] = "second_order"
            dyn.matrix_options["sparse_exp"] = False
        elif prop_type in ['third_order', 3]:
            dyn.matrix_options["method"] = "third_order"
            dyn.matrix_options["sparse_exp"] = False
        else:
            raise errors.UsageError("No option for prop_type: " + prop_type)
        dyn.matrix_options.update(mat_params)

        dyn.optimization(self, mat_mode=mat_mode, tslot_mode=tslot_mode)

    if True: # optimizer section
        # Create the Optimiser instance
        optim_method_up = _upper_safe(optim_method)
        if optim_method is None or optim_method_up == '':
            pass
        elif optim_method_up == 'FMIN_BFGS':
            dyn.opt_method = 'BFGS'
        elif optim_method_up == 'LBFGSB' or optim_method_up == 'FMIN_L_BFGS_B':
            dyn.opt_method = 'L-BFGS-B'
        elif optim_method_up == 'FMIN':
            dyn.opt_method = 'Nelder-Mead'
        else:
            dyn.opt_method = optim_method

        dyn.termination_conditions["fid_err_targ"] = fid_err_targ
        dyn.termination_conditions["min_gradient_norm"] = min_grad
        dyn.termination_conditions["max_wall_time"] = max_wall_time
        dyn.termination_conditions["max_iterations"] = max_iter
        dyn.termination_conditions["max_fid_func_calls"] = max_iter*10
        dyn.solver_method_options.update(method_params)

    if alg == 'GRAPE':
        tf_param = {}
        transfer_function_type = _upper_safe(transfer_function_type)
        if transfer_function_type == 'DEF' or \
                transfer_function_type is None or \
                transfer_function_type == '':
            transfer_function_type = None
        elif transfer_function_type == 'FOURRIER':
            transfer_function_type = 'fourrier'
            if 'num_x' in transfer_function_params:
                tf_param['num_x'] = transfer_function_params['num_x']
        elif transfer_function_type == 'SPLINE':
            transfer_function_type = 'spline'
            if 'overSampleRate' in transfer_function_params:
                tf_param['overSampleRate'] = \
                        transfer_function_params['overSampleRate']
            if 'start' in transfer_function_params:
                tf_param['start'] = transfer_function_params['start']
            if 'end' in transfer_function_params:
                tf_param['end'] = transfer_function_params['end']
        elif transfer_function_type == 'GAUSSIAN':
            transfer_function_type = 'gaussian'
            if 'overSampleRate' in transfer_function_params:
                tf_param['overSampleRate'] = \
                        transfer_function_params['overSampleRate']
            if 'start' in transfer_function_params:
                tf_param['start'] = transfer_function_params['start']
            if 'end' in transfer_function_params:
                tf_param['end'] = transfer_function_params['end']
            if 'omega' in transfer_function_params:
                tf_param['omega'] = transfer_function_params['omega']
            if 'bound_type' in transfer_function_params:
                tf_param['bound_type'] = transfer_function_params['bound_type']
        elif isinstance(transfer_function_type, transfer_functions):
            pass
        dyn.set_transfer_function(transfer_function_type, **tf_param)

        if isinstance(init_pulse, str):
            # Create a pulse generator of the type specified
            pgen = pulsegen.create_pulse_gen(pulse_type=init_pulse,
                                             pulse_params=init_pulse_params)
            #pgen.scaling = pulse_scaling   # Now in init_pulse_params
            #pgen.offset = pulse_offset     # Now in init_pulse_params
            pgen.tau = np.diff(times)
            pgen.num_tslots = len(times)-1
            pgen.pulse_time = times[-1]
            pgen.lbound = amp_lbound
            pgen.ubound = amp_ubound
            dyn.set_initial_state(pgen)
        elif callable(init_pulse):
            dyn.set_initial_state(init_pulse)
        elif isinstance(init_pulse, (list, np.ndarray)):
            dyn.set_initial_state(np.array(init_pulse))

    elif alg == 'CRAB':
        if isinstance(ramping_pulse, str):
            # Create a pulse generator of the type specified
            ramping_pgen = pulsegen.create_pulse_gen(pulse_type=ramping_pulse,
                                             pulse_params=ramping_pulse_params)
            ramping_pgen.tau = np.diff(times)
            ramping_pgen.num_tslots = len(times)-1
            ramping_pgen.pulse_time = times[-1]
            ramping_pulse = ramping_pgen
        elif callable(ramping_pulse):
            pass # dyn.set_initial_state(ramping_pulse)
        elif isinstance(ramping_pulse, (list, np.ndarray)):
            ramping_pulse = (np.array(ramping_pulse))

        if isinstance(init_pulse, str):
            # Create a pulse generator of the type specified
            pgen = pulsegen.create_pulse_gen(pulse_type=init_pulse,
                                             pulse_params=init_pulse_params)
            pgen.tau = np.diff(times)
            pgen.num_tslots = len(times)-1
            pgen.pulse_time = times[-1]
            pgen.lbound = amp_lbound
            pgen.ubound = amp_ubound
            init_pulse = pgen
        elif callable(init_pulse):
            pass # dyn.set_initial_state(init_pulse)
        elif isinstance(init_pulse, (list, np.ndarray)):
            init_pulse = (np.array(init_pulse))

        crab_pulse_params = {}
        if "num_x" in crab_params:
            crab_pulse_params["num_x"] = crab_params["num_x"]
        if "opt_freq" in crab_params:
            crab_pulse_params["opt_freq"] = crab_params["opt_freq"]
        if "randomize_freqs" in crab_params:
            crab_pulse_params["randomize_freqs"] = crab_params["randomize_freqs"]
        if "guess_pulse_action" in crab_params:
            crab_pulse_params["guess_pulse_action"] = crab_params["guess_pulse_action"]
        dyn.set_crab_pulsegen(init_pulse=init_pulse,
                              ramping_pulse=ramping_pulse,
                              crab_pulse_params=crab_pulse_params)

    if False: #log_level <= logging.DEBUG:
        logger.debug(
            "Optimisation config summary...\n"
            "  object classes:\n"
            "    optimizer: " + optim.__class__.__name__ +
            "\n    dynamics: " + dyn.__class__.__name__ +
            "\n    tslotcomp: " + dyn.tslot_computer.__class__.__name__ +
            "\n    fidcomp: " + dyn.fid_computer.__class__.__name__ +
            "\n    propcomp: " + dyn.prop_computer.__class__.__name__ +
            "\n    pulsegen: " + pgen.__class__.__name__)


    return dyn
