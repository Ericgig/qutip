

import numpy as np
import qutip as qt


import importlib
import importlib.util

moduleName = "/home/eric/algo/qutip/qutip/qutip/control_2/dynamic.py"
spec = importlib.util.spec_from_file_location("dynamic", moduleName)
dynamic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dynamic)

"""moduleName = "/home/eric/algo/qutip/qutip/qutip/control_2/optimize.py"
spec = importlib.util.spec_from_file_location("optimize", moduleName)
optimize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(optimize)"""

def grape_unitary(H, ctrls, target=None, times=None,
                  u_start=None, u_limits=None, phase_sensitive=False,
                  run=True):
    """
    Calculate control pulses for the Hamiltonian operators in H_ops so that the
    unitary U is realized.

    Experimental: Work in progress.

    Parameters
    ----------

    H0 : Qobj
        Static Hamiltonian (that cannot be tuned by the control fields).

    H_ops: list of Qobj
        A list of operators that can be tuned in the Hamiltonian via the
        control fields.

    target : Qobj
        Target unitary evolution operator.

    times : array / list
        Array of time coordinates for control pulse evalutation.

    u_start : array, pulsegen
        Optional array with initial control pulse values or a pulsegen object.

    u_limits : list, [min, max]
        Minimum and maximum values acceptable for control pulses values.

    run: bool
        Option to run the grape optimisation.
        If False, will return an dynamic object.

    Returns
    -------


    """
    system = dynamic.dynamics()
    H_phase = H*(-1j)
    ctrls_phase = [ctrl*-1j for ctrl in ctrls]
    target_phase = target#*(-1j)
    initial = np.eye(target_phase.shape[0])

    system.set_physic(H=H_phase, ctrls=ctrls_phase,
                      initial=initial, target=target_phase)
    system.result_phase = 1#j
    if phase_sensitive:
        mode = "TrDiff"
    else:
        mode = "TrAbs"
    system.set_cost(mode=mode)

    if times is not None:
        system.set_times(times=times)

    if u_start is not None:
        system.set_initial_state(u_start)


    if run:
        result = system.run()
        #result.evo_full_final *= 1j
        return result
    else:
        return system


def grape_unitary_state(H, ctrls, target, initial, times=None, u_start=None,
                        u_limits=None, phase_sensitive=False, run=True):
    """
    Calculate control pulses for the Hamiltonian operators in H_ops so that the
    unitary U is realized.

    Experimental: Work in progress.

    Parameters
    ----------

    H0 : Qobj
        Static Hamiltonian (that cannot be tuned by the control fields).

    H_ops: list of Qobj
        A list of operators that can be tuned in the Hamiltonian via the
        control fields.

    target : Qobj
        Target unitary evolution operator.

    times : array / list
        Array of time coordinates for control pulse evalutation.

    u_start : array, pulsegen
        Optional array with initial control pulse values or a pulsegen object.

    u_limits : list, [min, max]
        Minimum and maximum values acceptable for control pulses values.

    run: bool
        Option to run the grape optimisation.
        If False, will return an dynamic object.

    Returns
    -------

    """
    system = dynamic.dynamics()
    H_phase = H*(-1j)
    ctrls_phase = [ctrl*-1j for ctrl in ctrls]
    system.set_physic(H=H_phase, ctrls=ctrls_phase,
                      initial=initial, target=target)
    if phase_sensitive:
        mode = "SU"
    else:
        mode = "PSU"
    system.set_cost(mode=mode)
    if times is not None:
        system.set_times(times=times)
    if u_start is not None:
        system.set_initial_state(u_start)
    if run:
        result = system.run()
        return result
    else:
        return system


def opengrape_state(H, c_ops, target, initial,
                    ctrls=[], c_ops_ctrls=[],
                    times=None, u_start=None, u_limits=None, filter=None,
                    phase_sensitive=False, run=True):
    system = dynamic.dynamics()
    L = qt.liouvillian(H, c_ops)
    L_ctrls = []
    L_ctrls += [qt.liouvillian(ctrl, []) for ctrl in ctrls]
    L_ctrls += [qt.liouvillian(None, [ctrl]) for ctrl in c_ops_ctrls]
    rho_0 = qt.mat2vec((initial*initial.dag()).full())
    rho_t = qt.mat2vec((target*target.dag()).full())
    system.set_physic(H=L, ctrls=L_ctrls, initial=rho_0, target=rho_t)
    system.set_cost(mode="Diff")
    if times is not None:
        system.set_times(times=times)
    if u_start is not None:
        system.set_initial_state(u_start)
    if run:
        result = system.run()
        return result
    else:
        return system


def opengrape(H, c_ops, target,
              ctrls=[], c_ops_ctrls=[],
              times=None, u_start=None, u_limits=None, filter=None,
              phase_sensitive=False, run=True):
    system = dynamic.dynamics()
    L = qt.liouvillian(H, c_ops)
    L_ctrls = []
    L_ctrls += [qt.liouvillian(ctrl, []) for ctrl in ctrls]
    L_ctrls += [qt.liouvillian(None, [ctrl]) for ctrl in c_ops_ctrls]
    rho_0 = qt.mat2vec((initial*initial.dag()).full())
    rho_t = qt.mat2vec((target*target.dag()).full())
    system.set_physic(H=L, ctrls=L_ctrls, target=rho_t)
    system.set_cost(mode="Diff")
    if times is not None:
        system.set_times(times=times)
    if u_start is not None:
        system.set_initial_state(u_start)
    if run:
        result = system.run()
        return result
    else:
        return system
