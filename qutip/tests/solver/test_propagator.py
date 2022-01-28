import numpy as np
from qutip import *


def testPropHOB():
    a = destroy(5)
    H = a.dag()*a
    U = propagator(H, 1)
    U2 = (-1j * H).expm()
    assert (U - U2).norm('max') < 1e-4


def func(t):
    return np.cos(t)


def testPropHOTd():
    "Propagator: func td format"
    a = destroy(5)
    H = a.dag()*a
    Htd = [H, [H, func]]
    U = propagator(Htd, 1)
    ts = np.linspace(0, 1, 101)
    U2 = (-1j * H * np.trapz(1+func(ts), ts)).expm()
    assert (U - U2).norm('max') < 1e-4


def testPropHOSteady():
    "Propagator: steady state"
    a = destroy(5)
    H = a.dag()*a
    c_op_list = []
    kappa = 0.1
    n_th = 2
    rate = kappa * (1 + n_th)
    c_op_list.append(np.sqrt(rate) * a)
    rate = kappa * n_th
    c_op_list.append(np.sqrt(rate) * a.dag())
    U = propagator(H, 2*np.pi, c_op_list)
    rho_prop = propagator_steadystate(U)
    rho_ss = steadystate(H, c_op_list)
    assert (rho_prop - rho_ss).norm('max') < 1e-4


def testPropHDims():
    "Propagator: preserve H dims (unitary_mode='single', parallel=False)"
    H = tensor([qeye(2), qeye(2)])
    U = propagator(H, 1)
    assert U.dims == H.dims
