#cython: language_level=3

from qutip.core.data cimport Data
from qutip.core.cy.qobjevo cimport QobjEvo
from qutip.core.data.constant import zeros_like

# Migrating integrator from supporting QobjEvo (matmul) only, to functions
# with signature:
#
#     (double t, Data state) -> Data dstate or
#     (double t, Data state, Data dstate)
#
# We still want fast cython access to `QobjEvo.matmul_data` and support
# both direct calls from python and inplace calls from cython.


__all__ = ["RHS"]


cdef class RHS:
    """
    Container for the derivative function in qutip's integrators.

    Parameters
    ----------
    derivative: Callable[[float, _data.Data, ...], _data.Data] | QobjEvo
        Function to integrate.
        Can be either a QobjEvo where QobjEvo @ state is the derivative or
        a callable. This function can be both inplace or not.

    inplace: bool
        If ``derivative`` is a callable, whether that function take the output
        inplace or not.

    """
    def __init__(
        self,
        derivative:Callable[[float, _data.Data], _data.Data] | QobjEvo,
        inplace; bool=False
    ):
        self.derivative = derivative
        self.inplace = inplace
        self.qevo_derr = False
        if isinstance(derivative, QobjEvo):
            self.qevo = derivative
            self.qevo_derr = True

    def __call__(self, t: float, state: Data):
        """
        Apply the derivative function
        """
        return self.apply(t, state)

    cdef Data apply(self, double t, Data state, Data out=None):
        """
        Cython interface for the derivative function.
        """
        if qevo_derr:
            return self.qevo.matmul_data(t, state, out=out)

        if self.inplace:
            if out is None:
                out = zero_like(state)
            self.derivative(t, state, out)
        elif out is None:
            out = _data.iadd(self.derivative(t, state), out)
        else:
            out = self.derivative(t, state)
        return out
