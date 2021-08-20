from .herm_matmul import make_herm_rhs
from types import MethodType
from ..integrator import integrator_collection
from qutip.core import data as _data
from qutip.core.data import to
from qutip import unstack_columns

def set_state(self, t, state0):
    data_dm = unstack_columns(state0)
    if not _data.isherm(data_dm):
        raise ValueError("`herm` rhs options only available "
                         "for hermitian states")
    self.__class__.set_state(self, t, _data.to(_data.Dense, state0))

def prepare_herm_integrator(integrator_cls, system, options):
    """
Overwrite the system matmul function for one that make uses of the hermiticity
of the state for faster solving. To be used in mesolve."""
    system = make_herm_rhs(system)
    integrator = integrator_cls(system, options)
    # Add a check on to the set_state
    integrator.set_state = MethodType(set_state, integrator)
    return integrator

prepare_herm_integrator.long_description = prepare_herm_integrator.__doc__
prepare_herm_integrator.description = (
    "Matrix produce using density matrices's hermiticity for speedup")

integrator_collection.add_rhs(prepare_herm_integrator, "herm",
                              ['mesolve'], True)
