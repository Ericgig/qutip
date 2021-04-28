# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
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
""" Define `Integrator`: ODE solver wrapper to use in qutip's Solver """


__all__ = ['integrator_collection', 'IntegratorException']


import numpy as np
from itertools import product
from qutip.core import data as _data
from .options import SolverOptions, SolverOdeOptions
from qutip import QobjEvo, qeye, basis
from functools import partial


class IntegratorException(Exception):
    pass


class _IntegratorCollection:
    """
    Collection of ODE :obj:`Integrator` available to Qutip's solvers.

    :obj:`Integrator` are composed of 2 parts: `method` and `rhs`:
    `method` are ODE integration method such as Runge-Kutta or Adams–Moulton.
    `rhs` are options to control the :obj:`QobjEvo`'s matmul function.

    Parameters
    ----------
    known_solvers : list of str
        list of solver using this ensemble of integrator

    options_class : :func:optionsclass decorated class
        Option object to add integrator's option to the accepted keys.
    """
    def __init__(self, known_solvers, options_class):
        self.known_solvers = known_solvers
        self.options_class = options_class
        # map from method key to integrator
        self.method2integrator = {}
        # map from rhs key to rhs function
        self.rhs2system = {}
        # methods's keys which support alternative rhs
        self.base_methods = []
        # Information about methods
        self.method_data = {}
        # Information about rhs
        self.rhs_data = {}


    def add_method(self, integrator, keys, solver,
                   use_QobjEvo_matmul, time_dependent):
        """
        Add a new integrator to the set available to solvers.

        Parameters
        ----------

        integrator : class derived from :obj:`Integrator`
            New integrator to add.

        keys : list of str
            List of keys supported by the integrator.
            When `options["methods"] in keys`, this integrator will be used.

        solver : list of str
            List of the [...]solve function that are supported by the
            integrator.

        use_QobjEvo_matmul : bool
            Whether the Integrator use `QobjEvo.matmul` as the function of the
            ODE. When `False`, rhs cannot be used.

        time_dependent : bool
            Whether integrator support time-dependent system.
        """
        if not isinstance(keys, list):
            keys = [keys]
        for key in keys:
            if key in self.method2integrator:
                raise ValueError("method '{}' already used".format(key))

        integrator_data = self._complete_data(integrator, solver,
                                              use_QobjEvo_matmul,
                                              time_dependent)
        for key in keys:
            self.method2integrator[key] = integrator
            self.method_data[key] = integrator_data
        if use_QobjEvo_matmul:
            self.base_methods += keys

    def add_rhs(self, integrator, keys, solver, time_dependent):
        """
        Add a new rhs to the set available to solvers.

        Parameters
        ----------

        integrator : callable
            Function with the signature::

                rhs(
                    integrator: class,
                    system: QobjEvo,
                    options: SolverOptions
                ) -> Integrator

            that create the :obj:`Integrator` instance. The integrator can be
            any integrator registered that has `"use_QobjEvo_matmul" == True`.
            `system` is the :obj:`QobjEvo` driving the ODE: 'L', '-i*H', etc.
            `options` is the :obj:`SolverOptions` of the solver.

        keys : list of str
            List of keys supported by the integrator.
            When `options["methods"] in keys`, this integrator will be used.

        solver : list of str
            List of the [...]solve function that are supported by the
            integrator.

        time_dependent : bool
            True if the integrator can solve time dependent systems.
        """
        if not isinstance(keys, list):
            keys = [keys]
        for key in keys:
            if key in self.rhs2system:
                raise ValueError("rhs keyword '{}' already used")
        integrator_data = self._complete_data(integrator, solver,
                                              True,
                                              time_dependent)
        for key in keys:
            self.rhs2system[key] = integrator
            self.rhs_data[key] = integrator_data

    def _complete_data(self, integrator, solver, base, td):
        """
        Create the information container for the integrator.
        """
        integrator_data = {
            "integrator": integrator,
            "description": "",
            "long_description": "",
            # options used by the integrator, to add to the accepted option
            # by the SolverOptions object.
            "options": [],
            # list of supported solver, sesolve, mesolve, etc.
            "solver": [],
            # The `method` use QobjEvo's matmul.
            # If False, refuse `rhs` option.
            "use_QobjEvo_matmul": base,
            # Support of time-dependent system
            "time_dependent": td,
        }
        for sol in solver:
            if sol not in self.known_solvers:
                raise ValueError(f"Unknown solver '{sol}', known solver are"
                                 + str(self.known_solvers))
            integrator_data["solver"].append(sol)

        if hasattr(integrator, 'description'):
            integrator_data['description'] = integrator.description

        if hasattr(integrator, 'long_description'):
            integrator_data['long_description'] = integrator.long_description
        elif hasattr(integrator, '__doc__'):
            integrator_data['long_description'] = integrator.__doc__

        if hasattr(integrator, 'used_options'):
            integrator_data['options'] = integrator.used_options
            for opt in integrator.used_options:
                self.options_class.extra_options.add(opt)

        return integrator_data

    def __getitem__(self, key):
        """
        Obtain the integrator from the (method, rhs) key pair.
        """
        method, rhs = key
        try:
            integrator = self.method2integrator[method]
        except KeyError:
            raise KeyError(f"ode method {method} not found")
        if rhs == "":
            build_func = prepare_integrator
        elif self.method_data[method]["use_QobjEvo_matmul"]:
            try:
                build_func = self.rhs2system[rhs]
            except KeyError:
                raise KeyError(f"ode rhs {rhs} not found")
        else:
            raise KeyError(f"ode method {method} does not support rhs")
        return partial(build_func, integrator)

    def list_keys(self, keytype="methods", solver="",
                  use_QobjEvo_matmul=None, time_dependent=None):
        """
        List integrators available corresponding to the conditions given.
        """
        # used in test
        if keytype == "methods":
            names = [method for method in self.method2integrator
                     if self.check_condition(method, "", solver,
                                             use_QobjEvo_matmul,
                                             time_dependent)]
        elif keytype == "rhs":
            names = [rhs for rhs in self.rhs2system
                     if self.check_condition("", rhs, solver,
                                             use_QobjEvo_matmul,
                                             time_dependent)]
        elif keytype == "pairs":
            names = [(method, "") for method in self.method2integrator
                     if self.check_condition(method, "", solver,
                                             use_QobjEvo_matmul,
                                             time_dependent)]
            names += [(method, rhs)
                for method, rhs in product(self.base_methods, self.rhs2system)
                if rhs and self.check_condition(method, rhs, solver,
                    use_QobjEvo_matmul, time_dependent)
            ]
        else:
            raise ValueError("keytype must be one of "
                             "'rhs', 'methods' or 'pairs'")
        return names

    def check_condition(self, method, rhs, solver="",
                        use_QobjEvo_matmul=None, time_dependent=None):
        if method in self.method_data:
            data = self.method_data[method]
            if solver and solver not in data['solver']:
                return False
            if (
                use_QobjEvo_matmul is not None and
                use_QobjEvo_matmul != data['use_QobjEvo_matmul']
            ):
                return False
            if (
                time_dependent is not None and
                time_dependent != data['time_dependent']
            ):
                return False

        if rhs in self.rhs_data:
            data = self.rhs_data[rhs]
            if solver and solver not in data['solver']:
                return False
            if (
                use_QobjEvo_matmul is not None and
                use_QobjEvo_matmul != data['use_QobjEvo_matmul']
            ):
                return False
            if (
                time_dependent is not None and
                time_dependent != data['time_dependent']
            ):
                return False

        return True

    def full_description(self, method=None, rhs=None):
        """ Return a description of the given ODE integrator scheme.
        """
        if method and rhs:
            raise ValueError("Need a method or a rhs key")
        if method:
            data = self.method_data[method]
        elif rhs:
            data = self.rhs_data[method]
        else:
            raise ValueError("Unknown method or rhs key")
        out = f"""{method or rhs} :
{data['long_description'] or data['description']}
Used options: {data['options']}
Supported solver: {data['solver']}
"""
        if not data['time_dependent']:
            out += "*Does not support time-dependent system."
        return out

    def short_description(self, method=None, rhs=None):
        """ Return a one line description of the given ODE integrator scheme.
        """
        if method and rhs:
            raise ValueError("Need a method or a rhs key")
        if method:
            data = self.method_data[method]
        elif rhs:
            data = self.method_data[method]
        else:
            raise ValueError("Unknown method or rhs key")
        return data['description'] or data['integrator'].__name__


integrator_collection = _IntegratorCollection(
    ['sesolve', 'mesolve', 'mcsolve'],
    SolverOdeOptions
)
# mcsolve requires the integrator to allow intergrating in non sequential order


def prepare_integrator(integrator, system, options):
    """Default rhs function"""
    return integrator(system, options)


integrator_collection.add_rhs(prepare_integrator, "",
                              ['sesolve', 'mesolve', 'mcsolve'], True)
