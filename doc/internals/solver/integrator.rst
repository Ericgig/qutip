.. _integrator_internal::

Integrator design
#################

Motivation
==========
The problem of Differential equation is complex and has multiple numerical algorithm
to solve it, the best depending on the specific of the problem. It also has a lot of packages that provide
tools to solve them. In QuTiP, having access to efficient ODE method needed as it's a key tool in solver.
Scipy provide a good range of integration method, however even inside scipy, the interface change between algorithm (some don't support complex, solve_ivp vs ODE vs ode_int).
Moreover, they don't support GPUs, and there are algorithm that are adapted to quantum evolution that are too specific for scipy to have.

This is why in QuTiP we have the Integrator class. It wraps over scipy and other ODE solver solution to provide a consistent interface making them interchangeble.


Design
======

The :class:"Integrator" class is the parent of all integrators.

For the most common solver, all that is needed is, from and initial time and state, get the state at increasing time in order.
For that, the class has the ``set_state``, ``get_state``, ``integrate`` and ``run`` methods

set_state: input the state and time.
get_state: output the state and time.
integrate: integrate to the time `t`. Use the state set by set_state.
run: yield the results at time in the ``tlist``.

The the return of get_state, integrate and run is always the tuple ``(t, state)``.

The second use for integrator in to find collapses in Monte Carlo evolution.
For this, the method ``mcstep`` is used. This method accepts time in non-increasing order.
In QuTiP, it's mcsolve that search for the collapse, and not the ODE event feature.
This is because the scipy ``ode`` interface was used as the default (only) integration method in previous version.
Algorithms from these interface are fast (optimized fortran code), but the interface lack advanced feature such as events.
(Add event to integrator???)


Lastly, the most ODE algorithm support user controlled parameters which must be controlable from our interface.
All such options are included in the ``options`` attribute. Changing value in that dictionary does not need to take into effect immediately,
the `reset` method exist to restart the integrator while keeping the state to continue with the new options.

The class has also the following attributes:

- integrator_options: dictionary of the options supported by the method with their default values.
  The instance will have an independent `options` dict with the live values.
- RHS_format: Format of the RHS of the ODE.
- name: Long format name as shown in the result.
- method: Short name, key for the available integrator directory. (Needed??)
