Terminology
===========

This is a short terminology glossary for the solver and related classes.
See the rest of the guide for more in depth information.

Quantum system
  Set of operator and physical parameters that are used to build the equation
  solved by the solver. Common components are the Hamiltonian, collapse operators,
  bath coupling operators and environment information.

method
  Name of the integration method used for the evolution. Can refer to ODE or
  SODE integration algorithm, depending on the solver.

ODE
  Ordinary Differential Equation:

    $$ dX/dt = f(t, X) $$

SDE
  Stochastic Differential Equation:

    $$ dX = f(t, X) dt + \Sum_i g_i(t, X) dW_i $$

  with $dW_i$ independent gaussian noise.

Feedback
  General term for non-linear equation. Having the quantum system depend on the
  state. For example, having the Hamiltonian depend on the density matrix:
  $H(t, \rho)$. Different :class:`Solver` can support different types of Feedback.

Trajectory
  Single run of a non-deterministic evolution. For example, an evolution of
  :class:`MCSolver` with it's own unique jump timing.
