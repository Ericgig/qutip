
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


Key Development Paradigms
==========================

When developing or extending solver features, three major architectural paradigms must be observed:

Single vs. Multi-Trajectory Evolution
-------------------------------------
Internally, deterministic individual trajectories (such as standard Lindblad master equations)
and stochastic, multi-trajectory simulations (such as Quantum Monte Carlo or stochastic master
equations) diverge significantly. Multi-trajectory frameworks must orchestrate independent parallel
worker runs, aggregate trajectory statistics, and handle probabilistic branch steps.

Dynamical State Feedback
------------------------
QuTiP supports state-dependent feedback loops. In feedback-driven systems, the system operators
or the Hamiltonian itself explicitly depend on the instantaneous expectation values or the state
of the system at time $t$. This requires a constant bidirectional data flow between the
integrator's current step state and the underlying time-dependent coefficient system (:class:`QobjEvo`).

Performance and Cythonization
-----------------------------
To minimize Python interpreter overhead during dense integration loops, Cython is deployed
strategically. Rather than wrapping the entire solver framework, Cython is localized to critical
numerical bottlenecks: specific ODE backends, high-frequency stochastic loops within
stochastic integrators, and ultra-fast data translation wrappers.


$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


Quantum Solver
##############

- The solver in QuTiP refer to any time-evolution of Quantum Systems.

- Different quantum equation (Schrodinger, Master Equation, Bloch RedField, etc.)
each correspond to a unique `XXSolver` class that is a child of `Solver`.

- The `xxxsolve` function (`sesolve`, `mesolve`) are helper function to build the class and get the result in one call.

- Multiple `xxxsolve` function can refer to the same physics (`sesolve`, `fsesolve`, `krylovsolve`).

- The `Solver` correspond to the physical equation, the what.
The how is worked in the `Integrator` class. Which integrator is used in controlled by the solver's options.

- The "method" options control which integrator to use.

- Integrator are generally ODE solver, but some can be specialized to some equation and mix the physic with the numerics.

- The user typically only interacts with the solver and solve function.
Integrator are almost never used directly, the SolverOptions class also.
The Result class is expected to be Read only for end users.

- Internally, single trajectory and multi trajectory solver are quite different.

- `Feedback` mean having the Hamiltonian or other operator of the system depend on the State.

- Cython is used moderatly: some Integrator and most Stochastic Integrator as well as some bottlenecks functions and wrapper.


.. toctree::
:maxdepth: 2
:caption: Sections

terminology
motivation
Solver
Integrator
Result
SolverOptions
Feedback
