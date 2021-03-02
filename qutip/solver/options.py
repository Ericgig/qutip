__all__ = ['SolverOptions',
           'SolverResultsOptions', 'SolverOdeOptions',
           'McOptions']

from ..optionsclass import optionsclass
import multiprocessing

@optionsclass("solver")
class SolverOptions:
    """
    Class of options for evolution solvers such as :func:`qutip.mesolve` and
    :func:`qutip.mcsolve`. Options can be specified either as arguments to the
    constructor::

        opts = SolverOptions(order=10, ...)

    or by changing the class attributes after creation::

        opts = SolverOptions()
        opts.order = 10

    Returns options class to be used as options in evolution solvers.

    The default can be changed by::

        qutip.settings.solver['order'] = 10

    Options
    -------
    progress_bar : str {'text', 'enhanced', 'tqdm', ''}
        How to present the solver progress.
        'tqdm' use the python module of the same name and raise an error if
        not installed.
        True will result in 'text'.
        Empty string or False will disable the bar.

    progress_kwargs : dict
        kwargs to pass to the progress_bar. Qutip's bars use `chunk_size`.
    """
    options = {
        # (turned off for batch unitary propagator mode)
        "progress_bar": "text",
        # Normalize output of solvers
        # (turned off for batch unitary propagator mode)
        "progress_kwargs": {"chunk_size":10},
    }


@optionsclass("ode", SolverOptions)
class SolverOdeOptions:
    """
    Class of options for evolution solvers such as :func:`qutip.mesolve` and
    :func:`qutip.mcsolve`. Options can be specified either as arguments to the
    constructor::

        opts = SolverOptions(order=10, ...)

    or by changing the class attributes after creation::

        opts = SolverOptions()
        opts.order = 10

    Returns options class to be used as options in evolution solvers.

    The default can be changed by::

        qutip.settings.solver['order'] = 10

    Options
    -------
    method : str {'adams', 'bdf', 'dop853', 'lsoda', 'vern7', 'vern9', 'diag'}
        Integration method.

    atol : float {1e-8}
        Absolute tolerance.

    rtol : float {1e-6}
        Relative tolerance.

    order : int {12}
        Order of integrator (<=12 'adams', <=5 'bdf')

    nsteps : int {2500}
        Max. number of internal steps/call.

    first_step : float {0}
        Size of initial step (0 = automatic).

    min_step : float {0}
        Minimum step size (0 = automatic).

    max_step : float {0}
        Maximum step size (0 = automatic)

    tidy: bool {True}
        tidyup Hamiltonian before calculation

    Operator_data_type: :class:`qutip.data.Data`, str {"input"}
        Data type of the system, some method can overwrite it.

    State_data_type: :class:`qutip.data.Data`, str {qutip.data.Dense}
        Data type of the state, most solver can only work with `Dense`.

    feedback_normalize: bool


    """
    options = {
        # Integration method (default = 'adams', for stiff 'bdf')
        "method": 'adams',

        "rhs": '',

        # Absolute tolerance (default = 1e-8)
        "atol": 1e-8,
        # Relative tolerance (default = 1e-6)
        "rtol": 1e-6,
        # Maximum order used by integrator (<=12 for 'adams', <=5 for 'bdf')
        "order": 12,
        # Max. number of internal steps/call
        "nsteps": 2500,
        # Size of initial step (0 = determined by solver)
        "first_step": 0,
        # Max step size (0 = determined by solver)
        "max_step": 0,
        # Minimal step size (0 = determined by solver)
        "min_step": 0,
        # tidyup Hamiltonian before calculation (default = True)
        "tidy": True,

        "Operator_data_type": "input",

        "State_data_type": "dense",
        # Normalize the states received in feedback_args
        "feedback_normalize": True,
    }
    extra_options = set()


@optionsclass("results", SolverOptions)
class SolverResultsOptions:
    """
    Class of options for evolution solvers such as :func:`qutip.mesolve` and
    :func:`qutip.mcsolve`. Options can be specified either as arguments to the
    constructor::

        opts = SolverOptions(order=10, ...)

    or by changing the class attributes after creation::

        opts = SolverOptions()
        opts.order = 10

    Returns options class to be used as options in evolution solvers.

    The default can be changed by::

        qutip.settings.solver['order'] = 10

    Options
    -------
    store_final_state : bool {False, True}
        Whether or not to store the final state of the evolution in the
        result class.
    store_states : bool {False, True}
        Whether or not to store the state vectors or density matrices in the
        result class, even if expectation values operators are given. If no
        expectation are provided, then states are stored by default and this
        option has no effect.
    normalize_output : str {"", "ket", "all"}
        normalize output state to hide ODE numerical errors.
        On "ket", only 'ket' output are normalized.
    """
    options = {
        # store final state?
        "store_final_state": False,
        # store states even if expectation operators are given?
        "store_states": False,
        # Normalize output of solvers
        # (turned off for batch unitary propagator mode)
        "normalize_output": "ket",
    }


@optionsclass("mcsolve", SolverOptions)
class McOptions:
    """
    Class of options for evolution solvers such as :func:`qutip.mesolve` and
    :func:`qutip.mcsolve`. Options can be specified either as arguments to the
    constructor::

        opts = SolverOptions(norm_tol=1e-3, ...)

    or by changing the class attributes after creation::

        opts = SolverOptions()
        opts['norm_tol'] = 1e-3

    Returns options class to be used as options in evolution solvers.

    The default can be changed by::

        qutip.settings.options.mcsolve['norm_tol'] = 1e-3

    Options
    -------

    norm_tol : float {1e-4}
        Tolerance used when finding wavefunction norm in mcsolve.

    norm_t_tol : float {1e-6}
        Tolerance used when finding wavefunction time in mcsolve.

    norm_steps : int {5}
        Max. number of steps used to find wavefunction norm to within norm_tol
        in mcsolve.

    keep_runs_results: bool
        Keep all trajectories results or save only the average.

    map : str  {'parallel', 'serial', 'loky'}
        How to run the trajectories.
        'parallel' use python's multiprocessing.
        'loky' use the pyhon module of the same name (not installed with qutip).

    map_options: dict
        keys:
            'num_cpus': number of cpus to use.
            'timeout': maximum time for all trajectories. (sec)
            'job_timeout': maximum time per trajectory. (sec)
        Only finished trajectories will be returned when timeout is reached.

    mc_corr_eps : float {1e-10}
        Arbitrarily small value for eliminating any divide-by-zero errors in
        correlation calculations when using mcsolve.
    """
    options = {
        # Tolerance for wavefunction norm (mcsolve only)
        "norm_tol": 1e-4,
        # Tolerance for collapse time precision (mcsolve only)
        "norm_t_tol": 1e-6,
        # Max. number of steps taken to find wavefunction norm to within
        # norm_tol (mcsolve only)
        "norm_steps": 5,

        "map": "parallel_map",

        "keep_runs_results": False,

        "mc_corr_eps": 1e-10,

        "map_options": {
            'num_cpus': multiprocessing.cpu_count(),
            'timeout':1e8,
            'job_timeout':1e8
        },
    }
