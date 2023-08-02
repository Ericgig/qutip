"""Permutational Invariant Quantum Solver (PIQS)

This module calculates the Liouvillian for the dynamics of ensembles of
identical two-level systems (TLS) in the presence of local and collective
processes by exploiting permutational symmetry and using the Dicke basis.
It also allows to characterize nonlinear functions of the density matrix.
"""

# Authors: Nathan Shammah, Shahnawaz Ahmed
# Contact: nathan.shammah@gmail.com, shahnawaz.ahmed95@gmail.com

from math import factorial
from decimal import Decimal

import numpy as np
from scipy.integrate import odeint
from scipy.linalg import eigvalsh
from scipy.special import entr
from scipy.sparse import dok_matrix, block_diag, lil_matrix
from .. import (
    Qobj, spre, spost, tensor, identity, ket2dm, sigmax, sigmay, sigmaz,
    sigmap, sigmam,
)
from ..entropy import entropy_vn
from ._piqs import Dicke as _Dicke
from ._piqs import (
    jmm1_dictionary,
    _num_dicke_states,
    _num_dicke_ladders,
    get_blocks,
    j_min,
    j_vals,
)

__all__ = [
    "num_dicke_states",
    "num_dicke_ladders",
    "num_tls",
    "dicke_blocks",
    "dicke_blocks_full",
    "dicke_function_trace",
    "purity_dicke",
    "entropy_vn_dicke",
    "Dicke",
    "tau_column",
    "Pim",
]


def _ensure_int(x):
    """
    Ensure that a floating-point value `x` is exactly an integer, and return it
    as an int.
    """
    out = int(x)
    if out != x:
        raise ValueError(f"{x} is not an integral value")
    return out


# Functions necessary to generate the Lindbladian/Liouvillian
def num_dicke_states(N):
    """Calculate the number of Dicke states.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    Returns
    -------
    nds: int
        The number of Dicke states.
    """
    return _num_dicke_states(N)


def num_dicke_ladders(N):
    """Calculate the total number of ladders in the Dicke space.

    For a collection of N two-level systems it counts how many different
    "j" exist or the number of blocks in the block-diagonal matrix.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    Returns
    -------
    Nj: int
        The number of Dicke ladders.
    """
    return _num_dicke_ladders(N)


def num_tls(nds):
    """Calculate the number of two-level systems.

    Parameters
    ----------
    nds: int
         The number of Dicke states.

    Returns
    -------
    N: int
        The number of two-level systems.
    """
    if np.sqrt(nds).is_integer():
        # N is even
        N = 2 * (np.sqrt(nds) - 1)
    else:
        # N is odd
        N = 2 * (np.sqrt(nds + 1 / 4) - 1)
    return int(N)


def _isdiagonal(mat):
    """
    Check if the input matrix is diagonal.

    Parameters
    ==========
    mat: ndarray/Qobj
        A 2D numpy array

    Returns
    =======
    diag: bool
        True/False depending on whether the input matrix is diagonal.
    """
    if isinstance(mat, Qobj):
        mat = mat.full()

    return np.all(mat == np.diag(np.diagonal(mat)))


# nonlinear functions of the density matrix
def dicke_blocks(rho):
    """Create the list of blocks for block-diagonal density matrix in the Dicke basis.

    Parameters
    ----------
    rho : :class:`qutip.Qobj`
        A 2D block-diagonal matrix of ones with dimension (nds,nds),
        where nds is the number of Dicke states for N two-level
        systems.

    Returns
    -------
    square_blocks: list of np.array
        Give back the blocks list.

    """
    shape_dimension = rho.shape[0]
    N = num_tls(shape_dimension)
    ladders = num_dicke_ladders(N)
    # create a list with the sizes of the blocks, in order
    blocks_dimensions = int(N / 2 + 1 - 0.5 * (N % 2))
    blocks_list = [
        (2 * (i + 1 * (N % 2)) + 1 * ((N + 1) % 2))
        for i in range(blocks_dimensions)
    ]
    blocks_list = np.flip(blocks_list, 0)
    # create a list with each block matrix as element
    square_blocks = []
    block_position = 0
    for block_size in blocks_list:
        start_m = block_position
        end_m = block_position + block_size
        square_block = rho[start_m:end_m, start_m:end_m]
        block_position = block_position + block_size
        square_blocks.append(square_block)

    return square_blocks


def dicke_blocks_full(rho):
    """Give the full (2^N-dimensional) list of blocks for a Dicke-basis matrix.

    Parameters
    ----------
    rho : :class:`qutip.Qobj`
        A 2D block-diagonal matrix of ones with dimension (nds,nds),
        where nds is the number of Dicke states for N two-level
        systems.

    Returns
    -------
    full_blocks : list
        The list of blocks expanded in the 2^N space for N qubits.

    """
    shape_dimension = rho.shape[0]
    N = num_tls(shape_dimension)
    ladders = num_dicke_ladders(N)
    # create a list with the sizes of the blocks, in order
    blocks_dimensions = int(N / 2 + 1 - 0.5 * (N % 2))
    blocks_list = [
        (2 * (i + 1 * (N % 2)) + 1 * ((N + 1) % 2))
        for i in range(blocks_dimensions)
    ]
    blocks_list = np.flip(blocks_list, 0)
    # create a list with each block matrix as element
    full_blocks = []
    k = 0
    block_position = 0
    for block_size in blocks_list:
        start_m = block_position
        end_m = block_position + block_size
        square_block = rho[start_m:end_m, start_m:end_m]
        block_position = block_position + block_size
        j = N / 2 - k
        djn = state_degeneracy(N, j)
        for block_counter in range(0, djn):
            full_blocks.append(square_block / djn)  # preserve trace
        k = k + 1
    return full_blocks


def dicke_function_trace(f, rho):
    """Calculate the trace of a function on a Dicke density matrix.
    Parameters
    ----------
    f : function
        A Taylor-expandable function of `rho`.

    rho : :class:`qutip.Qobj`
        A density matrix in the Dicke basis.

    Returns
    -------
    res : float
        Trace of a nonlinear function on `rho`.
    """
    N = num_tls(rho.shape[0])
    blocks = dicke_blocks(rho)
    block_f = []
    degen_blocks = np.flip(j_vals(N))
    state_degeneracies = []
    for j in degen_blocks:
        dj = state_degeneracy(N, j)
        state_degeneracies.append(dj)
    eigenvals_degeneracy = []
    deg = []
    for i, block in enumerate(blocks):
        dj = state_degeneracies[i]
        normalized_block = block / dj
        eigenvals_block = eigvalsh(normalized_block)
        for val in eigenvals_block:
            eigenvals_degeneracy.append(val)
            deg.append(dj)

    eigenvalue = np.array(eigenvals_degeneracy)
    function_array = f(eigenvalue) * deg
    return sum(function_array)


def entropy_vn_dicke(rho):
    """Von Neumann Entropy of a Dicke-basis density matrix.

    Parameters
    ----------
    rho : :class:`qutip.Qobj`
        A 2D block-diagonal matrix of ones with dimension (nds,nds),
        where nds is the number of Dicke states for N two-level
        systems.

    Returns
    -------
    entropy_dm: float
        Entropy. Use degeneracy to multiply each block.

    """
    return dicke_function_trace(entr, rho)


def purity_dicke(rho):
    """Calculate purity of a density matrix in the Dicke basis.
    It accounts for the degenerate blocks in the density matrix.

    Parameters
    ----------
    rho : :class:`qutip.Qobj`
        Density matrix in the Dicke basis of qutip.piqs.jspin(N), for N spins.

    Returns
    -------
    purity : float
        The purity of the quantum state.
        It's 1 for pure states, 0<=purity<1 for mixed states.
    """
    f = lambda x: x * x
    return dicke_function_trace(f, rho)


class Result:
    pass


class Dicke(object):
    """The Dicke class which builds the Lindbladian and Liouvillian matrix.

    Examples
    --------
    >>> from qutip.piqs import Dicke, jspin
    >>> N = 2
    >>> jx, jy, jz = jspin(N)
    >>> jp = jspin(N, "+")
    >>> jm = jspin(N, "-")
    >>> ensemble = Dicke(N, emission=1.)
    >>> L = ensemble.liouvillian()

    Parameters
    ----------
    N: int
        The number of two-level systems.

    hamiltonian : :class:`qutip.Qobj`
        A Hamiltonian in the Dicke basis.

        The matrix dimensions are (nds, nds),
        with nds being the number of Dicke states.
        The Hamiltonian can be built with the operators
        given by the `jspin` functions.

    emission: float
        Incoherent emission coefficient (also nonradiative emission).
        default: 0.0

    dephasing: float
        Local dephasing coefficient.
        default: 0.0

    pumping: float
        Incoherent pumping coefficient.
        default: 0.0

    collective_emission: float
        Collective (superradiant) emmission coefficient.
        default: 0.0

    collective_pumping: float
        Collective pumping coefficient.
        default: 0.0

    collective_dephasing: float
        Collective dephasing coefficient.
        default: 0.0

    Attributes
    ----------
    N: int
        The number of two-level systems.

    hamiltonian : :class:`qutip.Qobj`
        A Hamiltonian in the Dicke basis.

        The matrix dimensions are (nds, nds),
        with nds being the number of Dicke states.
        The Hamiltonian can be built with the operators given
        by the `jspin` function in the "dicke" basis.

    emission: float
        Incoherent emission coefficient (also nonradiative emission).
        default: 0.0

    dephasing: float
        Local dephasing coefficient.
        default: 0.0

    pumping: float
        Incoherent pumping coefficient.
        default: 0.0

    collective_emission: float
        Collective (superradiant) emmission coefficient.
        default: 0.0

    collective_dephasing: float
        Collective dephasing coefficient.
        default: 0.0

    collective_pumping: float
        Collective pumping coefficient.
        default: 0.0

    nds: int
        The number of Dicke states.

    dshape: tuple
        The shape of the Hilbert space in the Dicke or uncoupled basis.
        default: (nds, nds).
    """

    def __init__(
        self,
        N,
        hamiltonian=None,
        emission=0.0,
        dephasing=0.0,
        pumping=0.0,
        collective_emission=0.0,
        collective_dephasing=0.0,
        collective_pumping=0.0,
    ):
        self.N = N
        self.hamiltonian = hamiltonian
        self.emission = emission
        self.dephasing = dephasing
        self.pumping = pumping
        self.collective_emission = collective_emission
        self.collective_dephasing = collective_dephasing
        self.collective_pumping = collective_pumping
        self.nds = num_dicke_states(self.N)
        self.dshape = (num_dicke_states(self.N), num_dicke_states(self.N))

    def __repr__(self):
        """Print the current parameters of the system."""
        string = []
        string.append("N = {}".format(self.N))
        string.append("Hilbert space dim = {}".format(self.dshape))
        string.append("Number of Dicke states = {}".format(self.nds))
        string.append(
            "Liouvillian space dim = {}".format((self.nds ** 2, self.nds ** 2))
        )
        if self.emission != 0:
            string.append("emission = {}".format(self.emission))
        if self.dephasing != 0:
            string.append("dephasing = {}".format(self.dephasing))
        if self.pumping != 0:
            string.append("pumping = {}".format(self.pumping))
        if self.collective_emission != 0:
            string.append(
                "collective_emission = {}".format(self.collective_emission)
            )
        if self.collective_dephasing != 0:
            string.append(
                "collective_dephasing = {}".format(self.collective_dephasing)
            )
        if self.collective_pumping != 0:
            string.append(
                "collective_pumping = {}".format(self.collective_pumping)
            )
        return "\n".join(string)

    def lindbladian(self):
        """Build the Lindbladian superoperator of the dissipative dynamics.

        Returns
        -------
        lindbladian : :class:`qutip.Qobj`
            The Lindbladian matrix as a `qutip.Qobj`.
        """
        cythonized_dicke = _Dicke(
            int(self.N),
            float(self.emission),
            float(self.dephasing),
            float(self.pumping),
            float(self.collective_emission),
            float(self.collective_dephasing),
            float(self.collective_pumping),
        )
        return cythonized_dicke.lindbladian()

    def liouvillian(self):
        """Build the total Liouvillian using the Dicke basis.

        Returns
        -------
        liouv : :class:`qutip.Qobj`
            The Liouvillian matrix for the system.
        """
        lindblad = self.lindbladian()
        if self.hamiltonian is None:
            liouv = lindblad

        else:
            hamiltonian = self.hamiltonian
            hamiltonian_superoperator = -1j * spre(hamiltonian) + 1j * spost(
                hamiltonian
            )
            liouv = lindblad + hamiltonian_superoperator
        return liouv

    def pisolve(self, initial_state, tlist):
        """
        Solve for diagonal Hamiltonians and initial states faster.

        Parameters
        ==========
        initial_state : :class:`qutip.Qobj`
            An initial state specified as a density matrix of
            `qutip.Qbj` type.

        tlist: ndarray
            A 1D numpy array of list of timesteps to integrate

        Returns
        =======
        result: list
            A dictionary of the type `qutip.piqs.Result` which holds the
            results of the evolution.
        """
        if _isdiagonal(initial_state) == False:
            msg = "`pisolve` requires a diagonal initial density matrix. "
            msg += "In general construct the Liouvillian using "
            msg += "`piqs.liouvillian` and use qutip.mesolve."
            raise ValueError(msg)

        if self.hamiltonian and _isdiagonal(self.hamiltonian) == False:
            msg = "`pisolve` should only be used for diagonal Hamiltonians. "
            msg += "Construct the Liouvillian using `piqs.liouvillian` and"
            msg += "  use `qutip.mesolve`."
            raise ValueError(msg)

        if initial_state.full().shape != self.dshape:
            msg = "Initial density matrix should be diagonal."
            raise ValueError(msg)

        pim = Pim(
            self.N,
            self.emission,
            self.dephasing,
            self.pumping,
            self.collective_emission,
            self.collective_pumping,
            self.collective_dephasing,
        )
        result = pim.solve(initial_state, tlist)
        return result

    def c_ops(self):
        """Build collapse operators in the full Hilbert space 2^N.

        Returns
        -------
        c_ops_list: list
            The list with the collapse operators in the 2^N Hilbert space.
        """
        ce = self.collective_emission
        cd = self.collective_dephasing
        cp = self.collective_pumping
        c_ops_list = collapse_uncoupled(
            N=self.N,
            emission=self.emission,
            dephasing=self.dephasing,
            pumping=self.pumping,
            collective_emission=ce,
            collective_dephasing=cd,
            collective_pumping=cp,
        )
        return c_ops_list

    def coefficient_matrix(self):
        """Build coefficient matrix for ODE for a diagonal problem.

        Returns
        -------
        M: ndarray
            The matrix M of the coefficients for the ODE dp/dt = Mp.
            p is the vector of the diagonal matrix elements
            of the density matrix rho in the Dicke basis.
        """
        diagonal_system = Pim(
            N=self.N,
            emission=self.emission,
            dephasing=self.dephasing,
            pumping=self.pumping,
            collective_emission=self.collective_emission,
            collective_dephasing=self.collective_dephasing,
            collective_pumping=self.collective_pumping,
        )
        coef_matrix = diagonal_system.coefficient_matrix()
        return coef_matrix


# ============================================================================
# Adding a faster version to make a Permutational Invariant matrix
# ============================================================================
def tau_column(tau, k, j):
    """
    Determine the column index for the non-zero elements of the matrix for a
    particular row `k` and the value of `j` from the Dicke space.

    Parameters
    ----------
    tau: str
        The tau function to check for this `k` and `j`.

    k: int
        The row of the matrix M for which the non zero elements have
        to be calculated.

    j: float
        The value of `j` for this row.
    """
    # In the notes, we indexed from k = 1, here we do it from k = 0
    k = k + 1
    mapping = {
        "tau3": k - (2 * j + 3),
        "tau2": k - 1,
        "tau4": k + (2 * j - 1),
        "tau5": k - (2 * j + 2),
        "tau1": k,
        "tau6": k + (2 * j),
        "tau7": k - (2 * j + 1),
        "tau8": k + 1,
        "tau9": k + (2 * j + 1),
    }
    # we need to decrement k again as indexing is from 0
    return int(mapping[tau] - 1)


class Pim(object):
    """
    The Permutation Invariant Matrix class.

    Initialize the class with the parameters for generating a Permutation
    Invariant matrix which evolves a given diagonal initial state `p` as:

                                dp/dt = Mp

    Parameters
    ----------
    N: int
        The number of two-level systems.

    emission: float
        Incoherent emission coefficient (also nonradiative emission).
        default: 0.0

    dephasing: float
        Local dephasing coefficient.
        default: 0.0

    pumping: float
        Incoherent pumping coefficient.
        default: 0.0

    collective_emission: float
        Collective (superradiant) emmission coefficient.
        default: 0.0

    collective_pumping: float
        Collective pumping coefficient.
        default: 0.0

    collective_dephasing: float
        Collective dephasing coefficient.
        default: 0.0

    Attributes
    ----------
    N: int
        The number of two-level systems.

    emission: float
        Incoherent emission coefficient (also nonradiative emission).
        default: 0.0

    dephasing: float
        Local dephasing coefficient.
        default: 0.0

    pumping: float
        Incoherent pumping coefficient.
        default: 0.0

    collective_emission: float
        Collective (superradiant) emmission coefficient.
        default: 0.0

    collective_dephasing: float
        Collective dephasing coefficient.
        default: 0.0

    collective_pumping: float
        Collective pumping coefficient.
        default: 0.0

    M: dict
        A nested dictionary of the structure {row: {col: val}} which holds
        non zero elements of the matrix M
    """

    def __init__(
        self,
        N,
        emission=0.0,
        dephasing=0,
        pumping=0,
        collective_emission=0,
        collective_pumping=0,
        collective_dephasing=0,
    ):
        self.N = N
        self.emission = emission
        self.dephasing = dephasing
        self.pumping = pumping
        self.collective_pumping = collective_pumping
        self.collective_dephasing = collective_dephasing
        self.collective_emission = collective_emission
        self.M = {}

    def isdicke(self, dicke_row, dicke_col):
        """
        Check if an element in a matrix is a valid element in the Dicke space.
        Dicke row: j value index. Dicke column: m value index.
        The function returns True if the element exists in the Dicke space and
        False otherwise.

        Parameters
        ----------
        dicke_row, dicke_col : int
            Index of the element in Dicke space which needs to be checked
        """
        rows = self.N + 1
        cols = 0

        if (self.N % 2) == 0:
            cols = int(self.N / 2 + 1)
        else:
            cols = int(self.N / 2 + 1 / 2)
        if (dicke_row > rows) or (dicke_row < 0):
            return False
        if (dicke_col > cols) or (dicke_col < 0):
            return False
        if (dicke_row < int(rows / 2)) and (dicke_col > dicke_row):
            return False
        if (dicke_row >= int(rows / 2)) and (rows - dicke_row <= dicke_col):
            return False
        else:
            return True

    def tau_valid(self, dicke_row, dicke_col):
        """
        Find the Tau functions which are valid for this value of (dicke_row,
        dicke_col) given the number of TLS. This calculates the valid tau
        values and reurns a dictionary specifying the tau function name and
        the value.

        Parameters
        ----------
        dicke_row, dicke_col : int
            Index of the element in Dicke space which needs to be checked.

        Returns
        -------
        taus: dict
            A dictionary of key, val as {tau: value} consisting of the valid
            taus for this row and column of the Dicke space element.
        """
        tau_functions = [
            self.tau3,
            self.tau2,
            self.tau4,
            self.tau5,
            self.tau1,
            self.tau6,
            self.tau7,
            self.tau8,
            self.tau9,
        ]
        N = self.N
        if self.isdicke(dicke_row, dicke_col) is False:
            return False
        # The 3x3 sub matrix surrounding the Dicke space element to
        # run the tau functions
        indices = [
            (dicke_row + x, dicke_col + y)
            for x in range(-1, 2)
            for y in range(-1, 2)
        ]
        taus = {}
        for idx, tau in zip(indices, tau_functions):
            if self.isdicke(idx[0], idx[1]):
                j, m = self.calculate_j_m(idx[0], idx[1])
                taus[tau.__name__] = tau(j, m)
        return taus

    def calculate_j_m(self, dicke_row, dicke_col):
        """
        Get the value of j and m for the particular Dicke space element.

        Parameters
        ----------
        dicke_row, dicke_col: int
            The row and column from the Dicke space matrix

        Returns
        -------
        j, m: float
            The j and m values.
        """
        N = self.N
        j = N / 2 - dicke_col
        m = N / 2 - dicke_row
        return (j, m)

    def calculate_k(self, dicke_row, dicke_col):
        """
        Get k value from the current row and column element in the Dicke space.

        Parameters
        ----------
        dicke_row, dicke_col: int
            The row and column from the Dicke space matrix.
        Returns
        -------
        k: int
            The row index for the matrix M for given Dicke space
            element.
        """
        N = self.N
        if dicke_row == 0:
            k = dicke_col
        else:
            k = int(
                ((dicke_col) / 2) * (2 * (N + 1) - 2 * (dicke_col - 1))
                + (dicke_row - (dicke_col))
            )
        return k

    def coefficient_matrix(self):
        """
        Generate the matrix M governing the dynamics for diagonal cases.

        If the initial density matrix and the Hamiltonian is diagonal, the
        evolution of the system is given by the simple ODE: dp/dt = Mp.
        """
        N = self.N
        nds = num_dicke_states(N)
        rows = self.N + 1
        cols = 0

        sparse_M = lil_matrix((nds, nds), dtype=float)
        if (self.N % 2) == 0:
            cols = int(self.N / 2 + 1)
        else:
            cols = int(self.N / 2 + 1 / 2)
        for (dicke_row, dicke_col) in np.ndindex(rows, cols):
            if self.isdicke(dicke_row, dicke_col):
                k = int(self.calculate_k(dicke_row, dicke_col))
                row = {}
                taus = self.tau_valid(dicke_row, dicke_col)
                for tau in taus:
                    j, m = self.calculate_j_m(dicke_row, dicke_col)
                    current_col = tau_column(tau, k, j)
                    sparse_M[k, int(current_col)] = taus[tau]
        return sparse_M.tocsr()

    def solve(self, rho0, tlist):
        """
        Solve the ODE for the evolution of diagonal states and Hamiltonians.
        """
        output = Result()
        output.solver = "pisolve"
        output.times = tlist
        output.states = []
        output.states.append(Qobj(rho0))
        rhs_generate = lambda y, tt, M: M.dot(y)
        rho0_flat = np.diag(np.real(rho0.full()))
        L = self.coefficient_matrix()
        rho_t = odeint(rhs_generate, rho0_flat, tlist, args=(L,))
        for r in rho_t[1:]:
            diag = np.diag(r)
            output.states.append(Qobj(diag))
        return output

    def tau1(self, j, m):
        """
        Calculate coefficient matrix element relative to (j, m, m).
        """
        yS = self.collective_emission
        yL = self.emission
        yD = self.dephasing
        yP = self.pumping
        yCP = self.collective_pumping
        N = float(self.N)
        spontaneous = yS * (1 + j - m) * (j + m)
        losses = yL * (N / 2 + m)
        pump = yP * (N / 2 - m)
        collective_pump = yCP * (1 + j + m) * (j - m)
        if j == 0:
            dephase = yD * N / 4
        else:
            dephase = yD * (N / 4 - m ** 2 * ((1 + N / 2) / (2 * j * (j + 1))))
        t1 = spontaneous + losses + pump + dephase + collective_pump
        return -t1

    def tau2(self, j, m):
        """
        Calculate coefficient matrix element relative to (j, m+1, m+1).
        """
        yS = self.collective_emission
        yL = self.emission
        N = float(self.N)
        spontaneous = yS * (1 + j - m) * (j + m)
        losses = yL * (
            ((N / 2 + 1) * (j - m + 1) * (j + m)) / (2 * j * (j + 1))
        )
        t2 = spontaneous + losses
        return t2

    def tau3(self, j, m):
        """
        Calculate coefficient matrix element relative to (j+1, m+1, m+1).
        """
        yL = self.emission
        N = float(self.N)
        num = (j + m - 1) * (j + m) * (j + 1 + N / 2)
        den = 2 * j * (2 * j + 1)
        t3 = yL * (num / den)
        return t3

    def tau4(self, j, m):
        """
        Calculate coefficient matrix element relative to (j-1, m+1, m+1).
        """
        yL = self.emission
        N = float(self.N)
        num = (j - m + 1) * (j - m + 2) * (N / 2 - j)
        den = 2 * (j + 1) * (2 * j + 1)
        t4 = yL * (num / den)
        return t4

    def tau5(self, j, m):
        """
        Calculate coefficient matrix element relative to (j+1, m, m).
        """
        yD = self.dephasing
        N = float(self.N)
        num = (j - m) * (j + m) * (j + 1 + N / 2)
        den = 2 * j * (2 * j + 1)
        t5 = yD * (num / den)
        return t5

    def tau6(self, j, m):
        """
        Calculate coefficient matrix element relative to (j-1, m, m).
        """
        yD = self.dephasing
        N = float(self.N)
        num = (j - m + 1) * (j + m + 1) * (N / 2 - j)
        den = 2 * (j + 1) * (2 * j + 1)
        t6 = yD * (num / den)
        return t6

    def tau7(self, j, m):
        """
        Calculate coefficient matrix element relative to (j+1, m-1, m-1).
        """
        yP = self.pumping
        N = float(self.N)
        num = (j - m - 1) * (j - m) * (j + 1 + N / 2)
        den = 2 * j * (2 * j + 1)
        t7 = yP * (float(num) / den)
        return t7

    def tau8(self, j, m):
        """
        Calculate coefficient matrix element relative to (j, m-1, m-1).
        """
        yP = self.pumping
        yCP = self.collective_pumping
        N = float(self.N)

        num = (1 + N / 2) * (j - m) * (j + m + 1)
        den = 2 * j * (j + 1)
        pump = yP * (float(num) / den)
        collective_pump = yCP * (j - m) * (j + m + 1)
        t8 = pump + collective_pump
        return t8

    def tau9(self, j, m):
        """
        Calculate coefficient matrix element relative to (j-1, m-1, m-1).
        """
        yP = self.pumping
        N = float(self.N)
        num = (j + m + 1) * (j + m + 2) * (N / 2 - j)
        den = 2 * (j + 1) * (2 * j + 1)
        t9 = yP * (float(num) / den)
        return t9
