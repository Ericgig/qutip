from .. import (
    Qobj, tensor, identity, ket2dm,
    sigmax, sigmay, sigmaz,
    sigmap, sigmam,
)
from scipy.sparse import dok_matrix, block_diag
from ._piqs import jmm1_dictionary, _num_dicke_states


__all__ = [
    "spin_algebra",
    "jspin",
    "collapse_uncoupled",
    "dicke_basis",
    "dicke",
    "excited",
    "superradiant",
    "css",
    "ghz",
    "ground",
    "block_matrix",
]

#Dicke only
def spin_algebra(N, op=None):
    """Create the list [sx, sy, sz] with the spin operators.

    The operators are constructed for a collection of N two-level systems
    (TLSs). Each element of the list, i.e., sx, is a vector of `qutip.Qobj`
    objects (spin matrices), as it cointains the list of the SU(2) Pauli
    matrices for the N TLSs. Each TLS operator sx[i], with i = 0, ..., (N-1),
    is placed in a :math:`2^N`-dimensional Hilbert space.

    Notes
    -----
    sx[i] is :math:`\\frac{\\sigma_x}{2}` in the composite Hilbert space.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    Returns
    -------
    spin_operators: list or :class: qutip.Qobj
        A list of `qutip.Qobj` operators - [sx, sy, sz] or the
        requested operator.
    """
    # 1. Define N TLS spin-1/2 matrices in the uncoupled basis
    N = int(N)
    sx = [0 for i in range(N)]
    sy = [0 for i in range(N)]
    sz = [0 for i in range(N)]
    sp = [0 for i in range(N)]
    sm = [0 for i in range(N)]

    sx[0] = 0.5 * sigmax()
    sy[0] = 0.5 * sigmay()
    sz[0] = 0.5 * sigmaz()
    sp[0] = sigmap()
    sm[0] = sigmam()

    # 2. Place operators in total Hilbert space
    for k in range(N - 1):
        sx[0] = tensor(sx[0], identity(2))
        sy[0] = tensor(sy[0], identity(2))
        sz[0] = tensor(sz[0], identity(2))
        sp[0] = tensor(sp[0], identity(2))
        sm[0] = tensor(sm[0], identity(2))

    # 3. Cyclic sequence to create all N operators
    a = [i for i in range(N)]
    b = [[a[i - i2] for i in range(N)] for i2 in range(N)]

    # 4. Create N operators
    for i in range(1, N):
        sx[i] = sx[0].permute(b[i])
        sy[i] = sy[0].permute(b[i])
        sz[i] = sz[0].permute(b[i])
        sp[i] = sp[0].permute(b[i])
        sm[i] = sm[0].permute(b[i])

    spin_operators = [sx, sy, sz]

    if not op:
        return spin_operators
    elif op == "x":
        return sx
    elif op == "y":
        return sy
    elif op == "z":
        return sz
    elif op == "+":
        return sp
    elif op == "-":
        return sm
    else:
        raise TypeError("Invalid type")


def _jspin_uncoupled(N, op=None):
    """Construct the the collective spin algebra in the uncoupled basis.

    jx, jy, jz, jp, jm are constructed in the uncoupled basis of the
    two-level system (TLS). Each collective operator is placed in a
    Hilbert space of dimension 2^N.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    op: str
        The operator to return 'x','y','z','+','-'.
        If no operator given, then output is the list of operators
        for ['x','y','z',].

    Returns
    -------
    collective_operators: list or :class: qutip.Qobj
        A list of `qutip.Qobj` representing all the operators in
        uncoupled" basis or a single operator requested.
    """
    # 1. Define N TLS spin-1/2 matrices in the uncoupled basis
    N = int(N)

    sx, sy, sz = spin_algebra(N)
    sp, sm = spin_algebra(N, "+"), spin_algebra(N, "-")

    jx = sum(sx)
    jy = sum(sy)
    jz = sum(sz)
    jp = sum(sp)
    jm = sum(sm)

    collective_operators = [jx, jy, jz]

    if not op:
        return collective_operators
    elif op == "x":
        return jx
    elif op == "y":
        return jy
    elif op == "z":
        return jz
    elif op == "+":
        return jp
    elif op == "-":
        return jm
    else:
        raise TypeError("Invalid type")


#Dicke only
def jspin(N, op=None, basis="dicke"):
    r"""
    Calculate the list of collective operators of the total algebra.

    The Dicke basis :math:`\lvert j,m\rangle\langle j,m'\rvert` is used by
    default. Otherwise with "uncoupled" the operators are in a
    :math:`2^N` space.

    Parameters
    ----------
    N: int
        Number of two-level systems.

    op: str
        The operator to return 'x','y','z','+','-'.
        If no operator given, then output is the list of operators
        for ['x','y','z'].

    basis: str
        The basis of the operators - "dicke" or "uncoupled"
        default: "dicke".

    Returns
    -------
    j_alg: list or :class: qutip.Qobj
        A list of `qutip.Qobj` representing all the operators in
        the "dicke" or "uncoupled" basis or a single operator requested.
    """
    if basis == "uncoupled":
        return _jspin_uncoupled(N, op)

    nds = num_dicke_states(N)
    num_ladders = num_dicke_ladders(N)
    jz_operator = dok_matrix((nds, nds), dtype=np.complex128)
    jp_operator = dok_matrix((nds, nds), dtype=np.complex128)
    jm_operator = dok_matrix((nds, nds), dtype=np.complex128)
    s = 0

    for k in range(0, num_ladders):
        j = 0.5 * N - k
        mmax = int(2 * j + 1)
        for i in range(0, mmax):
            m = j - i
            jz_operator[s, s] = m
            if (s + 1) in range(0, nds):
                jp_operator[s, s + 1] = ap(j, m - 1)
            if (s - 1) in range(0, nds):
                jm_operator[s, s - 1] = am(j, m + 1)
            s = s + 1

    jx_operator = 1 / 2 * (jp_operator + jm_operator)
    jy_operator = 1j / 2 * (jm_operator - jp_operator)
    jx = Qobj(jx_operator)
    jy = Qobj(jy_operator)
    jz = Qobj(jz_operator)
    jp = Qobj(jp_operator)
    jm = Qobj(jm_operator)

    if not op:
        return [jx, jy, jz]
    if op == "+":
        return jp
    elif op == "-":
        return jm
    elif op == "x":
        return jx
    elif op == "y":
        return jy
    elif op == "z":
        return jz
    else:
        raise TypeError("Invalid type")


def collapse_uncoupled(
    N,
    emission=0.0,
    dephasing=0.0,
    pumping=0.0,
    collective_emission=0.0,
    collective_dephasing=0.0,
    collective_pumping=0.0,
):
    """
    Create the collapse operators (c_ops) of the Lindbladian in the
    uncoupled basis

    These operators are in the uncoupled basis of the two-level
    system (TLS) SU(2) Pauli matrices.

    Notes
    -----
    The collapse operator list can be given to `qutip.mesolve`.
    Notice that the operators are placed in a Hilbert space of
    dimension :math:`2^N`. Thus the method is suitable only for
    small N (of the order of 10).

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

    Returns
    -------
    c_ops: list
        The list of collapse operators as `qutip.Qobj` for the system.
    """
    N = int(N)

    if N > 10:
        msg = "N > 10. dim(H) = 2^N. "
        msg += "Better use `piqs.lindbladian` to reduce Hilbert space "
        msg += "dimension and exploit permutational symmetry."
        raise Warning(msg)

    [sx, sy, sz] = spin_algebra(N)
    sp, sm = spin_algebra(N, "+"), spin_algebra(N, "-")
    [jx, jy, jz] = jspin(N, basis="uncoupled")
    jp, jm = (
        jspin(N, "+", basis="uncoupled"),
        jspin(N, "-", basis="uncoupled"),
    )

    c_ops = []

    if emission != 0:
        for i in range(0, N):
            c_ops.append(np.sqrt(emission) * sm[i])

    if dephasing != 0:
        for i in range(0, N):
            c_ops.append(np.sqrt(dephasing) * sz[i])

    if pumping != 0:
        for i in range(0, N):
            c_ops.append(np.sqrt(pumping) * sp[i])

    if collective_emission != 0:
        c_ops.append(np.sqrt(collective_emission) * jm)

    if collective_dephasing != 0:
        c_ops.append(np.sqrt(collective_dephasing) * jz)

    if collective_pumping != 0:
        c_ops.append(np.sqrt(collective_pumping) * jp)

    return c_ops


# State definitions in the Dicke basis with an option for basis transformation
def dicke_basis(N, jmm1=None, density_matrix=True, *, dtype=None):
    r"""
    Initialize the density matrix of a Dicke state for several (j, m, m1).

    This function can be used to build arbitrary states in the Dicke basis
    :math:`\lvert j, m\rangle\langle j, m'\rvert`. We create coefficients for
    each (j, m, m1) value in the dictionary jmm1. The mapping for the (i, k)
    index of the density matrix to the :math:`\lvert j, m\rangle` values is
    given by the cythonized function `jmm1_dictionary`. A density matrix is
    created from the given dictionary of coefficients for each (j, m, m1).

    Parameters
    ----------
    N: int
        The number of two-level systems.

    jmm1: dict
        A dictionary of {(j, m, m1): p} that gives a density p for the
        (j, m, m1) matrix element.

    density_matrix: bool
        Whether to return a density matrix or a ket.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    rho: :class: qutip.Qobj
        The density matrix in the Dicke basis.
    """
    if jmm1 is None:
        msg = "Please specify the jmm1 values as a dictionary"
        msg += "or use the `excited(N)` function to create an"
        msg += "excited state where jmm1 = {(N/2, N/2, N/2): 1}"
        raise AttributeError(msg)

    nds = _num_dicke_states(N)
    rho = np.zeros(nds)
    jmm1_dict = jmm1_dictionary(N)[1]
    for key in jmm1:
        i, _ = jmm1_dict[key]
        rho[i] = jmm1[key]

    state = Qobj(rho)
    if density_matrix:
        state = state.proj()
    if dtype:
        state = state.to(dtype)
    return state


def dicke(N, j, m, density_matrix=True, *, dtype=None):
    r"""
    Generate a Dicke state as a pure density matrix in the Dicke basis.

    For instance, the superradiant state given by
    :math:`\lvert  j, m\rangle = \lvert 1, 0\rangle` for N = 2,
    and the state is represented as a density matrix of size (nds, nds) or
    (4, 4), with the (1, 1) element set to 1.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    j: float
        The eigenvalue j of the Dicke state (j, m).

    m: float
        The eigenvalue m of the Dicke state (j, m).

    density_matrix: bool
        Whether to return a density matrix or a ket.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    rho: :class: qutip.Qobj
        The density matrix.
    """
    nds = num_dicke_states(N)
    rho = np.zeros(nds)

    jmm1_dict = jmm1_dictionary(N)[1]

    i, _ = jmm1_dict[(j, m, m)]
    rho[i] = 1.0

    state = Qobj(rho)
    if density_matrix:
        state = state.proj()
    if dtype:
        state = state.to(dtype)
    return state


def excited(N, density_matrix=True, *, dtype=None):
    """
    Generate the density matrix for the excited state.

    This state is given by (N/2, N/2) in the default Dicke basis. If the
    argument `basis` is "uncoupled" then it generates the state in a
    2**N dim Hilbert space.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    density_matrix: bool
        Whether to return a density matrix or a ket.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    state: :class: qutip.Qobj
        The excited state density matrix in the requested basis.
    """
    jmm1 = {(N / 2, N / 2, N / 2): 1}
    return dicke_basis(N, jmm1, density_matrix, dtype=dtype)


def superradiant(N, density_matrix=True, *, dtype=None):
    """
    Generate the density matrix of the superradiant state.

    This state is given by (N/2, 0) or (N/2, 0.5) in the Dicke basis.
    If the argument `basis` is "uncoupled" then it generates the state
    in a 2**N dim Hilbert space.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    density_matrix: bool
        Whether to return a density matrix or a ket.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    state: :class: qutip.Qobj
        The superradiant state density matrix in the requested basis.
    """
    if N % 2 == 0:
        jmm1 = {(N / 2, 0, 0): 1.0}
    else:
        jmm1 = {(N / 2, 0.5, 0.5): 1.0}
    return dicke_basis(N, jmm1, density_matrix, dtype=dtype)


#dicke only
def coherent_spin_state(
    N,
    x=1 / np.sqrt(2),
    y=1 / np.sqrt(2),
    coordinates="cartesian",
    density_matrix=True, *, dtype=None
):
    r"""
    Generate the density matrix of the Coherent Spin State (CSS).

    It can be defined as,
    :math:`\lvert CSS\rangle = \prod_i^N(a\lvert1\rangle_i+b\lvert0\rangle_i)`
    with :math:`a = sin(\frac{\theta}{2})`,
    :math:`b = e^{i \phi}\cos(\frac{\theta}{2})`.
    The default basis is that of Dicke space
    :math:`\lvert j, m\rangle \langle j, m'\rvert`.
    The default state is the symmetric CSS,
    :math:`\lvert CSS\rangle = \lvert+\rangle`.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    x, y: float
        The coefficients of the CSS state.

    coordinates: str
        Either "cartesian" or "polar". If polar then the coefficients
        are constructed as sin(x/2), cos(x/2)e^(iy).

    density_matrix: bool
        Whether to return a density matrix or a ket.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    rho: :class: qutip.Qobj
        The CSS state density matrix.
    """
    if coordinates == "polar":
        a = np.cos(0.5 * x) * np.exp(1j * y)
        b = np.sin(0.5 * x)
    else:
        a = x
        b = y

    nds = num_dicke_states(N)
    psi = np.zeros(nds, dtype=np.complex128)

    j = 0.5 * N
    mmax = int(2 * j + 1)
    for i in range(0, mmax):
        m = j - i
        psi[i] = (
            np.sqrt(float(energy_degeneracy(N, m)))
            * a ** (N * 0.5 + m)
            * b ** (N * 0.5 - m)
        )
    state = Qobj(psi)
    if density_matrix:
        state = state.proj()
    if dtype:
        state = state.to(dtype)
    return state


def ghz(N, density_matrix=True):
    """
    Generate the density matrix of the GHZ state.

    If the argument `basis` is "uncoupled" then it generates the state
    in a :math:`2^N`-dimensional Hilbert space.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    density_matrix: bool
        Whether to return a density matrix or a ket.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    state: :class: qutip.Qobj
        The GHZ state density matrix in the requested basis.
    """
    nds = _num_dicke_states(N)
    state = (basis(nds, 0, dtype=dtype) + basis(nds, N, dtype=dtype)).unit()
    if density_matrix:
        state = state.proj()
    return state


def ground(N, density_matrix=True, *, dtype=None):
    """
    Generate the density matrix of the ground state.

    This state is given by (N/2, -N/2) in the Dicke basis. If the argument
    `basis` is "uncoupled" then it generates the state in a
    :math:`2^N`-dimensional Hilbert space.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    density_matrix: bool
        Whether to return a density matrix or a ket.

    dtype : type or str
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    state: :class: qutip.Qobj
        The ground state density matrix in the requested basis.
    """
    nds = _num_dicke_states(N)
    state = basis(nds, N, dtype=dtype)
    if density_matrix:
        state = state.proj()
    return state


def block_matrix(N, elements="ones"):
    """Construct the block-diagonal matrix for the Dicke basis.

    Parameters
    ----------
    N : int
        Number of two-level systems.
    elements : str {'ones' (default),'degeneracy'}

    Returns
    -------
    block_matr : ndarray
        A 2D block-diagonal matrix with dimension (nds,nds),
        where nds is the number of Dicke states for N two-level
        systems. Filled with ones or the value of degeneracy
        at each matrix element.
    """
    # create a list with the sizes of the blocks, in order
    blocks_dimensions = int(N / 2 + 1 - 0.5 * (N % 2))
    blocks_list = [
        (2 * (i + 1 * (N % 2)) + 1 * ((N + 1) % 2))
        for i in range(blocks_dimensions)
    ]
    blocks_list = np.flip(blocks_list, 0)
    # create a list with each block matrix as element
    square_blocks = []
    k = 0
    for i in blocks_list:
        if elements == "ones":
            square_blocks.append(np.ones((i, i)))
        elif elements == "degeneracy":
            j = N / 2 - k
            dj = state_degeneracy(N, j)
            square_blocks.append(dj * np.ones((i, i)))
        k = k + 1
    return block_diag(square_blocks)
