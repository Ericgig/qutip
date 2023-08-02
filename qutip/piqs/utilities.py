from decimal import Decimal
from math import factorial
import numpy as np

__all__ = [
    "state_degeneracy",
    "m_degeneracy",
    "energy_degeneracy",
    "ap",
    "am",
]


# Utility functions for properties of the Dicke space
def energy_degeneracy(N, m):
    """Calculate the number of Dicke states with same energy.

    The use of the `Decimals` class allows to explore N > 1000,
    unlike the built-in function `scipy.special.binom`

    Parameters
    ----------
    N: int
        The number of two-level systems.

    m: float
        Total spin z-axis projection eigenvalue.
        This is proportional to the total energy.

    Returns
    -------
    degeneracy: int
        The energy degeneracy
    """
    numerator = Decimal(factorial(N))
    d1 = Decimal(factorial(_ensure_int(N / 2 + m)))
    d2 = Decimal(factorial(_ensure_int(N / 2 - m)))
    degeneracy = numerator / (d1 * d2)
    return int(degeneracy)


def state_degeneracy(N, j):
    r"""Calculate the degeneracy of the Dicke state.

    Each state :math:`\lvert j, m\rangle` includes D(N,j) irreducible
    representations :math:`\lvert j, m, \alpha\rangle`.

    Uses Decimals to calculate higher numerator and denominators numbers.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    j: float
        Total spin eigenvalue (cooperativity).

    Returns
    -------
    degeneracy: int
        The state degeneracy.
    """
    if j < 0:
        raise ValueError("j value should be >= 0")
    numerator = Decimal(factorial(N)) * Decimal(2 * j + 1)
    denominator_1 = Decimal(factorial(_ensure_int(N / 2 + j + 1)))
    denominator_2 = Decimal(factorial(_ensure_int(N / 2 - j)))
    degeneracy = numerator / (denominator_1 * denominator_2)
    degeneracy = int(np.round(float(degeneracy)))
    return degeneracy


def m_degeneracy(N, m):
    r"""Calculate the number of Dicke states :math:`\lvert j, m\rangle` with
    same energy.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    m: float
        Total spin z-axis projection eigenvalue (proportional to the total
        energy).

    Returns
    -------
    degeneracy: int
        The m-degeneracy.
    """
    jvals = j_vals(N)
    maxj = np.max(jvals)
    if m < -maxj:
        e = "m value is incorrect for this N."
        e += " Minimum m value can be {}".format(-maxj)
        raise ValueError(e)
    degeneracy = N / 2 + 1 - abs(m)
    return int(degeneracy)


def ap(j, m):
    r"""
    Calculate the coefficient ``ap`` by applying :math:`J_+\lvert j,m\rangle`.

    The action of ap is given by:
    :math:`J_{+}\lvert j, m\rangle = A_{+}(j, m) \lvert j, m+1\rangle`

    Parameters
    ----------
    j, m: float
        The value for j and m in the dicke basis :math:`\lvert j, m\rangle`.

    Returns
    -------
    a_plus: float
        The value of :math:`a_{+}`.
    """
    a_plus = np.sqrt((j - m) * (j + m + 1))
    return a_plus


def am(j, m):
    r"""Calculate the operator ``am`` used later.

    The action of ``ap`` is given by:
    :math:`J_{-}\lvert j,m\rangle = A_{-}(jm)\lvert j,m-1\rangle`

    Parameters
    ----------
    j: float
        The value for j.

    m: float
        The value for m.

    Returns
    -------
    a_minus: float
        The value of :math:`a_{-}`.
    """
    a_minus = np.sqrt((j + m) * (j - m + 1))
    return a_minus
