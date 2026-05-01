from qutip import Qobj, qeye_like
from .cy.dysolve import cy_compute_integrals
from numpy.typing import ArrayLike
import numpy as np
import scipy as sp
from numbers import Number
import itertools


__all__ = ['DysolvePropagator', 'dysolve_propagator']


class DysolvePropagator:
    """
    A generator of propagator using Dysolve.
    https://arxiv.org/abs/2012.09282

    Parameters
    ----------
    H_0 : Qobj
        The base hamiltonian of the system.

    X : Qobj
        A cosine perturbation applied on the system.

    omega : float
        The frequency of the cosine perturbation.

    options : dict, optional
        Extra parameters.

        - "max_order"

            A given integer to indicate the highest order of
            approximation used to compute the propagators (default is 4).
            This corresponds to n in eq. (4) of Ref.

        - "a_tol"

            The absolute tolerance used when computing the propagators
            (default is 1e-10).

        - "max_dt"

            The maximum time increment used when computing propagators
            (default is 0.1).

    Notes
    -----
    The system's hamiltonian must be of the form
    H = H_0 + cos(omega*t)X for Dysolve to work.

    For the moment, only a cosine perturbation is allowed. Dysolve can
    manage more exotic perturbations, but this is not implemented yet.

    .. note:: Experimental.

    """

    def __init__(
        self,
        H_0: Qobj,
        X: Qobj,
        omega: float,
        options: dict[str] = None,
    ):
        # System
        self._eigenenergies, self._basis = H_0.eigenstates()
        self.H_0 = H_0
        self._H_0 = H_0.transform(self._basis)
        self.td = False
        self.perturbation = []
        for perturbation in X:
            oper = perturbation[0]
            omega = perturbation[1]
            if len(perturbation) >= 3:
                form = perturbation[2]
            else:
                form = "cos"
            if len(perturbation) >= 4:
                coeff = perturbation[3]
            else:
                coeff = 1.
            if isinstance(coeff, Coefficient):
                self.td = True
            else:
                oper = oper * coeff
                coeff = coefficient(1.)

            oper = oper.transform(self._basis)
            if form == "cos":
                self.perturbation.append((oper * 0.5, omega, coeff))
                self.perturbation.append((oper * 0.5, -omega, coeff))
            elif form == "sin":
                self.perturbation.append((oper * -0.5j, omega, coeff))
                self.perturbation.append((oper * 0.5j, -omega, coeff))
            elif form == "exp":
                self.perturbation.append((oper, omega, coeff))

        self.options = {
            "max_order" : 4,
            "a_tol" : 1e-10,
            "max_dt" : 0.1,
            "r_tol" : 10,
            "envelope_order" : 0,
        }
        if options:
            self.options.update(options)

        # Memoization
        self._dt_Sns = {}

        # Time propagator
        self.U = qeye_like(self.H_0).full()
        self.t = 0

    def __call__(self, t_f: float, t_i: float = 0.0) -> Qobj:
        """
        Computes the propagator from t_i to t_f. If t_i is not provided,
        computes the propagator from 0 to t_f.

        Parameters
        ----------
        t_f : float
            Final time of the evolution.

        t_i : float, default = 0.0
            Initial time of the evolution.

        Returns
        -------
        U : Qobj
            The propagator U(t_f, t_i) from t_i to t_f.

        Notes
        -----
        If t_f - t_i > max_dt, splits the evolution into smaller ones
        to then reconstruct U(t_f, t_i).

        Memoization is used. When ``t_f`` is a multiple of max_dt,
        first call may be slow but the next calls should be faster.

        """
        dt = self.options["max_dt"]
        if t_i == 0 and abs(t_f - self.t) <  abs(t_f - t_i):
            t_i = self.t
            U = self.U
        else:
            U = qeye_like(self.H_0).full()
        time_diff = t_f - t_i
        n_steps = abs(int(time_diff / dt))

        if n_steps == 0:
            pass
        elif time_diff > 0:
            for j in range(0, n_steps):
                U = self._get_subprop(t_i + j * dt) @ U
        else:
            for j in range(n_steps):
                U = self._get_subprop(t_i - (j + 1) * dt).dag() @ U

        # We only save propagator at time multiple of max_dt
        self.U = U
        self.t = t_i + n_steps * dt * np.sign(time_diff)

        remaining = time_diff - n_steps * dt * np.sign(time_diff)
        if abs(remaining) > self.options["a_tol"]:
            U = self._get_subprop(t_f - remaining, remaining) @ U

        return Qobj(U, self._H_0._dims, copy=False).transform(
            self._basis, True
        )

    def _get_subprop(self, current_time: float, dt: float = None) -> ArrayLike:
        """
        Computes a subpropagator U(current_time + dt, current_time).

        Parameters
        ----------
        current_time : float
            The starting time of the evolution. Can be positive or negative.

        dt : float
            The time increment.

        Returns
        -------
        subpropagator : ArrayLike
            U(current_time + dt, current_time).

        """
        if dt is None:
            dt = self.options["max_dt"]
        Sns = self._compute_Sns(dt)

        N = len(self._eigenenergies)
        num_omega = len(self.perturbation)

        subpropagator = np.zeros((N, N), dtype=np.complex128)
        subpropagator += Sns[0]

        ws_vec = np.array([p[1] for p in self.perturbation])
        ws = np.add.outer(self._eigenenergies, -self._eigenenergies)

        for n in range(1, self.options["max_order"] + 1):
            ws = np.add.outer(ws_vec, ws)
            subpropagator += (Sns[n] * np.exp(1j * ws * current_time)).reshape((-1, N, N)).sum(axis=0)

        return subpropagator

    def _compute_Sns(self, dt: float) -> dict:
        """
        Computes Sns for each omega vector. This implements a similar equation
        to eq. (14) in Ref, but the function "f" is not used to avoid dealing
        explicitly with limits.

        Parameters
        ----------
        dt : float
            The time increment.

        Returns
        -------
        Sns : dict
            Sns for each omega vector. key = order with the result for each
            omega vector.

        """
        if dt in self._dt_Sns:
            return self._dt_Sns[dt]
        else:
            self._dt_Sns[dt] = self._make_SNS_sparse(dt)
            return self._dt_Sns[dt]

    def _make_SNS_sparse(self, dt: float) -> dict:

        def _to_COO_format(matrix):
            """
            Convert to COO, (location, data) pair.
            """
            coo = scipy.sparse.coo_array(matrix)
            idx = np.c_[coo.row, coo.col]
            return idx, coo.data

        def _outer_matmul(left_coo, right_coo):
            """
            Outer matmul of COO arrays: einsum("abc,cde->abcde")
            """
            idx_l, data_l = left_coo
            idx_r, data_r = right_coo
            new_idx = []
            new_data = []

            for j in range(idx_r.shape[0]):
                idx = np.where(idx_l[:, -1] == idx_r[j, 0])[0]
                for i in idx:
                    new_idx.append(np.r_[idx_l[i, :], idx_r[j, 1:]])
                    new_data.append(data_l[i] * data_r[j])
            return np.array(new_idx), np.array(new_data)

        Sns = {}
        energies = self._eigenenergies
        num_ls = len(energies)
        omegas = [p[1] for p in self.perturbation]
        num_ws = len(omegas)

        dE = energies[:, np.newaxis] - energies[np.newaxis, :]
        exp_H_0_diag = np.exp(-1j * energies * dt)
        exp_H_0 = np.diag(exp_H_0_diag)
        Sns[0] = exp_H_0

        # Since usually the operators comes in pairs, we only
        # store unique operators and keeps and index from the
        # pertubation index to it's location.
        path_cache = []
        opers_loc = []
        for i, oper in enumerate([p[0] for p in self.perturbation]):
            if oper not in path_cache:
                opers_loc.append(len(path_cache))
                path_cache.append(oper)
            else:
                opers_loc.append(path_cache.index(oper))

        path_cache = {
            (idx,): _to_COO_format(oper.to("csr").data_as())
            for idx, oper in enumerate(path_cache)
        }

        for n in range(1, self.options["max_order"] + 1):
            shape = [num_ws] * n + [num_ls, num_ls]
            Sn = np.zeros(shape, dtype=np.complex128)

            for pert_idx in itertools.product(range(num_ws), repeat=n):
                current_omegas = [omegas[i] for i in pert_idx]
                unique_idx = tuple(opers_loc[idx] for idx in pert_idx)
                if unique_idx not in path_cache:
                    path_cache[unique_idx] = _outer_matmul(
                        path_cache[unique_idx[:1]], path_cache[unique_idx[1:]]
                    )

                paths, amplitudes = path_cache[unique_idx]
                ws_matrix = np.zeros((len(amplitudes), n))
                for i in range(n):
                    ws_matrix[:, i] = (
                        current_omegas[i] + dE[paths[:, i], paths[:, i+1]]
                    )

                integrals = np.array([
                    cy_compute_integrals(row, dt) for row in ws_matrix
                ])
                start_indices = paths[:, 0]
                end_indices = paths[:, -1]
                np.add.at(
                    Sn[*pert_idx],
                    (start_indices, end_indices),
                    amplitudes * integrals
                )

                Sn[*pert_idx] = exp_H_0 @ Sn[*pert_idx]

            Sn *= (-1j) ** n
            Sns[n] = Sn

        return Sns



def dysolve_propagator(
        H_0: Qobj,
        X: Qobj,
        omega: float,
        t: float | list[float],
        options: dict[str] = None
) -> Qobj | list[Qobj]:
    """
    A generator of propagator(s) using Dysolve.
    https://arxiv.org/abs/2012.09282.

    Parameters
    ----------
    H_0 : Qobj
        The hamiltonian of the system.

    X : Qobj
        A cosine perturbation applied on the system.

    omega : float
        The frequency of the cosine perturbation.

    t : float | list[float]
        Time or list of times for which to evaluate the propagator(s). If t
        is a single number, the propagator from 0 to t is computed. When
        t is a list, the propagators from the first time to each elements in
        t is returned. In that case, the first output will always be the
        identity matrix. Also, in that case, have the same time increment in
        between elements for better performance.

    options : dict, optional
        Extra parameters.

        - "max_order"

            A given integer to indicate the highest order of
            approximation used to compute the propagators (default is 4).
            This corresponds to n in eq. (4) of Ref.

        - "a_tol"

            The absolute tolerance used when computing the propagators
            (default is 1e-10).

        - "max_dt"

            The maximum time increment used when computing propagators
            (default is 0.1).

    Returns
    -------
    Us : Qobj | list[Qobj]
        The time evolution propagator U(t,0) if t is a single number or else
        a list of propagators [U(t[i], t[0])] for all elements t[i] in t.

    Notes
    -----
    The system's hamiltonian must be of the form
    H = H_0 + cos(omega*t)X for Dysolve to work.

    For the moment, only a cosine perturbation is allowed. Dysolve can
    manage more exotic perturbations, but this is not implemented yet.

    .. note:: Experimental.

    """
    if isinstance(t, Number):
        dysolve = DysolvePropagator(H_0, X, omega, options)
        return dysolve(t)

    else:
        Us = []
        Us.append(qeye_like(H_0))  # U(t_0, t_0) = identity

        dysolve = DysolvePropagator(H_0, X, omega, options)
        for i in range(len(t[:-1])):  # Compute individual U(t[i+1], t[i])
            U = dysolve(t[i+1], t[i])
            Us.append(U)

        for i in range(1, len(Us)):  # [U(t[i], t[0])]
            Us[i] = Us[i] @ Us[i - 1]

    return Us
