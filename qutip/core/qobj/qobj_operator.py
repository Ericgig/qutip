from .qobj_base import Qobj

class _RecOperator(Qobj):
    def __init__(self, data, dims, copy):
        super.__init__(data, dims, copy)
        if self._dims.type not in ["oper", "super"]:
            raise ValueError(
                f"Expected operator dimensions, but got {self._dims.type}"
            )

    def matrix_element(self, bra: Qobj, ket: Qobj) -> Qobj:
        """Calculates a matrix element.

        Gives the matrix element for the quantum object sandwiched between a
        `bra` and `ket` vector.

        Parameters
        ----------
        bra : :class:`.Qobj`
            Quantum object of type 'bra' or 'ket'

        ket : :class:`.Qobj`
            Quantum object of type 'ket'.

        Returns
        -------
        elem : complex
            Complex valued matrix element.

        Notes
        -----
        It is slightly more computationally efficient to use a ket
        vector for the 'bra' input.

        """
        if bra.type not in ('bra', 'ket') or ket.type not in ('bra', 'ket'):
            msg = "Can only calculate matrix elements between a bra and a ket."
            raise TypeError(msg)
        left, op, right = bra.data, self.data, ket.data
        if ket.isbra:
            right = right.adjoint()
        return _data.inner_op(left, op, right, bra.isket)

    def dnorm(self, B: Qobj = None) -> float:
        """Calculates the diamond norm, or the diamond distance to another
        operator.

        Parameters
        ----------
        B : :class:`.Qobj` or None
            If B is not None, the diamond distance d(A, B) = dnorm(A - B)
            between this operator and B is returned instead of the diamond norm.

        Returns
        -------
        d : float
            Either the diamond norm of this operator, or the diamond distance
            from this operator to B.

        """
        return qutip.dnorm(self, B)

    @property
    def ishp(self) -> bool:
        return False

    @property
    def iscp(self) -> bool:
        return False

    @property
    def istp(self) -> bool:
        return False

    @property
    def iscptp(self) -> bool:
        return False

    @property
    def isoper(self) -> bool:
        return True

    @property
    def isoperket(self) -> bool:
        return False

    @property
    def isoperbra(self) -> bool:
        return False

    @property
    def isbra(self) -> bool:
        return False

    @property
    def isket(self) -> bool:
        return False


class Operator(_RecOperator):
    def __init__(self, data, dims, copy):
        super.__init__(data, dims, copy)
        if not self._dims.issquare:
            raise ValueError(
                f"Expected square operator dimensions, "
                "but got {self._dims.type}."
            )

    def __pow__(self, n: int, m=None) -> Qobj:
        # calculates powers of Qobj
        if (
            m is not None
            or not isinstance(n, numbers.Integral)
            or n < 0
        ):
            return NotImplemented
        return Qobj(_data.pow(self._data, n),
                    dims=self._dims,
                    isherm=self._isherm,
                    isunitary=self._isunitary,
                    copy=False)

    def tr(self) -> complex:
        """Trace of a quantum object.

        Returns
        -------
        trace : float
            Returns the trace of the quantum object.

        """
        out = _data.trace(self._data)
        # This ensures that trace can return something that is not a number such
        # as a `tensorflow.Tensor` in qutip-tensorflow.
        if settings.core["auto_real_casting"] and self.isherm:
            out = out.real
        return out

    def diag(self) -> np.ndarray:
        """Diagonal elements of quantum object.

        Returns
        -------
        diags : array
            Returns array of ``real`` values if operators is Hermitian,
            otherwise ``complex`` values are returned.
        """
        # TODO: add a `diagonal` method to the data layer?
        out = _data.to(_data.CSR, self.data).as_scipy().diagonal()
        if settings.core["auto_real_casting"] and self.isherm:
            out = np.real(out)
        return out


    def expm(self, dtype: LayerType = None) -> Qobj:
        """Matrix exponential of quantum operator.

        Input operator must be square.

        Parameters
        ----------
        dtype : type
            The data-layer type that should be output.

        Returns
        -------
        oper : :class:`.Qobj`
            Exponentiated quantum operator.

        Raises
        ------
        TypeError
            Quantum operator is not square.
        """
        if not self._dims.issquare:
            raise TypeError("expm is only valid for square operators")
        if dtype is None and isinstance(self.data, (_data.CSR, _data.Dia)):
            dtype = _data.Dense
        return Qobj(_data.expm(self._data, dtype=dtype),
                    dims=self._dims,
                    isherm=self._isherm,
                    copy=False)

    def logm(self) -> Qobj:
        """Matrix logarithm of quantum operator.

        Input operator must be square.

        Returns
        -------
        oper : :class:`.Qobj`
            Logarithm of the quantum operator.

        Raises
        ------
        TypeError
            Quantum operator is not square.
        """
        if not self._dims.issquare:
            raise TypeError("expm is only valid for square operators")
        return Qobj(_data.logm(self._data),
                    dims=self._dims,
                    isherm=self._isherm,
                    copy=False)

    def sqrtm(
        self,
        sparse: bool = False,
        tol: float = 0,
        maxiter: int = 100000
    ) -> Qobj:
        """
        Sqrt of a quantum operator.  Operator must be square.

        Parameters
        ----------
        sparse : bool
            Use sparse eigenvalue/vector solver.
        tol : float
            Tolerance used by sparse solver (0 = machine precision).
        maxiter : int
            Maximum number of iterations used by sparse solver.

        Returns
        -------
        oper : :class:`.Qobj`
            Matrix square root of operator.

        Raises
        ------
        TypeError
            Quantum object is not square.

        Notes
        -----
        The sparse eigensolver is much slower than the dense version.
        Use sparse only if memory requirements demand it.
        """
        if self._dims[0] != self._dims[1]:
            raise TypeError('sqrt only valid on square matrices')
        return Qobj(_data.sqrtm(self._data),
                    dims=self._dims,
                    copy=False)

    def cosm(self) -> Qobj:
        """Cosine of a quantum operator.

        Operator must be square.

        Returns
        -------
        oper : :class:`.Qobj`
            Matrix cosine of operator.

        Raises
        ------
        TypeError
            Quantum object is not square.

        Notes
        -----
        Uses the Q.expm() method.

        """
        if self._dims[0] != self._dims[1]:
            raise TypeError('invalid operand for matrix cosine')
        return 0.5 * ((1j * self).expm() + (-1j * self).expm())

    def sinm(self) -> Qobj:
        """Sine of a quantum operator.

        Operator must be square.

        Returns
        -------
        oper : :class:`.Qobj`
            Matrix sine of operator.

        Raises
        ------
        TypeError
            Quantum object is not square.

        Notes
        -----
        Uses the Q.expm() method.
        """
        if self._dims[0] != self._dims[1]:
            raise TypeError('invalid operand for matrix sine')
        return -0.5j * ((1j * self).expm() - (-1j * self).expm())

    def inv(self, sparse: bool = False) -> Qobj:
        """Matrix inverse of a quantum operator

        Operator must be square.

        Returns
        -------
        oper : :class:`.Qobj`
            Matrix inverse of operator.

        Raises
        ------
        TypeError
            Quantum object is not square.
        """
        if self.data.shape[0] != self.data.shape[1]:
            raise TypeError('Invalid operand for matrix inverse')
        if isinstance(self.data, _data.CSR) and not sparse:
            data = _data.to(_data.Dense, self.data)
        else:
            data = self.data

        return Qobj(_data.inv(data),
                    dims=[self._dims[1], self._dims[0]],
                    copy=False)

    def check_herm(self) -> bool:
        """Check if the quantum object is hermitian.

        Returns
        -------
        isherm : bool
            Returns the new value of isherm property.
        """
        self._isherm = None
        return self.isherm

    def ptrace(self, sel: int | list[int], dtype: LayerType = None) -> Qobj:
        """
        Take the partial trace of the quantum object leaving the selected
        subspaces.  In other words, trace out all subspaces which are _not_
        passed.

        This is typically a function which acts on operators; bras and kets
        will be promoted to density matrices before the operation takes place
        since the partial trace is inherently undefined on pure states.

        For operators which are currently being represented as states in the
        superoperator formalism (i.e. the object has type `operator-ket` or
        `operator-bra`), the partial trace is applied as if the operator were
        in the conventional form.  This means that for any operator `x`,
        ``operator_to_vector(x).ptrace(0) == operator_to_vector(x.ptrace(0))``
        and similar for `operator-bra`.

        The story is different for full superoperators.  In the formalism that
        QuTiP uses, if an operator has dimensions (`dims`) of
        `[[2, 3], [2, 3]]` then it can be represented as a state on a Hilbert
        space of dimensions `[2, 3, 2, 3]`, and a superoperator would be an
        operator which acts on this joint space.  This function performs the
        partial trace on superoperators by letting the selected components
        refer to elements of the _joint_ _space_, and then returns a regular
        operator (of type `oper`).

        Parameters
        ----------
        sel : int or iterable of int
            An ``int`` or ``list`` of components to keep after partial trace.
            The selected subspaces will _not_ be reordered, no matter order
            they are supplied to `ptrace`.

        Returns
        -------
        oper : :class:`.Qobj`
            Quantum object representing partial trace with selected components
            remaining.
        """
        try:
            sel = sorted(sel)
        except TypeError:
            if not isinstance(sel, numbers.Integral):
                raise TypeError(
                    "selection must be an integer or list of integers"
                ) from None
            sel = [sel]
        if self.isoperket:
            dims = self.dims[0]
            data = qutip.vector_to_operator(self).data
        elif self.isoperbra:
            dims = self.dims[1]
            data = qutip.vector_to_operator(self.dag()).data
        elif self.issuper or self.isoper:
            dims = self.dims
            data = self.data
        else:
            dims = [self.dims[0] if self.isket else self.dims[1]] * 2
            data = _data.project(self.data)
        if dims[0] != dims[1]:
            raise ValueError("partial trace is not defined on non-square maps")
        dims = flatten(dims[0])
        new_data = _data.ptrace(data, dims, sel, dtype=dtype)
        new_dims = [[dims[x] for x in sel]] * 2 if sel else None
        out = Qobj(new_data, dims=new_dims, copy=False)
        if self.isoperket:
            return qutip.operator_to_vector(out)
        if self.isoperbra:
            return qutip.operator_to_vector(out).dag()
        return out


    def eigenstates(
        self,
        sparse: bool = False,
        sort: Literal["low", "high"] = 'low',
        eigvals: int = 0,
        tol: float = 0,
        maxiter: int = 100000,
        phase_fix: int = None
    ) -> tuple[np.ndarray, list[Qobj]]:
        """Eigenstates and eigenenergies.

        Eigenstates and eigenenergies are defined for operators and
        superoperators only.

        Parameters
        ----------
        sparse : bool
            Use sparse Eigensolver

        sort : str
            Sort eigenvalues (and vectors) 'low' to high, or 'high' to low.

        eigvals : int
            Number of requested eigenvalues. Default is all eigenvalues.

        tol : float
            Tolerance used by sparse Eigensolver (0 = machine precision).
            The sparse solver may not converge if the tolerance is set too low.

        maxiter : int
            Maximum number of iterations performed by sparse solver (if used).

        phase_fix : int, None
            If not None, set the phase of each kets so that ket[phase_fix,0]
            is real positive.

        Returns
        -------
        eigvals : array
            Array of eigenvalues for operator.

        eigvecs : array
            Array of quantum operators representing the oprator eigenkets.
            Order of eigenkets is determined by order of eigenvalues.

        Notes
        -----
        The sparse eigensolver is much slower than the dense version.
        Use sparse only if memory requirements demand it.

        """
        if isinstance(self.data, _data.CSR) and sparse:
            evals, evecs = _data.eigs_csr(self.data,
                                          isherm=self._isherm,
                                          sort=sort, eigvals=eigvals, tol=tol,
                                          maxiter=maxiter)
        elif isinstance(self.data, (_data.CSR, _data.Dia)):
            evals, evecs = _data.eigs(_data.to(_data.Dense, self.data),
                                      isherm=self._isherm,
                                      sort=sort, eigvals=eigvals)
        else:
            evals, evecs = _data.eigs(self.data, isherm=self._isherm,
                                      sort=sort, eigvals=eigvals)

        if self.type == 'super':
            new_dims = [self.dims[0], [1]]
        else:
            new_dims = [self.dims[0], [1]*len(self.dims[0])]
        ekets = np.empty((evecs.shape[1],), dtype=object)
        ekets[:] = [Qobj(vec, dims=new_dims, copy=False)
                    for vec in _data.split_columns(evecs, False)]
        norms = np.array([ket.norm() for ket in ekets])
        if phase_fix is None:
            phase = np.array([1] * len(ekets))
        else:
            phase = np.array([np.abs(ket[phase_fix, 0]) / ket[phase_fix, 0]
                              if ket[phase_fix, 0] else 1
                              for ket in ekets])
        return evals, ekets / norms * phase

    def eigenenergies(
        self,
        sparse: bool = False,
        sort: Literal["low", "high"] = 'low',
        eigvals: int = 0,
        tol: float = 0,
        maxiter: int = 100000,
    ) -> np.ndarray:
        """Eigenenergies of a quantum object.

        Eigenenergies (eigenvalues) are defined for operators or superoperators
        only.

        Parameters
        ----------
        sparse : bool
            Use sparse Eigensolver
        sort : str
            Sort eigenvalues 'low' to high, or 'high' to low.
        eigvals : int
            Number of requested eigenvalues. Default is all eigenvalues.
        tol : float
            Tolerance used by sparse Eigensolver (0=machine precision).
            The sparse solver may not converge if the tolerance is set too low.
        maxiter : int
            Maximum number of iterations performed by sparse solver (if used).

        Returns
        -------
        eigvals : array
            Array of eigenvalues for operator.

        Notes
        -----
        The sparse eigensolver is much slower than the dense version.
        Use sparse only if memory requirements demand it.

        """
        # TODO: consider another way of handling the dispatch here.
        if isinstance(self.data, _data.CSR) and sparse:
            return _data.eigs_csr(self.data,
                                  vecs=False,
                                  isherm=self._isherm,
                                  sort=sort, eigvals=eigvals,
                                  tol=tol, maxiter=maxiter)
        elif isinstance(self.data, (_data.CSR, _data.Dia)):
            return _data.eigs(_data.to(_data.Dense, self.data),
                              vecs=False, isherm=self._isherm,
                              sort=sort, eigvals=eigvals)

        return _data.eigs(self.data,
                          vecs=False,
                          isherm=self._isherm, sort=sort, eigvals=eigvals)

    def groundstate(
        self,
        sparse: bool = False,
        tol: float = 0,
        maxiter: int = 100000,
        safe: bool = True
    ) -> tuple[float, Qobj]:
        """Ground state Eigenvalue and Eigenvector.

        Defined for quantum operators or superoperators only.

        Parameters
        ----------
        sparse : bool
            Use sparse Eigensolver
        tol : float
            Tolerance used by sparse Eigensolver (0 = machine precision).
            The sparse solver may not converge if the tolerance is set too low.
        maxiter : int
            Maximum number of iterations performed by sparse solver (if used).
        safe : bool (default=True)
            Check for degenerate ground state

        Returns
        -------
        eigval : float
            Eigenvalue for the ground state of quantum operator.
        eigvec : :class:`.Qobj`
            Eigenket for the ground state of quantum operator.

        Notes
        -----
        The sparse eigensolver is much slower than the dense version.
        Use sparse only if memory requirements demand it.
        """
        eigvals = 2 if safe else 1
        evals, evecs = self.eigenstates(sparse=sparse, eigvals=eigvals,
                                        tol=tol, maxiter=maxiter)

        if safe:
            tol = tol or settings.core['atol']
            # This tol should be less strick than the tol for the eigensolver
            # so it's numerical errors are not seens as degenerate states.
            if (evals[1]-evals[0]) <= 10 * tol:
                warnings.warn("Ground state may be degenerate.", UserWarning)
        return evals[0], evecs[0]

    def purity(self) -> complex:
        """Calculate purity of a quantum object.

        Returns
        -------
        state_purity : float
            Returns the purity of a quantum object.
            For a pure state, the purity is 1.
            For a mixed state of dimension `d`, 1/d<=purity<1.

        """
        if self.type in ("super", "operator-ket", "operator-bra"):
            raise TypeError('purity is only defined for states.')
        if self.isket or self.isbra:
            return _data.norm.l2(self._data)**2
        return _data.trace(_data.matmul(self._data, self._data)).real

    def trunc_neg(self, method: Literal["clip", "sgs"] = "clip") -> Qobj:
        """Truncates negative eigenvalues and renormalizes.

        Returns a new Qobj by removing the negative eigenvalues
        of this instance, then renormalizing to obtain a valid density
        operator.

        Parameters
        ----------
        method : str
            Algorithm to use to remove negative eigenvalues. "clip"
            simply discards negative eigenvalues, then renormalizes.
            "sgs" uses the SGS algorithm (doi:10/bb76) to find the
            positive operator that is nearest in the Shatten 2-norm.

        Returns
        -------
        oper : :class:`.Qobj`
            A valid density operator.
        """
        if not self.isherm:
            raise ValueError("Must be a Hermitian operator to remove negative "
                             "eigenvalues.")
        if method not in ('clip', 'sgs'):
            raise ValueError("Method {} not recognized.".format(method))

        eigvals, eigstates = self.eigenstates()
        if all(eigval >= 0 for eigval in eigvals):
            # All positive, so just renormalize.
            return self.unit()
        idx_nonzero = eigvals != 0
        eigvals = eigvals[idx_nonzero]
        eigstates = eigstates[idx_nonzero]

        if method == 'clip':
            eigvals[eigvals < 0] = 0
        elif method == 'sgs':
            eigvals = eigvals[::-1]
            eigstates = eigstates[::-1]
            acc = 0.0
            n_eigs = len(eigvals)
            for idx in reversed(range(n_eigs)):
                if eigvals[idx] + acc / (idx + 1) >= 0:
                    break
                acc += eigvals[idx]
                eigvals[idx] = 0.0
            eigvals[:idx+1] += acc / (idx + 1)
        out_data = _data.zeros(*self.shape)
        for value, state in zip(eigvals, eigstates):
            if value:
                # add in 3-argument form is fused-add-multiply
                out_data = _data.add(out_data,
                                     _data.project(state.data),
                                     value)
        out_data = _data.mul(out_data, 1/_data.norm.trace(out_data))
        return Qobj(out_data, dims=self._dims, isherm=True, copy=False)
