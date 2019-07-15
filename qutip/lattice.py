# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, The QuTiP Project.
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
__all__ = ['Lattice1d','Lattice2d','create_cell_Hamiltonian']
from matplotlib.pyplot import * # remove
import matplotlib.pyplot as plt # keep this

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import # remove

from matplotlib import cm  # remove
from matplotlib.ticker import LinearLocator, FormatStrFormatter # remove
import sympy

import warnings # remove
import types    # remove

try:
    import builtins
except:
    import __builtin__ as builtins

# import math functions from numpy.math: required for td string evaluation
from numpy import (arccos, arccosh, arcsin, arcsinh, arctan, arctan2, arctanh,
                   ceil, copysign, cos, cosh, degrees, e, exp, expm1, fabs,
                   floor, fmod, frexp, hypot, isinf, isnan, ldexp, log, log10,
                   log1p, modf, pi, radians, sin, sinh, sqrt, tan, tanh, trunc)# remove
from scipy.sparse import (_sparsetools, isspmatrix, isspmatrix_csr,
                          csr_matrix, coo_matrix, csc_matrix, dia_matrix)
from qutip.fastsparse import fast_csr_matrix, fast_identity # remove
from qutip.qobj import Qobj
from qutip.qobj import isherm # remove
from qutip import * # remove
import numpy as np
from scipy.sparse.linalg import eigs


def Hamiltonian_2d(base_h, inter_hop_x, inter_hop_y, space_dims,
                   nx_units=1, PBCx=0, ny_units=1, PBCy=0):
    """
    Returns the Hamiltonian as a csr_matrix from the specified parameters for a
    2d space of sites in the x-major format.

    Parameters
    ==========
    base_h : numpy matrix
        The matrix to be diagonalized
    inter_hop_x : numpy matrix
        The matrix to be diagonalized
    inter_hop_y : numpy matrix
        The matrix to be diagonalized
    dims : numpy matrix
        The matrix to be diagonalized
    nx_units : numpy matrix
        The matrix to be diagonalized
    PBCx : numpy matrix
        The matrix to be diagonalized
    ny_units : numpy matrix
        The matrix to be diagonalized
    PBCy : numpy matrix
        The matrix to be diagonalized

    Returns
    -------
    Hamt : csr_matrix
        The 2d Hamiltonian matrix for the specified parameters.
    """
    # Why not use tensor?
    xi_len = len(inter_hop_x) # inter_hop_x is a list but not inter_hop_y
    (x0,y0) = np.shape(inter_hop_x[0]) # Why use np.shape? inter_hop_x is not a Qobj?

    (x1,y1) = np.shape(inter_hop_y) # Same here?
    (xx,yy) = np.shape(base_h) # Same here?

    row_ind = np.array([]); col_ind = np.array([]);  data = np.array([]);
    # here using list instead of array would be better

    NS = space_dims[0][0]*space_dims[1][0]

    for i in range(nx_units):
        for j in range(ny_units):
            lin_RI = i + j* nx_units
            for k in range(xx):
                for l in range(yy):
                    row_ind = np.append(row_ind,[lin_RI*NS+k])
                    col_ind = np.append(col_ind,[lin_RI*NS+l])
                    data = np.append(data,[base_h[k,l] ])

    for i in range(nx_units):
        for j in range(ny_units):
            lin_RI = i + j* nx_units;
            for m in range(xi_len):
                for k in range(x0):
                    for l in range(y0): # 5 loops ... not great in python
                        if (i>0):
                            row_ind = np.append(row_ind, [lin_RI*NS+k])
                            col_ind = np.append(col_ind, [(lin_RI-1)*NS+l])
                            data = np.append(data, [np.conj(inter_hop_x[m][l,k])])

                for k in range(x0):
                    for l in range(y0):
                        if (i < (nx_units-1)):
                            row_ind = np.append(row_ind,[lin_RI*NS+k])
                            col_ind = np.append(col_ind,[(lin_RI+1)*NS+l])
                            data = np.append(data,[inter_hop_x[m][k,l] ])

            for k in range(x1):
                for l in range(y1):
                    if (j>0):
                        row_ind = np.append(row_ind,[lin_RI*NS+k])
                        col_ind = np.append(col_ind,[(lin_RI-nx_units)*NS+l])
                        data = np.append(data,[np.conj(inter_hop_y[l,k]) ])

            for k in range(x1):
                for l in range(y1):
                    if (j<(ny_units-1)):
                        row_ind = np.append(row_ind,[lin_RI*NS+k])
                        col_ind = np.append(col_ind,[(lin_RI+nx_units)*NS+l])
                        data = np.append(data,[inter_hop_y[k,l] ])

    M = nx_units*ny_units*NS # rename with more descriptive name
    N = nx_units*ny_units*NS # M==N no need for 2 variable

    for i in range(nx_units):
        lin_RI = i;
        if (PBCy == 1 and ny_units*space_dims[1][0] > 2):
            for k in range(x1):
                for l in range(y1):
                    (Aw,) = np.where(row_ind == (lin_RI*NS+k))
                    (Bw,) = np.where(col_ind == ((lin_RI+(ny_units-1)*nx_units)*NS+l))
                    if ( len(np.intersect1d(Aw,Bw)  ) == 0 ):
                        row_ind = np.append(row_ind, [lin_RI*NS+k, (lin_RI+(ny_units-1)*nx_units)*NS+l])
                        col_ind = np.append(col_ind, [(lin_RI+(ny_units-1)*nx_units)*NS+l, lin_RI*NS+k])
                        data = np.append(data,[np.conj(inter_hop_y[l,k]), inter_hop_y[l,k] ])

    for j in range(ny_units):
        lin_RI = j;
        if (PBCx == 1 and nx_units*space_dims[0][0] > 2):
            for k in range(x0):
                for l in range(y0):
                    (Aw,) = np.where(row_ind == (lin_RI*nx_units*NS+k))
                    (Bw,) = np.where(col_ind == (((lin_RI+1)*nx_units-1)*NS+l))
                    if ( len(np.intersect1d(Aw,Bw)) == 0 ):
                        for m in range(xi_len):
                            row_ind = np.append(row_ind, [lin_RI*nx_units*NS+k, ((lin_RI+1)*nx_units-1)*NS+l])
                            col_ind = np.append(col_ind, [((lin_RI+1)*nx_units-1)*NS+l, lin_RI*nx_units*NS+k])
                            data = np.append(data, [np.conj(inter_hop_x[m][l,k]), inter_hop_x[m][l,k]])

    Hamt = csr_matrix((data, (row_ind, col_ind)), [M, N], dtype=np.complex )
    return Hamt # please return a Qobj, not a scipy matrix

def matrix_sparcity_pattern(Hamt):
    # move to visualisation.py
    """
    Plots the non-zero elements of a matrix on a 2d plane.

    Parameters
    ==========
    Hamt : numpy matrix
        The matrix to be plotted on the 2d plane.
    """
    Hamt = csr_matrix(Hamt)

    fig, axs = plt.subplots(1, 1)
    axs.spy(Hamt, markersize=5)
    plt.show()

def diag_a_matrix( H_k, calc_evecs = False):
    """
    Returns eigen-values and/or eigen-vectors of an input matrix.

    Parameters
    ==========
    H_k : numpy matrix
        The matrix to be diagonalized

    Returns
    -------
    vecs : numpy ndarray
        vecs[:,:] = [band index,:] = eigen-vector of band_index
    vals : numpy ndarray
        The diagonalized matrix
    """
    if np.max(H_k-H_k.T.conj())>1.0E-20: #may be too stric 1e-16
        raise Exception("\n\nThe Hamiltonian matrix is not hermitian?!") # "The Hamiltonian need to be hermitian."

    if calc_evecs == False: # calculate only the eigenvalues
        vals=np.linalg.eigvalsh(H_k.todense())
        return np.array(vals,dtype=float)
    else: # find eigenvalues and eigenvectors
        (vals, vecs)=np.linalg.eigh(H_k.todense())
        vecs=vecs.T
        # now vecs[i,:] is eigenvector for vals[i,i]-th eigenvalue
        return (vals,vecs)


def create_cell_Hamiltonian( val_s, val_q):
    """
    Returns eigen-values and/or eigen-vectors of an input matrix.

    Parameters
    ==========
    H_k : numpy matrix
        The matrix to be diagonalized

    Returns
    -------
    vecs : numpy ndarray
        vecs[:,:] = [band index,:] = eigen-vector of band_index
    vals : numpy ndarray
        The diagonalized matrix
    """
    # be carefull about trailing whitespace

    QN = len(val_q)
    SN = len(val_s)
    k_Hamiltonian = [ [ [] for i in range(SN*QN)] for j in range(SN*QN) ]

    for ir in range(SN):
        for jr in range(QN):
            for ic in range(SN):
                for jc in range(QN):
                    # sympy is used at not much more than string here, not needed.
                    k_Hamiltonian[ir*QN+jr][ic*QN+jc] = sympy.Symbol("<"+val_s[ir]+val_q[jr]+" H "+val_s[ic]+val_q[jc] + ">" )

    return k_Hamiltonian
# One empty line in function and class, 2 in between


class Lattice1d():
    """A class for representing ...

    Parameters
    ----------

    Attributes
    ----------

    Methods
    -------

    """
    def __init__(self, num_cell=10, boundary = "periodic", cell_num_site = 1,
                 cell_site_dof = 1, cell_Hamiltonian = Qobj([[0]]) ,
                 inter_hop = Qobj([[-1]]), inter_vec_list = [[]] ):

        self.cell_num_site = cell_num_site
        self.cell_site_dof = cell_site_dof

        self.cell_tensor_config = np.append(cell_num_site,cell_site_dof)
        self._num_cell = num_cell
        l_u = 1
        for i in range(len(self.cell_tensor_config)):
            l_u = l_u*self.cell_tensor_config[i]
        self._length_of_unit_cell = l_u

        if ( type(cell_num_site) != int or cell_num_site < 0 ):
            raise Exception("\n\n cell_num_site is required to be a positive integer.") # Why start with two \n?

        if ( type(cell_site_dof) == list  ):
            for i in range(len(cell_site_dof)):
                if ( type(cell_site_dof[i]) != int or cell_site_dof[i] < 0 ): # isinstance(cell_site_dof[i], int) instead of type() == int
                    print('\n Unacceptable cell_site_dof list element at index: ',i)
                    raise Exception("\n\n Elements of cell_site_dof is required to be positive integers.") # Why start with two \n?
        elif ( type(cell_site_dof) == int  ):
            if ( cell_site_dof < 0 ):
                raise Exception("\n\n cell_site_dof is required to be a positive integer.") # Why start with two \n?
        else:
            raise Exception("\n\n cell_site_dof is required to be a positive integer or a list of positive integers.") # Why start with two \n?

        self._H_intra = cell_Hamiltonian
        if (type(inter_hop) == list):
            for i in range(len(inter_hop) ):
                if (type(inter_hop[i]) != qutip.qobj.Qobj ):
                    raise Exception("\n\nAll inter_hop list elements need to be Qobj's.") # Why start with two \n?
            self._H_inter_list = inter_hop
        else:
            if (type(inter_hop) != qutip.qobj.Qobj ):
                raise Exception("\n\n inter_hop need to be a Qobj.") # Why start with two \n?
            self._H_inter_list = [inter_hop]

        if ( len(inter_vec_list) == 0 ):
            self._H_inter_vector_list = [[1] for i in range(len(self._H_inter_list)) ] # Why list of list?
            # The default _H_inter_vector_list is only [1]s.
        else:
            self._H_inter_vector_list = inter_vec_list
            # The _H_inter_vector_list is set to user defined values.

        self._is_consistent = self._checks(check_if_consistent = True)
        self._lattice_vectors_list = [[1]]     #unit vectors

        if boundary == "periodic":
            self.PBCx = 1;
        elif (boundary == "aperiodic" or boundary == "hardwall" ):
            self.PBCx = 0;
        else:
            print("Error in boundary")
            raise Exception("Only recognized bounday options are:\"periodic\","
                            " \"aperiodic\" and \"hardwall\" ");

    def _checks(self,check_if_consistent = False):
        """
        A checking code that confirms all the entries in the Qbasis is
        consistent (code to be written).  The codenames indicate the last
        action, that we would verify the correctness of. If it is called with
        check_if_complete == True, it returns if the Qbasis instance is
        complete or not.
        """
        raise NotImplementedError

    def basis(self, cell = 0, ind = np.array([[0],[0]]) ):
        """
        A checking code that confirms all the entries in the Qbasis is
        """
        vec_i = tensor(basis(self._num_cell,cell),
                       basis(self.unit_tensor_dimensions[0][0], ind[0][0]),
                       basis(self.unit_tensor_dimensions[1][0], ind[1][0]))
        # broken? self.cell_num_site, self.cell_site_dof
        return vec_i

    def distribute_operator(self, op  ):
        """
        A checking code that confirms all the entries in the Qbasis is
        """
        (xx,yy) = np.shape(op)

        row_ind = np.array([])
        col_ind = np.array([])
        data = np.array([])

        NS = self._length_of_unit_cell
        nx_units = self._num_cell
        # ny_units = 1 # We don't need ny_units in 1D.
        for i in range(nx_units):
            #for j in range(ny_units):
                lin_RI = i #+ j* nx_units;
                for k in range(xx):
                    for l in range(yy):
                        row_ind = np.append(row_ind,[lin_RI*NS+k]);
                        col_ind = np.append(col_ind,[lin_RI*NS+l]);
                        data = np.append(data,[op[k,l] ]);

        op_H = csr_matrix((data, (row_ind, col_ind)),
                          [nx_units*NS, nx_units*NS], dtype=np.complex )
        return Qobj(op_H)

    def x(self):
        """
        The position operator.
        """
        Xar = num(self._num_cell) # not used, remove

        nx = self.unit_tensor_dimensions[0][0]
        ne = self.unit_tensor_dimensions[1][0]
        # broken? self.cell_num_site, self.cell_site_dof

        positions = [ i for i in range(nx) for _ in range(ne) ] # in python _ is used for dummy iterator
        R = np.kron(range(0, self._num_cell), [1 for i in range(nx*ne)])
        # range(0, self._num_cell) => range(self._num_cell)
        # [1 for _ in range(nx*ne) ] => np.ones(nx*ne)
        S = np.kron([1 for i in range(self._num_cell) ], positions)
        # [1 for _ in range(self._num_cell) ] => np.ones(self._num_cell)

        # print(R)
        # print(S)
        # S is the positions inside a cell, so no values should be over 1?
        xs = np.diagflat(R+S)
        return Qobj(xs)


    def operator(self, op, cell ):
        """
        """
        (xx,yy) = np.shape(op)
        row_ind = np.array([])
        col_ind = np.array([])
        data = np.array([])

        NS = self._length_of_unit_cell
        nx_units = self._num_cell
        # ny_units = 1 # not needed, remove

        for i in range(nx_units):
            #for j in range(ny_units):
                lin_RI = i # + j* nx_units;

                if i == cell:
                    for k in range(xx):
                        for l in range(yy):
                            row_ind = np.append(row_ind,[lin_RI*NS+k])
                            col_ind = np.append(col_ind,[lin_RI*NS+l])
                            data = np.append(data,[op[k,l] ])

                else:
                    for k in range(xx):
                        row_ind = np.append(row_ind,[lin_RI*NS+k]);
                        col_ind = np.append(col_ind,[lin_RI*NS+k]);
                        data = np.append(data,[1]);

        M = nx_units*NS # I letter capital name are not good variable name, lat_size?

        op_H = csr_matrix((data, (row_ind, col_ind)), [M, M], dtype=np.complex)
        return Qobj(op_H)


    def operator_set_cells(self, op ):
        """
        A checking code that confirms all the entries in the Qbasis is
        """ # This is not the right docstring
        op_set = []
        for i in range(self._num_cell):
            op_set.append(self.operator(op,i) )

        return op_set

    def operator_cross_site(self, op, site):
        """
        site = [m,n]
        """
        raise NotImplementedError

    def Hamiltonian(self):
        """
        Hamiltonian.
        """
        # self.dims = [[number_of_atoms],[1]]
        # Hamil = Hamiltonian_2d(self._H_intra, self._inter_hop_x,
        #                        self._inter_hop_y, dims=self._space_dims,
        #                        nx_units=self._x_num_cell, PBCx=self.PBCx,
        #                        ny_units=self._y_num_cell, PBCy=self.PBCy)
        Hamil = Hamiltonian_2d(self._H_intra, self._H_inter_list,
                               self._H_inter_list[0], space_dims=self._space_dims,
                               nx_units=self._num_cell, PBCx=self.PBCx)
        return Qobj(Hamil)
        # Hamil's dims=[[nx, num_site, dof], [nx, num_site, dof]]

    def dispersion(self, to_display=0, klist=[-np.pi, np.pi, 101]):
        """
        Calculates the dispersion for a 1-dimensional crystal
        """
        # if (not self.lattice_vectors.is_complete):
        #     raise Exception("\n\n Lattice vector is not defined.")

        # if (not self.hop_vectors.is_complete):
        #     raise Exception("\n\n H_inter_vec_dict/list defintions to be "
        #                     "accompanied with vectors associate with them.")

        # klist default should depend on the lattice number of sites, not a arbitrary 101
        if ( type(klist) != list or  np.shape(klist) != (3,) ):
            print(' The second argument of dispersion() needs to be a list in the')
            print('format [float,float,int]. For example [-numpy.pi,numpy.pi,101]')
            if (type(klist[2]) != int):
                print('klist[2] needs to be an int')
            raise Exception("\n\nUnacceptable arguments for Lattice1d.dispersion()")

        k_start = klist[0]
        k_end = klist[1]
        kpoints = klist[2]
        NS = self._length_of_unit_cell
        val_ks = np.zeros((NS, kpoints), dtype=float)
        kxA = np.zeros((kpoints, 1), dtype=float)
        G0_H = self._H_intra

        k_1 = kpoints - 1
        for ks in range(kpoints):
            kx = k_start + (ks * (k_end - k_start) / k_1)
            kxA[ks,0] = kx

            H_ka = G0_H
            k_cos = [[kx]]
            for m in range(len(self._H_inter_vector_list)):
                r_cos = self._H_inter_vector_list[m]
                kr_dotted = np.dot(k_cos, r_cos)
                H_int = self._H_inter_list[m] * exp(complex(0, kr_dotted))
                H_ka = H_ka + H_int + H_int.dag()

            H_k = csr_matrix(H_ka)
            vals = diag_a_matrix(H_k, calc_evecs=False)
            val_ks[:,ks] = vals[:]

        if to_display == 1:
            # To move to another method.
            fig, ax = subplots()
            for g in range(NS):
                ax.plot(kxA/pi, val_ks[g,:])
            ax.set_ylabel('Energy')
            ax.set_xlabel('$k_x(\pi)$')
            show(fig)
            fig.savefig('./Dispersion.pdf')

        return (kxA, val_ks)


class Lattice2d():
    def __init__(self, x_num_cell=1, y_num_cell=1 , boundary = ("periodic","periodic"), basis_Hamiltonian = Qobj([[0]]), dims = [[1],[1]] , inter_hop_x= Qobj([[-1]]), inter_hop_y = Qobj([[-1]])  ):

        self.dims = dims
        self._inter_hop_x = inter_hop_x
        self._inter_hop_y = inter_hop_y

        self._H_intra = basis_Hamiltonian
        self._x_num_cell = x_num_cell
        self._y_num_cell = y_num_cell
        self._lattice_vectors_list = [[1,0],[0,1]]   #default basis vector

        self._is_complete = self._checks(check_if_consistent = True)

        if ( not isinstance(boundary,tuple)  ):
            print("The specified format for bounday needs to be a tuple of two str's.")
            raise Exception("For example:  boundary = (\"periodic\",\"periodic\") ");

        if boundary[0] == "periodic":
            self.PBCx = 1
        elif (boundary[0] == "aperiodic" or boundary[0] == "hardwall" ):
            self.PBCx = 0
        else:
            print("Error in boundary[0]")
            raise Exception(" Only recognized bounday options are:\"periodic\",\"aperiodic\" and \"hardwall\" ");

        if boundary[1] == "periodic":
            self.PBCy = 1
        elif (boundary[1] == "aperiodic" or boundary[1] == "hardwall" ):
            self.PBCy = 0
        else:
            print("Error in boundary[1]")
            raise Exception(" Only recognized bounday options are:\"periodic\",\"aperiodic\" and \"hardwall\" ")


    def _checks(self,check_if_consistent = False):
        """
        A checking code that confirms all the entries in the Qbasis is
        consistent (code to be written).  The codenames indicate the last
        action, that we would verify the correctness of. If it is called with
        check_if_complete == True, it returns if the Qbasis instance is
        complete or not.
        """
        A = True
        return A

    def Hamiltonian(self):
        """
        Hamiltonian.
        """
        # Hamiltonian_2d(base_h, inter_hop_x, inter_hop_y, dims, nx_units = 1, PBCx = 0, ny_units = 1, PBCy = 0)
        Hamil = Hamiltonian_2d(self._H_intra, self._inter_hop_x, self._inter_hop_y, dims = self.dims, nx_units = self._x_num_cell, PBCx = self.PBCx, ny_units = self._y_num_cell, PBCy = self.PBCy)
        return Qobj(Hamil)

    def set_basis_vectors(self, word_key, user_input_vectors ):
        """
        Calculates the dispersion for a 2-dimensional crystal
        """
        if (word_key == 'graphene'):
            self._lattice_vectors_list = [ [np.sqrt(3)/2,1/2],[np.sqrt(3)/2,-1/2] ]
        elif (word_key == 'kagome'):
            self._lattice_vectors_list = [ [1,0],[1/2,-np.sqrt(3)/2] ]
        elif (word_key == 'user'):
            if (not isinstance(user_input_vectors) ):
                raise Exception("user_input_vectors need to be a list of lists.")
            if ((not isinstance(user_input_vectors[0]) ) or (not isinstance(user_input_vectors[1]) ) ):
                raise Exception("user_input_vectors[0] and user_input_vectors[1] need to be lists of two numbers")
            x1 = user_input_vectors[0][0]
            y1 = user_input_vectors[0][1]
            x2 = user_input_vectors[1][0]
            y2 = user_input_vectors[1][1]
            if (np.abs(x1*x1+y1*y1-1) > 1E-10):
                raise Exception("The first input basis vector is not a unit vector. ")

            if (np.abs(x2*x2+y2*y2-1) > 1E-10):
                raise Exception("The second input basis vector is not a unit vector. ")

            self._lattice_vectors_list = user_input_vectors
        else:
            raise Exception("Unrecognized lattice: Current options are: graphene, kagome, and user.")



    def dispersion(self, to_display = 0, klist = [-np.pi,np.pi,101]):
        """
        Calculates the dispersion for a 2-dimensional crystal
        """
        if (not self.lattice_vectors.is_complete):
            raise Exception("\n\n Lattice vector is not defined.")

        if (not self.hop_vectors.is_complete):
            raise Exception("\n\n H_inter_vec_dict/list defintions to be accompanied with vectors associate with them.")
