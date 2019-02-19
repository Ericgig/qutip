#!python
#cython: language_level=3
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
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
import numpy as np
cimport numpy as cnp
cimport cython
from libcpp.algorithm cimport sort
from libcpp.vector cimport vector
cnp.import_array()

include "parameters.pxi"

cdef extern from "numpy/arrayobject.h" nogil:
    void PyArray_ENABLEFLAGS(cnp.ndarray arr, int flags)
    void PyDataMem_FREE(void * ptr)
    void PyDataMem_RENEW(void * ptr, size_t size)
    void PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyDataMem_NEW(size_t size)

#Struct used for arg sorting
cdef struct _int_pair:
    int data
    int idx

cdef struct _long_pair:
    long data
    long idx

# Even if csr matrix are used here, these matrix do not represent Qobj:
# data is double instead of complex.
# Therefore we can't use the exising crs_matrix cdef class.
# Since these functions are used by only graph.py, the indptr, indices
# are kept in the python call. int32/int64 is supported through fused types.
ctypedef fused integer:
    int
    long

#qutip/cy/graph_utils.pyx:203:9: Compiler crash in AnalyseDeclarationsTransform
ctypedef fused uinteger:
    unsigned int
    unsigned long

ctypedef _int_pair int_pair
ctypedef int (*cfptr)(int_pair, int_pair)

ctypedef _long_pair long_pair
ctypedef long (*lcfptr)(long_pair, long_pair)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int int_sort(int_pair x, int_pair y):
    return x.data < y.data

@cython.boundscheck(False)
@cython.wraparound(False)
cdef long long_sort(long_pair x, long_pair y):
    return x.data < y.data


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int * int_argsort(int * x, int nrows):
    cdef vector[int_pair] pairs
    cdef cfptr cfptr_ = &int_sort
    cdef size_t kk
    pairs.resize(nrows)
    for kk in range(nrows):
        pairs[kk].data = x[kk]
        pairs[kk].idx = kk

    sort(pairs.begin(),pairs.end(),cfptr_)
    cdef int * out = <int *>PyDataMem_NEW(nrows *sizeof(int))
    for kk in range(nrows):
        out[kk] = pairs[kk].idx
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cdef long * long_argsort(long * x, long nrows):
    cdef vector[long_pair] pairs
    cdef lcfptr cfptr_ = &long_sort
    cdef size_t kk
    pairs.resize(nrows)
    for kk in range(nrows):
        pairs[kk].data = x[kk]
        pairs[kk].idx = kk

    sort(pairs.begin(),pairs.end(),cfptr_)
    cdef long * out = <long *>PyDataMem_NEW(nrows *sizeof(long))
    for kk in range(nrows):
        out[kk] = pairs[kk].idx
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef integer[::1] _node_degrees(integer[::1] ind,
                                 integer[::1] ptr,
                                 integer num_rows): #uinteger
    cdef size_t ii, jj
    cdef integer[::1] degree
    if integer is int:
        degree = np.zeros(num_rows, dtype=np.int32)
    else:
        degree = np.zeros(num_rows, dtype=np.int64)

    for ii in range(num_rows):
        degree[ii] = ptr[ii + 1] - ptr[ii]
        for jj in range(ptr[ii], ptr[ii + 1]):
            if ind[jj] == ii:
                # add one if the diagonal is in row ii
                degree[ii] += 1
                break

    return degree


@cython.boundscheck(False)
@cython.wraparound(False)
def _breadth_first_search(
        cnp.ndarray[integer, ndim=1, mode="c"] ind,
        cnp.ndarray[integer, ndim=1, mode="c"] ptr,
        integer num_rows, integer seed): #uinteger
    """
    Does a breath first search (BSF) of a graph in sparse CSR format matrix
    form starting at a given seed node.
    """

    cdef integer i, j, ii, jj, N = 1 #uinteger
    cdef integer level_start = 0 #uinteger
    cdef integer level_end = N #uinteger
    cdef integer current_level = 1 #uinteger
    cdef cnp.ndarray[integer] order
    cdef cnp.ndarray[integer] level

    if integer is int:
        order = -1 * np.ones(num_rows, dtype=np.int32)
        level = -1 * np.ones(num_rows, dtype=np.int32)
    else:
        order = -1 * np.ones(num_rows, dtype=np.int64)
        level = -1 * np.ones(num_rows, dtype=np.int64)

    level[seed] = 0
    order[0] = seed

    while level_start < level_end:
        # for nodes of the last level
        for ii in range(level_start, level_end):
            i = order[ii]
            # add unvisited neighbors to queue
            for jj in range(ptr[i], ptr[i + 1]):
                j = ind[jj]
                if level[j] == -1:
                    order[N] = j
                    level[j] = current_level
                    N += 1

        level_start = level_end
        level_end = N
        current_level += 1

    return order, level


@cython.boundscheck(False)
@cython.wraparound(False)
def _reverse_cuthill_mckee(integer[::1] ind,
                           integer[::1] ptr,
                           integer num_rows):
    """
    Reverse Cuthill-McKee ordering of a sparse csr or csc matrix.
    """
    cdef integer N = 0, N_old, seed, level_start, level_end  #uinteger
    cdef integer zz, i, j, ii, jj, kk, ll, level_len, temp, temp2, size  #uinteger
    cdef cnp.ndarray[integer, ndim=1] order
    cdef integer[::1] degree = _node_degrees(ind, ptr, num_rows)
    cdef integer * inds
    cdef integer * rev_inds
    cdef integer * temp_degrees = NULL

    if integer is int:
        order = np.zeros(num_rows, dtype=np.int32)
        inds = int_argsort(&degree[0], num_rows)
        rev_inds = int_argsort(inds, num_rows)
        size = sizeof(int)
    else:
        order = np.zeros(num_rows, dtype=np.int64)
        inds = long_argsort(&degree[0], num_rows)
        rev_inds = long_argsort(inds, num_rows)
        size = sizeof(long)

    # loop over zz takes into account possible disconnected graph.
    for zz in range(num_rows):
        if inds[zz] != -1:   # Do BFS with seed=inds[zz]
            seed = inds[zz]
            order[N] = seed
            N += 1
            inds[rev_inds[seed]] = -1
            level_start = N - 1
            level_end = N

            while level_start < level_end:
                for ii in range(level_start, level_end):
                    i = order[ii]
                    N_old = N

                    # add unvisited neighbors
                    for jj in range(ptr[i], ptr[i + 1]):
                        # j is node number connected to i
                        j = ind[jj]
                        if inds[rev_inds[j]] != -1:
                            inds[rev_inds[j]] = -1
                            order[N] = j
                            N += 1

                    # Add values to temp_degrees array for insertion sort
                    temp_degrees = <integer *>PyDataMem_RENEW(temp_degrees, (N-N_old)*size)
                    level_len = 0
                    for kk in range(N_old, N):
                        temp_degrees[level_len] = degree[order[kk]]
                        level_len += 1

                    # Do insertion sort for nodes from lowest to highest degree
                    for kk in range(1, level_len):
                        temp = temp_degrees[kk]
                        temp2 = order[N_old+kk]
                        ll = kk
                        while (ll > 0) and (temp < temp_degrees[ll-1]):
                            temp_degrees[ll] = temp_degrees[ll-1]
                            order[N_old+ll] = order[N_old+ll-1]
                            ll -= 1
                        temp_degrees[ll] = temp
                        order[N_old+ll] = temp2

                # set next level start and end ranges
                level_start = level_end
                level_end = N

        if N == num_rows:
            break
    PyDataMem_FREE(inds)
    PyDataMem_FREE(rev_inds)
    PyDataMem_FREE(temp_degrees)
    # return reversed order for RCM ordering
    return order[::-1]

"""
@cython.boundscheck(False)
@cython.wraparound(False)
def __pseudo_peripheral_node(
        cnp.ndarray[integer, ndim=1, mode="c"] ind,
        cnp.ndarray[integer, ndim=1, mode="c"] ptr,
        integer num_rows):
    \"""
    Find a pseudo peripheral node of a graph represented by a sparse
    csr_matrix.
    Never used in qutip and would fail.
    Left here for futur reference if ever useful/needed.
    \"""
    cdef unsigned int flag
    cdef integer ii, jj, delta, node, start
    cdef integer maxlevel, minlevel, minlastnodesdegree
    cdef cnp.ndarray[integer] lastnodes
    cdef cnp.ndarray[integer] lastnodesdegree
    cdef cnp.ndarray[integer] degree
    cdef integer[::1] tmp

    tmp = _node_degrees(ind, ptr, num_rows)
    if integer is int:
        degree = np.asarray(tmp, dtype=np.int32)
    else:
        degree = np.asarray(tmp, dtype=np.int64)

    start = 0
    delta = 0
    flag = 1

    while flag:
        # do a level-set traversal from x
        order, level = _breadth_first_search(ind, ptr, num_rows, start)

        # select node in last level with min degree
        maxlevel = max(level)
        lastnodes = np.where(level == maxlevel)[0] # This result in a scalar(int) not a vector
        lastnodesdegree = degree[lastnodes] # Then this also
        minlastnodesdegree = min(lastnodesdegree) # ...
        node = np.where(lastnodesdegree == minlastnodesdegree)[0][0]
        node = lastnodes[node]

        # if d(x,y) > delta, set, and do another BFS from this minimal node
        if level[node] > delta:
            start = node
            delta = level[node]
        else:
            flag = 0

    return start, order, level
"""

@cython.boundscheck(False)
@cython.wraparound(False)
def _maximum_bipartite_matching(
        cnp.ndarray[integer, ndim=1, mode="c"] inds,
        cnp.ndarray[integer, ndim=1, mode="c"] ptrs,
        integer n):

    cdef cnp.ndarray[integer] visited
    cdef cnp.ndarray[integer] queue
    cdef cnp.ndarray[integer] previous
    cdef cnp.ndarray[integer] match
    cdef cnp.ndarray[integer] row_match
    cdef integer queue_ptr, queue_col, ptr, i, j, queue_size
    cdef integer row, col, temp, eptr, next_num = 1

    if integer is int:
      visited = np.zeros(n, dtype=np.int32)
      queue = np.zeros(n, dtype=np.int32)
      previous = np.zeros(n, dtype=np.int32)
      match = -1 * np.ones(n, dtype=np.int32)
      row_match = -1 * np.ones(n, dtype=np.int32)
    else:
      visited = np.zeros(n, dtype=np.int64)
      queue = np.zeros(n, dtype=np.int64)
      previous = np.zeros(n, dtype=np.int64)
      match = -1 * np.ones(n, dtype=np.int64)
      row_match = -1 * np.ones(n, dtype=np.int64)

    for i in range(n):
        if match[i] == -1 and (ptrs[i] != ptrs[i + 1]):
            queue[0] = i
            queue_ptr = 0
            queue_size = 1
            while (queue_size > queue_ptr):
                queue_col = queue[queue_ptr]
                queue_ptr += 1
                eptr = ptrs[queue_col + 1]
                for ptr in range(ptrs[queue_col], eptr):
                    row = inds[ptr]
                    temp = visited[row]
                    if (temp != next_num and temp != -1):
                        previous[row] = queue_col
                        visited[row] = next_num
                        col = row_match[row]
                        if (col == -1):
                            while (row != -1):
                                col = previous[row]
                                temp = match[col]
                                match[col] = row
                                row_match[row] = col
                                row = temp
                            next_num += 1
                            queue_size = 0
                            break
                        else:
                            queue[queue_size] = col
                            queue_size += 1

            if match[i] == -1:
                for j in range(1, queue_size):
                    visited[match[queue[j]]] = -1

    return match


@cython.boundscheck(False)
@cython.wraparound(False)
def _max_row_weights(
        double[::1] data,
        integer[::1] inds,
        integer[::1] ptrs,
        integer ncols):
    """
    Finds the largest abs value in each matrix column
    and the max. total number of elements in the cols (given by weights[-1]).

    Here we assume that the user already took the ABS value of the data.
    This keeps us from having to call abs over and over.

    """
    cdef cnp.ndarray[double] weights = np.zeros(ncols + 1, dtype=np.double)
    cdef integer ln, mx, ii, jj
    cdef double weight, current

    mx = 0
    for jj in range(ncols):
        ln = (ptrs[jj + 1] - ptrs[jj])
        if ln > mx:
            mx = ln

        weight = data[ptrs[jj]]
        for ii in range(ptrs[jj] + 1, ptrs[jj + 1]):
            current = data[ii]
            if current > weight:
                weight = current

        weights[jj] = weight

    weights[ncols] = mx
    return weights


@cython.boundscheck(False)
@cython.wraparound(False)
def _weighted_bipartite_matching(
        double[::1] data,
        integer[::1] inds,
        integer[::1] ptrs,
        integer n):
    """
    Here we assume that the user already took the ABS value of the data.
    This keeps us from having to call abs over and over.
    """
    cdef cnp.ndarray[integer] visited
    cdef cnp.ndarray[integer] queue
    cdef cnp.ndarray[integer] previous
    cdef cnp.ndarray[integer] match
    cdef cnp.ndarray[integer] row_match
    cdef cnp.ndarray[double] weights = _max_row_weights(data, inds, ptrs, n)
    cdef cnp.ndarray[integer] order
    cdef cnp.ndarray[integer] row_order
    cdef cnp.ndarray[double] temp_weights = np.zeros(int(weights[n]), dtype=np.double)
    cdef integer queue_ptr, queue_col, queue_size, next_num
    cdef integer i, j, zz, ll, kk, row, col, temp, eptr, temp2
    if integer is int:
      visited = np.zeros(n, dtype=np.int32)
      queue = np.zeros(n, dtype=np.int32)
      previous = np.zeros(n, dtype=np.int32)
      match = -1 * np.ones(n, dtype=np.int32)
      row_match = -1 * np.ones(n, dtype=np.int32)
      order = np.argsort(-weights[0:n]).astype(np.int32)
      row_order = np.zeros(int(weights[n]), dtype=np.int32)
    else:
      visited = np.zeros(n, dtype=np.int64)
      queue = np.zeros(n, dtype=np.int64)
      previous = np.zeros(n, dtype=np.int64)
      match = -1 * np.ones(n, dtype=np.int64)
      row_match = -1 * np.ones(n, dtype=np.int64)
      order = np.argsort(-weights[0:n]).astype(np.int64)
      row_order = np.zeros(int(weights[n]), dtype=np.int64)

    next_num = 1
    for i in range(n):
        zz = order[i]  # cols with largest abs values first
        if (match[zz] == -1 and (ptrs[zz] != ptrs[zz + 1])):
            queue[0] = zz
            queue_ptr = 0
            queue_size = 1

            while (queue_size > queue_ptr):
                queue_col = queue[queue_ptr]
                queue_ptr += 1
                eptr = ptrs[queue_col + 1]

                # get row inds in current column
                temp = ptrs[queue_col]
                for kk in range(eptr - ptrs[queue_col]):
                    row_order[kk] = inds[temp]
                    temp_weights[kk] = data[temp]
                    temp += 1

                # linear sort by row weight
                for kk in range(1, (eptr - ptrs[queue_col])):
                    val = temp_weights[kk]
                    row_val = row_order[kk]
                    ll = kk - 1
                    while (ll >= 0) and (temp_weights[ll] > val):
                        temp_weights[ll + 1] = temp_weights[ll]
                        row_order[ll + 1] = row_order[ll]
                        ll -= 1

                    temp_weights[ll + 1] = val
                    row_order[ll + 1] = row_val

                # go through rows by decending weight
                temp2 = (eptr - ptrs[queue_col]) - 1
                for kk in range(eptr - ptrs[queue_col]):
                    row = row_order[temp2 - kk]
                    temp = visited[row]
                    if temp != next_num and temp != -1:
                        previous[row] = queue_col
                        visited[row] = next_num
                        col = row_match[row]
                        if col == -1:
                            while row != -1:
                                col = previous[row]
                                temp = match[col]
                                match[col] = row
                                row_match[row] = col
                                row = temp

                            next_num += 1
                            queue_size = 0
                            break
                        else:
                            queue[queue_size] = col
                            queue_size += 1

            if match[zz] == -1:
                for j in range(1, queue_size):
                    visited[match[queue[j]]] = -1

    return match
