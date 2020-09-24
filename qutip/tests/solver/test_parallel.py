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
import time
import pytest

from qutip.solver.parallel import parallel_map, serial_map, loky_pmap
try:
    import loky
except ModuleNotFoundError:
    loky = False

def _func1(x):
    return x**2


def _func2(x, a, b, c, d=0, e=0, f=0):
    assert d > 0
    assert e > 0
    assert f > 0
    time.sleep(np.random.rand() * 0.1)  # random delay
    return x**2


@pytest.mark.skipif(not loky, reason="module loky not available")
def test_loki_map():
    "parallel_map"

    args = (1, 2, 3)
    kwargs = {'d': 4, 'e': 5, 'f': 6}

    x = np.arange(10)
    y1 = [_func1(xx) for xx in x]

    y2 = loky_pmap(_func2, x, args, kwargs, num_cpus=1)
    assert ((np.array(y1) == np.array(y2)).all())

    y2 = loky_pmap(_func2, x, args, kwargs, num_cpus=2)
    assert ((np.array(y1) == np.array(y2)).all())


def test_parallel_map():
    "parallel_map"

    args = (1, 2, 3)
    kwargs = {'d': 4, 'e': 5, 'f': 6}

    x = np.arange(10)
    y1 = [_func1(xx) for xx in x]

    y2 = parallel_map(_func2, x, args, kwargs, num_cpus=1)
    assert ((np.array(y1) == np.array(y2)).all())

    y2 = parallel_map(_func2, x, args, kwargs, num_cpus=2)
    assert ((np.array(y1) == np.array(y2)).all())


def test_serial_map():
    "serial_map"

    args = (1, 2, 3)
    kwargs = {'d': 4, 'e': 5, 'f': 6}

    x = np.arange(10)
    y1 = [_func1(xx) for xx in x]

    y2 = serial_map(_func2, x, args, kwargs, num_cpus=1)
    assert ((np.array(y1) == np.array(y2)).all())

    y2 = serial_map(_func2, x, args, kwargs, num_cpus=2)
    assert ((np.array(y1) == np.array(y2)).all())
