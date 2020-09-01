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
"""
This function provides functions for parallel execution of loops and function
mappings, using the builtin Python module multiprocessing.
"""
__all__ = ['parallel_map', 'serial_map']

from scipy import array
import multiprocessing
from functools import partial
import os
import sys
import signal
from qutip.settings import settings as qset
from qutip.ui.progressbar import BaseProgressBar, TextProgressBar


if sys.platform == 'darwin':
    Pool = multiprocessing.get_context('fork').Pool
else:
    Pool = multiprocessing.Pool


def serial_map(task, values, task_args=tuple(), task_kwargs={}, **kwargs):
    """
    Serial mapping function with the same call signature as parallel_map, for
    easy switching between serial and parallel execution. This
    is functionally equivalent to::

        result = [task(value, *task_args, **task_kwargs) for value in values]

    This function work as a drop-in replacement of :func:`qutip.parallel_map`.

    Parameters
    ----------
    task : a Python function
        The function that is to be called for each value in ``task_vec``.
    values : array / list
        The list or array of values for which the ``task`` function is to be
        evaluated.
    task_args : list / dictionary
        The optional additional argument to the ``task`` function.
    task_kwargs : list / dictionary
        The optional additional keyword argument to the ``task`` function.
    progress_bar : ProgressBar
        Progress bar class instance for showing progress.

    Returns
    --------
    result : list
        The result list contains the value of
        ``task(value, *task_args, **task_kwargs)`` for each
        value in ``values``.

    """
    try:
        progress_bar = kwargs['progress_bar']
        if progress_bar is True:
            progress_bar = TextProgressBar()
    except:
        progress_bar = BaseProgressBar()

    progress_bar.start(len(values))
    results = []
    for n, value in enumerate(values):
        progress_bar.update(n)
        result = task(value, *task_args, **task_kwargs)
        results.append(result)
    progress_bar.finished()

    return results


def parallel_map(task, values, task_args=tuple(), task_kwargs={}, **kwargs):
    """
    Parallel execution of a mapping of `values` to the function `task`. This
    is functionally equivalent to::

        result = [task(value, *task_args, **task_kwargs) for value in values]

    Parameters
    ----------
    task : a Python function
        The function that is to be called for each value in ``task_vec``.
    values : array / list
        The list or array of values for which the ``task`` function is to be
        evaluated.
    task_args : list / dictionary
        The optional additional argument to the ``task`` function.
    task_kwargs : list / dictionary
        The optional additional keyword argument to the ``task`` function.
    progress_bar : ProgressBar
        Progress bar class instance for showing progress.

    Returns
    --------
    result : list
        The result list contains the value of
        ``task(value, *task_args, **task_kwargs)`` for
        each value in ``values``.

    """
    os.environ['QUTIP_IN_PARALLEL'] = 'TRUE'
    kw = _default_kwargs()
    if 'num_cpus' in kwargs:
        kw['num_cpus'] = kwargs['num_cpus']

    try:
        progress_bar = kwargs['progress_bar']
        if progress_bar is True:
            progress_bar = TextProgressBar()
    except:
        progress_bar = BaseProgressBar()

    progress_bar.start(len(values))
    nfinished = [0]

    def _update_progress_bar(x):
        nfinished[0] += 1
        progress_bar.update(nfinished[0])

    try:
        pool = Pool(processes=kw['num_cpus'])

        async_res = [pool.apply_async(task, (value,) + task_args, task_kwargs,
                                      _update_progress_bar)
                     for value in values]

        while not all([ar.ready() for ar in async_res]):
            for ar in async_res:
                ar.wait(timeout=0.1)

        pool.terminate()
        pool.join()

    except KeyboardInterrupt as e:
        os.environ['QUTIP_IN_PARALLEL'] = 'FALSE'
        pool.terminate()
        pool.join()
        raise e

    progress_bar.finished()
    os.environ['QUTIP_IN_PARALLEL'] = 'FALSE'
    return [ar.get() for ar in async_res]


def loky_pmap(task, values, task_args=tuple(), task_kwargs={}, **kwargs):
    """
    Parallel execution of a mapping of `values` to the function `task`. This
    is functionally equivalent to::

        result = [task(value, *task_args, **task_kwargs) for value in values]

    Parameters
    ----------
    task : a Python function
        The function that is to be called for each value in ``task_vec``.
    values : array / list
        The list or array of values for which the ``task`` function is to be
        evaluated.
    task_args : list / dictionary
        The optional additional argument to the ``task`` function.
    task_kwargs : list / dictionary
        The optional additional keyword argument to the ``task`` function.
    progress_bar : ProgressBar
        Progress bar class instance for showing progress.

    Returns
    --------
    result : list
        The result list contains the value of
        ``task(value, *task_args, **task_kwargs)`` for
        each value in ``values``.

    """
    os.environ['QUTIP_IN_PARALLEL'] = 'TRUE'
    from loky import get_reusable_executor, TimeoutError

    kw = _default_kwargs()
    if 'num_cpus' in kwargs:
        kw['num_cpus'] = kwargs['num_cpus']

    try:
        progress_bar = kwargs['progress_bar']
        if progress_bar is True:
            progress_bar = TextProgressBar()
    except:
        progress_bar = BaseProgressBar()

    progress_bar.start(len(values))

    executor = get_reusable_executor(max_workers=kw['num_cpus'])
    end_time = kw['timeout'] + time.time()
    job_time = kw['job_timeout']

    try:
        jobs = [executor.submit(task, value, *task_args, **task_kwargs)
               for value in values]

        results = []
        for job in jobs:
            remaining_time = end_time - time.time()
            result.append(job.result(min(remaining_time, job_time)))
            progress_bar.update()

    except (TimeoutError, KeyboardInterrupt) as e:
        [job.cancel() for job in jobs]
        executor.shutdown()
        raise e

    executor.shutdown()
    progress_bar.finished()
    os.environ['QUTIP_IN_PARALLEL'] = 'FALSE'
    return results

#TODO: move to options
def _default_kwargs():
    settings = {'num_cpus': multiprocessing.cpu_count(),
                'timeout':1e8,
                'job_timeout':1e8}
    return settings
