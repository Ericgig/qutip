"""
This module provides functions for parallel execution of loops and function
mappings, using the builtin Python module multiprocessing or the loky parallel execution library.
"""
__all__ = ['parallel_map', 'serial_map', 'loky_pmap', 'get_map']

import multiprocessing
import os
import sys
import time
import threading
import concurrent.futures
from qutip.ui.progressbar import progess_bars
from qutip.settings import available_cpu_count

if sys.platform == 'darwin':
    mp_context = multiprocessing.get_context('fork')
elif sys.platform == 'linux':
    # forkserver would handle threads better, but is much slower at starting
    # the executor and spawning tasks
    mp_context = multiprocessing.get_context('fork')
else:
    mp_context = multiprocessing.get_context()


default_map_kw = {
    'job_timeout': threading.TIMEOUT_MAX,
    'timeout': threading.TIMEOUT_MAX,
    'num_cpus': available_cpu_count(),
}


def _read_map_kw(options):
    options = options or {}
    map_kw = default_map_kw.copy()
    map_kw.update({k: v for k, v in options.items() if v is not None})
    return map_kw


def serial_map(task, values, task_args=None, task_kwargs=None,
               reduce_func=None, map_kw=None,
               progress_bar=None, progress_bar_kwargs={}):
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
    reduce_func : func (optional)
        If provided, it will be called with the output of each tasks instead of
        storing a them in a list.
    progress_bar : string
        Progress bar options's string for showing progress.
    progress_bar_kwargs : dict
        Options for the progress bar.
    map_kw: dict (optional)
        Dictionary containing entry for 'timeout' the maximum time for the
        whole map.

    Returns
    --------
    result : list
        The result list contains the value of
        ``task(value, *task_args, **task_kwargs)`` for each
        value in ``values``. If a ``reduce_func`` is provided, and empty list
        will be returned.

    """
    if task_args is None:
        task_args = ()
    if task_kwargs is None:
        task_kwargs = {}
    map_kw = _read_map_kw(map_kw)
    progress_bar = progess_bars[progress_bar]()
    progress_bar.start(len(values), **progress_bar_kwargs)
    end_time = map_kw['timeout'] + time.time()
    results = []
    for n, value in enumerate(values):
        if time.time() > end_time:
            break
        progress_bar.update(n)
        result = task(value, *task_args, **task_kwargs)
        if reduce_func is not None:
            reduce_func(result)
        else:
            results.append(result)
    progress_bar.finished()

    return results


def parallel_map(task, values, task_args=None, task_kwargs=None,
                 reduce_func=None, map_kw=None,
                 progress_bar=None, progress_bar_kwargs={}):
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
    reduce_func : func (optional)
        If provided, it will be called with the output of each tasks instead of
        storing a them in a list. Note that the order in which results are
        passed to ``reduce_func`` is not defined.
    progress_bar : string
        Progress bar options's string for showing progress.
    progress_bar_kwargs : dict
        Options for the progress bar.
    map_kw: dict (optional)
        Dictionary containing entry for:
        'timeout': Maximum time for the whole map.
        'job_timeout': Maximum time for each job in the map.
        'num_cpus': Number of job to run at once.

    Returns
    --------
    result : list
        The result list contains the value of
        ``task(value, *task_args, **task_kwargs)`` for
        each value in ``values``. If a ``reduce_func`` is provided, and empty
        list will be returned.

    """
    if task_args is None:
        task_args = ()
    if task_kwargs is None:
        task_kwargs = {}
    map_kw = _read_map_kw(map_kw)
    end_time = map_kw['timeout'] + time.time()
    job_time = map_kw['job_timeout']

    progress_bar = progess_bars[progress_bar]()
    progress_bar.start(len(values), **progress_bar_kwargs)

    if reduce_func is not None:
        results = None
        result_func = lambda i, value: reduce_func(value)
    else:
        results = [None] * len(values)
        result_func = lambda i, value: results.__setitem__(i, value)

    def _done_callback(future):
        if not future.cancelled():
            result_func(future._i, future.result())
        progress_bar.update()

    if sys.version_info >= (3, 7):
        # ProcessPoolExecutor only supports mp_context from 3.7 onwards
        ctx_kw = {"mp_context": mp_context}
    else:
        ctx_kw = {}

    os.environ['QUTIP_IN_PARALLEL'] = 'TRUE'
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=map_kw['num_cpus'], **ctx_kw,
        ) as executor:
            waiting = set()
            i = 0
            while i < len(values):
                # feed values to the executor, ensuring that there is at
                # most one task per worker at any moment in time so that
                # we can shutdown without waiting for greater than the time
                # taken by the longest task
                if len(waiting) >= map_kw['num_cpus']:
                    # no space left, wait for a task to complete or
                    # the time to run out
                    timeout = max(0, end_time - time.time())
                    _done, waiting = concurrent.futures.wait(
                        waiting,
                        timeout=timeout,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                if time.time() >= end_time:
                    # no time left, exit the loop
                    break
                while len(waiting) < map_kw['num_cpus'] and i < len(values):
                    # space and time available, add tasks
                    value = values[i]
                    future = executor.submit(
                        task, *((value,) + task_args), **task_kwargs,
                    )
                    # small hack to avoid add_done_callback not supporting
                    # extra arguments and closures inside loops retaining
                    # a reference not a value:
                    future._i = i
                    future.add_done_callback(_done_callback)
                    waiting.add(future)
                    i += 1

            timeout = max(0, end_time - time.time())
            concurrent.futures.wait(waiting, timeout=timeout)
    finally:
        os.environ['QUTIP_IN_PARALLEL'] = 'FALSE'

    progress_bar.finished()
    return results


def loky_pmap(task, values, task_args=None, task_kwargs=None,
              reduce_func=None, map_kw=None,
              progress_bar=None, progress_bar_kwargs={}):
    """
    Parallel execution of a mapping of `values` to the function `task`. This
    is functionally equivalent to::

        result = [task(value, *task_args, **task_kwargs) for value in values]

    Use the loky module instead of multiprocessing.

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
    reduce_func : func (optional)
        If provided, it will be called with the output of each tasks instead of
        storing a them in a list.
    progress_bar : string
        Progress bar options's string for showing progress.
    progress_bar_kwargs : dict
        Options for the progress bar.
    map_kw: dict (optional)
        Dictionary containing entry for:
        'timeout': Maximum time for the whole map.
        'job_timeout': Maximum time for each job in the map.
        'num_cpus': Number of job to run at once.

    Returns
    --------
    result : list
        The result list contains the value of
        ``task(value, *task_args, **task_kwargs)`` for
        each value in ``values``. If a ``reduce_func`` is provided, and empty
        list will be returned.

    """
    if task_args is None:
        task_args = ()
    if task_kwargs is None:
        task_kwargs = {}
    map_kw = _read_map_kw(map_kw)
    os.environ['QUTIP_IN_PARALLEL'] = 'TRUE'
    from loky import get_reusable_executor, TimeoutError

    progress_bar = progess_bars[progress_bar]()
    progress_bar.start(len(values), **progress_bar_kwargs)

    executor = get_reusable_executor(max_workers=map_kw['num_cpus'])
    end_time = map_kw['timeout'] + time.time()
    job_time = map_kw['job_timeout']
    results = []

    try:
        jobs = [executor.submit(task, value, *task_args, **task_kwargs)
               for value in values]

        for job in jobs:
            remaining_time = min(end_time - time.time(), job_time)
            result = job.result(remaining_time)
            if reduce_func is not None:
                reduce_func(result)
            else:
                results.append(result)
            progress_bar.update()

    except KeyboardInterrupt as e:
        [job.cancel() for job in jobs]
        raise e

    except TimeoutError:
        [job.cancel() for job in jobs]

    finally:
        executor.shutdown()
    progress_bar.finished()
    os.environ['QUTIP_IN_PARALLEL'] = 'FALSE'
    return results


def get_map(options):
    if "parallel" in options['map']:
        return parallel_map
    elif "serial" in options['map']:
        return serial_map
    elif "loky" in options['map']:
        return loky_pmap
    else:
        raise ValueError("map not found, available options are 'parallel',"
                         " 'serial' and 'loky'")
