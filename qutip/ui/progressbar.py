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
from __future__ import print_function

__all__ = ['BaseProgressBar', 'TextProgressBar',
           'EnhancedTextProgressBar', 'TqdmProgressBar',
           'get_progess_bar']

import time
import datetime
import sys


class BaseProgressBar(object):
    """
    An abstract progress bar with some shared functionality.

    Example usage:

        n_vec = linspace(0, 10, 100)
        pbar = TextProgressBar(len(n_vec))
        for n in n_vec:
            pbar.update(n)
            compute_with_n(n)
        pbar.finished()

    """

    def __init__(self, iterations=0, chunk_size=10):
        pass

    def start(self, iterations, chunk_size=10, **kwargs):
        self.N = float(iterations)
        self.n = 0
        self.p_chunk_size = chunk_size
        self.p_chunk = chunk_size
        self.t_start = time.time()
        self.t_done = self.t_start - 1

    def update(self, n=None):
        pass

    def total_time(self):
        return self.t_done - self.t_start

    def time_elapsed(self):
        return "%6.2fs" % (time.time() - self.t_start)

    def time_remaining_est(self, p):
        if p > 0.0:
            t_r_est = (time.time() - self.t_start) * (100.0 - p) / p
        else:
            t_r_est = 0

        dd = datetime.datetime(1, 1, 1) + datetime.timedelta(seconds=t_r_est)
        time_string = "%02d:%02d:%02d:%02d" % \
            (dd.day - 1, dd.hour, dd.minute, dd.second)

        return time_string

    def finished(self):
        self.t_done = time.time()


class TextProgressBar(BaseProgressBar):
    """
    A simple text-based progress bar.
    """

    def __init__(self, iterations=0, chunk_size=10):
        pass
        # super(TextProgressBar, self).start(iterations, chunk_size)

    def start(self, iterations, chunk_size=10, **kwargs):
        super(TextProgressBar, self).start(iterations, chunk_size)

    def update(self, n=None):
        self.n += 1
        n = self.n
        p = (n / self.N) * 100.0
        if p >= self.p_chunk:
            print("%4.1f%%." % p +
                  " Run time: %s." % self.time_elapsed() +
                  " Est. time left: %s" % self.time_remaining_est(p))
            sys.stdout.flush()
            self.p_chunk += self.p_chunk_size

    def finished(self):
        self.t_done = time.time()
        print("Total run time: %s" % self.time_elapsed())


class EnhancedTextProgressBar(BaseProgressBar):
    """
    An enhanced text-based progress bar.
    """

    def __init__(self, iterations=0, chunk_size=10):
        pass
        # super(EnhancedTextProgressBar, self).start(iterations, chunk_size)

    def start(self, iterations, chunk_size=10, **kwargs):
        super(EnhancedTextProgressBar, self).start(iterations, chunk_size)
        self.fill_char = '*'
        self.width = 25

    def update(self, n=None):
        self.n += 1
        n = self.n
        percent_done = int(round(n / self.N * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        prog_bar = ('[' + self.fill_char * num_hashes +
                    ' ' * (all_full - num_hashes) + ']')
        pct_place = (len(prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        prog_bar = (prog_bar[0:pct_place] +
                    (pct_string + prog_bar[pct_place + len(pct_string):]))
        prog_bar += ' Elapsed %s / Remaining %s' % (
            self.time_elapsed().strip(),
            self.time_remaining_est(percent_done))
        print('\r', prog_bar, end='')
        sys.stdout.flush()

    def finished(self):
        self.t_done = time.time()
        print("\r", "Total run time: %s" % self.time_elapsed())


class TqdmProgressBar(BaseProgressBar):
    """
    A progress bar using tqdm module
    """

    def __init__(self, iterations=0, chunk_size=10):
        from tqdm.auto import tqdm
        self.tqdm = tqdm

    def start(self, iterations, **kwargs):
        self.pbar = self.tqdm(total=iterations, **kwargs)
        self.t_start = time.time()
        self.t_done = self.t_start - 1

    def update(self, n=None):
        self.pbar.update()

    def finished(self):
        self.pbar.close()
        self.t_done = time.time()


def get_progess_bar(opt):
    if isinstance(opt, BaseProgressBar):
        return opt
    if opt in ["Enhanced", "enhanced"]:
        progress_bar = EnhancedTextProgressBar()
    elif opt in [True, "Text", "text"]:
        progress_bar = TextProgressBar()
    elif opt in ["Tqdm", "tqdm"]:
        progress_bar = TqdmProgressBar()
    elif opt in [False, "", None]:
        progress_bar = BaseProgressBar()
    else:
        raise ValueError("Progress Bar not understood")
    return progress_bar
