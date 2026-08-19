"""
Contain a global random number generator instance
"""

import numpy as np


__all__ = ["spawn", "seed"]


seedseq = np.random.SeedSequence()


def get_rng(seed=None):
    if seed is None:
        return np.random.default_rng(seedseq.spawn(1)[0])
    else:
        return np.random.default_rng(seed)


def get_SeedSequence(seed=None):
    if seed is None:
        return seedseq.spawn(1)[0]
    else:
        return np.random.SeedSequence(seed)
