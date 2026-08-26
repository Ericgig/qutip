import pytest
import functools
import os
import tempfile
import numpy as np


def _add_repeats_if_marked(metafunc):
    """
    If the metafunc is marked with the 'repeat' mark, then add the requisite
    number of repeats via parametrisation.
    """
    marker = metafunc.definition.get_closest_marker('repeat')
    if marker:
        count = marker.args[0]
        metafunc.fixturenames.append('_repeat_count')
        metafunc.parametrize('_repeat_count',
                             range(count),
                             ids=[f"rep({x+1})" for x in range(count)])


def _skip_cython_tests_if_unavailable(item):
    """
    Skip the current test item if Cython is unavailable for import, or isn't a
    high enough version.
    """
    if item.get_closest_marker("requires_cython"):
        # importorskip rather than mark.skipif because this way we get pytest's
        # version-handling semantics.
        pytest.importorskip('Cython', minversion='0.14')
        pytest.importorskip('filelock')


@pytest.hookimpl(trylast=True)
def pytest_generate_tests(metafunc):
    _add_repeats_if_marked(metafunc)


def pytest_runtest_setup(item):
    _skip_cython_tests_if_unavailable(item)


@pytest.fixture
def in_temporary_directory():
    """
    Creates a temporary directory for the lifetime of the fixture and changes
    into it.  All relative paths used will be in the temporary directory, and
    everything will automatically be cleaned up at the end of the fixture's
    life.
    """
    previous_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as temporary_dir:
        os.chdir(temporary_dir)
        yield
        # pytest should catch exceptions occuring in functions using the
        # fixture, so this should always be called.  We want it here rather
        # than outside to prevent the case of the directory failing to be
        # removed because it is 'busy'.
        os.chdir(previous_dir)


import warnings



SEEDSEQ = np.random.SeedSequence()


@pytest.fixture
def fixture_seeded_qt_random(request):
    import qutip._random as _random
    seed = SEEDSEQ.spawn(1)[0]
    request.node.user_properties.append(("qt_seed", seed))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="RANDOM", category=UserWarning)
        _random.seedseq = seed
        yield
        _random.seedseq = np.random.SeedSequence()


@pytest.fixture
def fixture_random_seed(request):
    seed = SEEDSEQ.spawn(1)[0]
    request.node.user_properties.append(("numpy_seed", seed))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="RANDOM", category=UserWarning)
        yield seed


@pytest.fixture
def fixture_generator(request):
    seed = SEEDSEQ.spawn(1)[0]
    request.node.user_properties.append(("numpy_generator", seed))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="RANDOM", category=UserWarning)
        yield np.random.default_rng(seed)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    # Print the seeds at the end of error messages
    outcome = yield
    report = outcome.get_result()

    if report.when == "call" and report.failed:
        if item.user_properties:
            props_str = "\n".join([
                f"{name}: {value}" for name, value in item.user_properties
            ])
            report.longrepr = f"{report.longrepr}\n\n{props_str}"



import weakref


class ML:
    tests = []

    def append(self, data):
        self.tests.append(data)

ml = ML()

def p():
    with open("test_using_random.txt", "w") as file:
        file.write("\n".join(ml.tests))

# weakref.finalize(ml, p)

@pytest.fixture(autouse=True)
def track_numpy_randomness(monkeypatch, request):
    """
    Detects any use of np.random functions or generator instantiation
    and issues a warning with the test ID.
    """
    test_id = request.node.nodeid

    def _warn_random_usage(func_name):
        ml.append(test_id)
        warnings.warn(
            f"RANDOM: Test '{test_id}' is using numpy randomness ({func_name}) without an explicit seed fixture.",
            UserWarning,
            stacklevel=3,
        )

    # 1. Intercept Legacy Random State Calls (np.random.randn, np.random.rand, etc.)
    # Legacy functions delegate internally through the global RandomState instance.
    orig_random_state = np.random.mtrand._rand

    class WrappedRandomState:
        def __getattr__(self, name):
            attr = getattr(orig_random_state, name)
            if callable(attr):

                def wrapper(*args, **kwargs):
                    _warn_random_usage(f"np.random.{name}")
                    return attr(*args, **kwargs)

                return wrapper
            return attr

    monkeypatch.setattr(np.random.mtrand, "_rand", WrappedRandomState())

    # 2. Intercept Modern Generator Creation (np.random.default_rng / np.random.Generator)
    orig_default_rng = np.random.default_rng

    def wrapped_default_rng(*args, **kwargs):
        _warn_random_usage("np.random.default_rng")
        return orig_default_rng(*args, **kwargs)

    monkeypatch.setattr(np.random, "default_rng", wrapped_default_rng)
