This pull request introduces a new `NumpyBackend `class that enables dynamic selection of the numpy_backend used in `qutip`. The class facilitates switching between different numpy implementations ( `numpy` and `jax.numpy` mainly) based on the configuration specified in the `settings.core` dictionary. 
