# Monte Carlo Markov Kernels

<!-- badges: start -->
[![CI](https://github.com/awllee/MonteCarloMarkovKernels.jl/workflows/CI/badge.svg)](https://github.com/awllee/MonteCarloMarkovKernels.jl/actions)
[![codecov](https://codecov.io/gh/awllee/MonteCarloMarkovKernels.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/awllee/MonteCarloMarkovKernels.jl)
<!-- badges: end -->

This package provides some simple Monte Carlo Markov kernel implementations.

Currently implemented:

* Random walk Metropolis
* Adaptive Metropolis

To be implemented in the future:

* Metropolis-adjusted Langevin
* Hamiltonian Monte Carlo

The package also provides implementations of some consistent asymptotic variance estimators:

* Batch means
* Spectral variance
