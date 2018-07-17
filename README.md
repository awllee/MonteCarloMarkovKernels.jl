# Monte Carlo Markov Kernels

[![Build Status](https://travis-ci.org/awllee/MonteCarloMarkovKernels.jl.svg?branch=master)](https://travis-ci.org/awllee/MonteCarloMarkovKernels.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/15g991vjv9ce1l4w?svg=true)](https://ci.appveyor.com/project/awllee/montecarlomarkovkernels-jl)
[![Coverage Status](https://coveralls.io/repos/github/awllee/MonteCarloMarkovKernels.jl/badge.svg?branch=master)](https://coveralls.io/github/awllee/MonteCarloMarkovKernels.jl?branch=master)
[![codecov.io](http://codecov.io/github/awllee/MonteCarloMarkovKernels.jl/coverage.svg?branch=master)](http://codecov.io/github/awllee/MonteCarloMarkovKernels.jl?branch=master)

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
