<h1>
<p align="center">
  <img
    src="https://raw.githubusercontent.com/bluthej/ploc/docs/improve-readme/assets/logo.svg"
    alt="ploc logo">
  <br>ploc
</p>
</h1>

[![tests](https://github.com/bluthej/ploc/workflows/test/badge.svg)](https://github.com/bluthej/ploc/actions)
[![Codecov](https://codecov.io/github/bluthej/ploc/coverage.svg?branch=main)](https://codecov.io/gh/bluthej/ploc)
![minimum rustc 1.65](https://img.shields.io/badge/rustc-1.65+-red.svg)

## About

Ploc is a Rust library for efficient [point location](https://en.wikipedia.org/wiki/Point_location) queries.
The implementation is strongly influenced by [`matplotlib`'s C++ implementation], but differentiates itself by being able to handle arbitrary planar subdivisions instead of only triangulations, and by leveraging parallelism with [rayon](https://github.com/rayon-rs/rayon) to accelerate the queries.

Python bindings will soon be available at [ploc-py](https://github.com/bluthej/ploc-py).

## Roadmap and Status

The goal is to provide a drop-in replacement for `matplotlib`'s `TrapezoidMapTriFinder`, but to go further in a few respects:
- [x] Generalize the implementation to arbitrary planar subdivisions
- [ ] Provide results that are compatible with the mpl implementation
- [x] Get better single-threaded performance
- [x] Accelerate the queries with parallelism
- [ ] Get peak memory usage to be competitive (currently ploc's peak memory usage is at least 50% higher than that of the mpl implementation)
- [ ] Provide Python bindings
- [ ] Provide richer output for edge/corner cases (litteraly, when a query point lies on an edge or is a vertex of the planar subdivision, it would be nice to report that)

## Influences

This work has been greatly influenced by the following:
- [`matplotlib`'s C++ implementation]: this is where I first learned about this algorithm, and the present project is very much indebted to this implementation. There was a comment that mentioned the De Berg book which I mention below.
- [De Berg, M. (2000). Computational geometry: algorithms and applications. Springer Science & Business Media.]: a really good book on computational geometry. The chapter on the trapezoidal map explains everything really well.

[`matplotlib`'s C++ implementation]: https://github.com/matplotlib/matplotlib/blob/c11175d142403ff9af6e55ccb1feabccb990a7f6/src/tri/_tri.cpp
[De Berg, M. (2000). Computational geometry: algorithms and applications. Springer Science & Business Media.]: https://doi.org/10.1007/978-3-540-77974-2
