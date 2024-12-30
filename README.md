<h1>
<p align="center">
  <img
    src="https://raw.githubusercontent.com/bluthej/ploc/docs/improve-readme/assets/logo.svg"
    alt="ploc logo">
  <br>ploc
</p>

## About

Ploc is a Rust library for efficient [point location](https://en.wikipedia.org/wiki/Point_location) queries.
The implementation is strongly influenced by [matplotlib's C++ implementation](https://github.com/matplotlib/matplotlib/blob/c11175d142403ff9af6e55ccb1feabccb990a7f6/src/tri/_tri.cpp), but differentiates itself by being able to handle arbitrary planar subdivisions instead of only triangulations, and by leveraging parallelism with [rayon](https://github.com/rayon-rs/rayon) to accelerate the queries.

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
