# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

- In `TrapMap::check` we now check that leaf nodes are
  _exactly_ trapezoid nodes (previously we would only
  check that leaf nodes were trapezoid nodes, but not
  the reverse)

### Fixed

### Removed

- The `parents` field of the `Node` type (it was only
  `pub(crate)` so this is not even a breaking change)

## [0.1.1] - 2025-05-15

### Fixed

- Results are now correct for multiply connected planar
  subdivisions (i.e. meshes with holes)

## [0.1.0] - 2025-01-03

### Added

- Rectilinear point locator to get an upper bound on the
  performance of point locators (nothing can beat that)
- Point in polygon algorithm based on the winding number,
  which allows for easy verification of the trapezoidal map
  implementation
- A few benchmarks
- Tests and proptests
- Trapezoidal map data structure for fast queries

[unreleased]: https://github.com/bluthej/ploc/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/bluthej/ploc/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/bluthej/ploc/releases/tag/v0.1.0

