# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed

- Results are now correct for multiply connected planar
  subdivisions (i.e. meshes with holes)

### Removed

## [0.0.1] - 2025-01-03

### Added

- Rectilinear point locator to get an upper bound on the
  performance of point locators (nothing can beat that)
- Point in polygon algorithm based on the winding number,
  which allows for easy verification of the trapezoidal map
  implementation
- A few benchmarks
- Tests and proptests
- Trapezoidal map data structure for fast queries

[unreleased]: https://github.com/bluthej/ploc/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/bluthej/ploc/releases/tag/v0.1.0

