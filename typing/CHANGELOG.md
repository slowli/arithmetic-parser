# Changelog

All notable changes to this project (the `arithmetic-typing` crate) will be
documented in this file. The project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Update `hashbrown` dependency to 0.15.

## 0.4.0-beta.1 - 2024-09-22

### Added

- Support block expressions in the name position for method calls, e.g., `xs.{Array.map}(|x| x > 0)`. (#117)

- Support no-std compilation mode.

### Changed

- Bump minimum supported Rust version to 1.70 and switch to 2021 Rust edition. (#107, #108, #112)

- Remove `Object::just()` constructor in favor of more general `From<[_; N]>` implementation. (#117)

- Rename `ErrorLocation` to `ErrorPathFragment` and its getter in `Error` from `location()` to `path()`
  in order to distinguish it from `Location` from the parser crate. (#124)

### Removed

- Remove lifetime generic from `Error` and related types. (#124)

### Fixed

- Fix false positive during recursive type check for native parameterized functions.
  Previously, an assignment such as `reduce = fold;` (with `fold` being
  a native parametric function) or importing an object / tuple with functional fields
  led to such an error. (#100, #105)

- Fix handling recursive type constraints, such as `|obj| (obj.len)(obj)`. Previously,
  such constraints led to stack overflow. (#105)

## 0.3.0 - 2021-05-24

The initial release of the `arithmetic-typing` crate.
