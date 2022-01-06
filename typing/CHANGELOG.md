# Changelog

All notable changes to this project (the `arithmetic-typing` crate) will be
documented in this file. The project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Fix false positive during recursive type check for native parameterized functions.
  Previously, an assignment such as `reduce = fold;` (with `fold` being
  a native parametric function) or importing an object / tuple with functional fields
  led to such an error. (#100, (#105)

- Fix handling recursive type constraints, such as `|obj| (obj.len)(obj)`. Previously,
  such constraints led to stack overflow. (#105)

## 0.3.0 - 2021-05-24

The initial release of the `arithmetic-typing` crate.
