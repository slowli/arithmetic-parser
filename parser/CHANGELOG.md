# Changelog

All notable changes to this project (the `arithmetic-parser` crate) will be
documented in this file. The project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add infrastructure for stripping code fragments (such as `Code` enum and
  `StripCode` trait). This allows breaking lifetime dependency between code
  and the outcome of its parsing. (#26) 

### Changed

- Use homegrown `LocatedSpan` instead of one from `nom_locate` crate.
  See the type docs for reasoning. (#26)

- Make most enums non-exhaustive (e.g., `Expr`, `Statement`, `Lvalue`). (#26)

## 0.2.0-beta.1 - 2020-10-04

### Added

- Implement an optional grammar feature: order comparisons, that is,
  `>`, `<`, `>=` and `<=` binary operations. (#23)

### Changed

- Update dependencies re-exported through the public interfaces, such as
  `nom_locate`. (#22)

## 0.1.0 - 2020-05-31

The initial release of `arithmetic-parser`.
