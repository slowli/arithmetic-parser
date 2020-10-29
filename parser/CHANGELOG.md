# Changelog

All notable changes to this project (the `arithmetic-parser` crate) will be
documented in this file. The project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add infrastructure for stripping code fragments (such as `Code` enum and
  `StripCode` trait). This allows breaking lifetime dependency between code
  and the outcome of its parsing. (#26) 

- Make `GrammarExt` methods more generic: they now accept inputs convertible
  to an `InputSpan`, such as `&str`. (#32)

### Changed

- Use homegrown `LocatedSpan` instead of one from `nom_locate` crate.
  See the type docs for reasoning. (#26)

- Make most enums and structs with public fields non-exhaustive (e.g., `Expr`,
  `Statement`, `Lvalue`). (#26)

- Rework errors (#32):

  - Rename error types: `Error` to `ErrorKind`, `SpannedError` to `Error`.
  - Use `Error` as the main error type instead of a `Spanned<_>` wrapper.
  - Implement `std::error::Error` for `Error`.

### Fixed

- Fix parsing of expressions like `1.abs()` for standard grammars. Previously,
  the parser consumed the `.` char as a part of the number literal, which led
  to a parser error. (#33)

- Fix relative priority of unary ops and method calls, so that `-1.abs()`
  is correctly parsed as `-(1.abs())`, not as `(-1).abs()`. (#33)

- Disallow using literals as function names. Thus, an expression like `1(2, x)`
  is no longer valid. (#33)

## 0.2.0-beta.1 - 2020-10-04

### Added

- Implement an optional grammar feature: order comparisons, that is,
  `>`, `<`, `>=` and `<=` binary operations. (#23)

### Changed

- Update dependencies re-exported through the public interfaces, such as
  `nom_locate`. (#22)

## 0.1.0 - 2020-05-31

The initial release of `arithmetic-parser`.
