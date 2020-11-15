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

- Allow switching Boolean expressions off. (#36)

### Changed

- Use homegrown `LocatedSpan` instead of one from `nom_locate` crate.
  See the type docs for reasoning. (#26)

- Make most enums and structs with public fields non-exhaustive (e.g., `Expr`,
  `Statement`, `Lvalue`). (#26)

- Rework errors (#32):

  - Rename error types: `Error` to `ErrorKind`, `SpannedError` to `Error`.
  - Use `Error` as the main error type instead of a `Spanned<_>` wrapper.
  - Implement `std::error::Error` for `Error`.

- Use `//` and `/* .. */` comments instead of `#` ones. (#36)

- Use the `OpPriority` enum to encode op priorities instead of integers. (#36)

- Split `Grammar` into several traits (#38):

  - `ParseLiteral` responsible for parsing literals
  - `Grammar: ParseLiteral` for a complete set of parsers (literals + type annotations)
  - `Parse` (renamed from `GrammarExt`) to contain parsing features and parse `Block`s
  - Add helper wrappers `Typed` and `Untyped` to assist in composing parsing functionality.

- Export `ParseLiteral`, `Grammar` and `Parse` from the `grammars` module. (#38)

### Fixed

- Fix parsing of expressions like `1.abs()` for standard grammars. Previously,
  the parser consumed the `.` char as a part of the number literal, which led
  to a parser error. (#33)

- Fix relative priority of unary ops and method calls, so that `-1.abs()`
  is correctly parsed as `-(1.abs())`, not as `(-1).abs()`. (#33)

- Disallow using literals as function names. Thus, an expression like `1(2, x)`
  is no longer valid. (#33)

- Disallow chained comparisons, such as `x < y < z`. (#36)

- Make `&&` have higher priority than `||`, as in Rust. (#36)

## 0.2.0-beta.1 - 2020-10-04

### Added

- Implement an optional grammar feature: order comparisons, that is,
  `>`, `<`, `>=` and `<=` binary operations. (#23)

### Changed

- Update dependencies re-exported through the public interfaces, such as
  `nom_locate`. (#22)

## 0.1.0 - 2020-05-31

The initial release of `arithmetic-parser`.
