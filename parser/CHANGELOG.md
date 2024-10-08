# Changelog

All notable changes to this project (the `arithmetic-parser` crate) will be
documented in this file. The project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

## 0.4.0-beta.1 - 2024-09-22

### Added

- Make crate no-std-compatible and check compatibility in CI. (#101)

- Enable parsing block expressions in the name position for field access / method calls;
  e.g., `xs.{Array.len}` or `xs.{Array.map}(|x| x > 0)`. (#117)

### Changed

- Update `nom` parser dependency to v7 and `nom_locate` to v4. (#99, #100)

- Bump minimum supported Rust version to 1.70 and switch to 2021 Rust edition. (#107, #108, #112)

- Remove `MaybeSpanned` type in favor of `Location`. (#121)

- Change `Grammar` and `Parse` traits to use a generic associated `Type` instead of trait lifetimes. (#123)

### Removed

- Remove `StripCode` and `StripResultExt` traits as obsolete. (#121)

- Remove lifetime generic from `Error`. (#123)

## 0.3.0 - 2021-05-24

### Added

- Parse type annotations for varargs and tuples with a middle, such as
  `|...xs: T, y| { /* ... */ }` or `(head, ...tail: T)`. (#72)

- Make `with_span` parsing combinator public. (#77)

- Add `Expr::TypeCast` for cast expressions, such as `x as Bool`. (#83)

- Add `Expr::FieldAccess` for field access expressions, such as
  `point.x` or `xs.0`. (#84)

- Add `Expr::Object` for creating objects, i.e. aggregate data structures
  with heterogeneous named fields (known in Rust as structs). The `Object`
  syntax is similar to struct initialization in Rust or object creation
  in JS / TS. (#85, #87)

- Add object destructuring via `Lvalue::Object`. (#86)

- Allow mocking type parsing if the list of type annotations is statically known. (#88)

### Changed

- Make `Grammar` and `Parse` traits parametric on the lifetime of input.
  This allows to have type annotations dependent on this lifetime as well. (#77)

- Re-license the crate to MIT or Apache-2.0. (#87)

## 0.2.0 - 2020-12-05

*(All changes are relative compared to [the 0.2.0-beta.1 release](#020-beta1---2020-10-04))* 

### Added

- Add infrastructure for stripping code fragments (such as `Code` enum and
  `StripCode` trait). This allows breaking lifetime dependency between code
  and the outcome of its parsing. (#26) 

- Make `GrammarExt` methods more generic: they now accept inputs convertible
  to an `InputSpan`, such as `&str`. (#32)

- Allow switching Boolean expressions off. (#36)

- Add parsers for integer numbers. (#40)

- Add parsers for big integers from the `num-bigint` crate. (#46)

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

- Update dependencies. (#39)

- Use the `bitflags` crate for parser `Features`. (#50)

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
