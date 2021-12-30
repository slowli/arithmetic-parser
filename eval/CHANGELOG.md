# Changelog

All notable changes to this project (the `arithmetic-eval` crate) will be
documented in this file. The project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Make crate no-std-compatible and check compatibility in CI. (#101)

- Allow adding prototypes for objects, which act similarly to
  prototypes in JavaScript, but immutable. Prototypes can be created
  from the host code using the `Prototype` type and from scripts
  using the `CreatePrototype` function. (#103)

- Add the `Defer` function for deferred / recursive variable initialization. (#103)

- Add `AssertClose` and `AssertFails` assertions for approximate equality comparisons
  and failure testing, respectively. (#103)

- Add `all` and `any` functions for arrays / tuples.

### Changed

- Introduce new structs (`Tuple` and `Object`) to represent tuple and object values,
  respectively. These objects are tied to the corresponding `Value` variants. (#103)

- Add `T: 'static + Clone` restriction for the type argument in `ExecutableModule`, `Function`
  and most native functions. (#103)

- Restructure the crate by extracting types related to `Environment` and to `ExecutableModule`
  to separate modules. (#103)

- Change `ExecutableModule` interface for clarity. Now, `ExecutableModule`
  is only responsible for code; imports are always provided by an `Environment`. (#103)

- Make values contained in standard collections, such as `Prelude` and `Assertions`,
  available using static methods instead of instance ones. (#103)

### Removed

- Remove `ExecutableModelBuilder::set_imports()` as error-prone. Instead, deferred imports
  can be set from a `Filler`. (#103)

- Remove `ModuleImports`, its iterator types and `VariableMap` as obsolete.
  Related to the changes in the `ExecutableModule` interface. (#103)

### Fixed

- Remove bogus `T: Debug` bound in `ExecutableModule<T>` and related types. (#103)

## 0.3.0 - 2021-05-24

### Added

- Support `Expr::TypeCast` as a no-op. (#83)

- Support `Expr::FieldAccess` for tuple indexing. (#84)

- Support `Expr::Object` for creating objects, i.e. aggregate data structures
  with heterogeneous named fields (known in Rust as structs). This construction
  works similarly to creating structs in Rust or objects in JS / TS. (#85, #87)

- Support `Expr::FieldAccess` for accessing fields in objects. (#85)

- Support object destructuring via `Lvalue::Object`. Destructuring is non-exhaustive,
  i.e., the destructured object may have extra fields. (#86)

- Add `Array` and `Len` functions to generate arrays and get array / tuple length,
  respectively. (#88)

### Changed

- Rename `AuxErrorInfo::UnbalancedRhs` to `AuxErrorInfo::UnbalancedRhsTuple`. (#86)

- Re-license the crate to MIT or Apache-2.0. (#87)

- Rename `Value::Number` variant to `Prim` to better correspond to possible crate
  usage. (#89)

### Removed

- Remove `ErrorKind::InvalidCmpResult` as obsolete. (#88)

## 0.2.0 - 2020-12-05

*(All changes are relative compared to [the 0.2.0-beta.1 release](#020-beta1---2020-10-04))* 

### Added

- Allow converting `ExecutableModule`s and related types (such as `ModuleImport`s)
  to versions with a static lifetime. (#26)

- Implement `while` as a native function. (#26)

- Add IDs for executable modules. These IDs can be used to locate code spans for errors
  and interpreted functions. (#31)

- Decouple `ExecutableModule` compilation and execution. This allows distinguishing
  between errors on these two steps and gives more control over module lifecycle. (#32)

- Make complex number support optional. (#39)

- Support integer and modular arithmetics. (#40)

- Add arithmetics for big integers from the `num-bigint` crate. (#46)

- Add `Dbg` native function that outputs the argument(s) to stderr. (#46)

- Add `AssertEq` native function that compares two args and raises an error
  if they are not equal. (#47)

### Changed

- Change APIs related to code spans according to the updates in the parser crate. (#26)

- Make most enums and structs with public fields non-exhaustive (e.g., error types,
  `Value`). (#26)

- Rename error types: `EvalError` to `ErrorKind`, `SpannedEvalError` to `Error`.
  Make the `error` module public and only re-export the most used types from it
  to the crate root. (#31)

- Move getters such as `main_span()` from `ErrorWithBacktrace` to `Error`, which
  can be accessed via `source()`. (#31)

- Make `Backtrace` type crate-private; move backtrace call iterator
  directly to `ErrorWithBacktrace`. (#31)

- Refactor `ExecutableModule` creation: add a builder and remove creation
  methods from an interpreter. (#32)

- Refactor `Interpreter` (#32):

  - Rename to `Environment`
  - Extract interface related to module construction into a trait, `VariableMap`
  - Unify method names with `ModuleImports`
  - Use iterator traits (`IntoIterator` / `FromIterator` / `Extend`) to make
    `Environment` creation more idiomatic.

- Make `CallContext::mock()` accept custom module ID and call span. (#32)

- Crate types now have a numeric literal type param, rather than a `Grammar`. (#38)

- Change module execution logic to use a customizable abstraction (arithmetic)
  for foundational arithmetic ops and comparisons. (#40)

- Move `assert` function to a separate container, `Assertions`. (#47)

## 0.2.0-beta.1 - 2020-10-04

### Added

- Improve native function definitions. For example, it is now possible to
  define native functions with primitive / tuple / vector arguments
  and primitive / tuple / vector / result return types. See the `FnWrapper` docs
  for more details and usage examples. (#5)

- Add import getters for `ExecutableModule`s. (#15)

- Implement `Error` trait from the standard library for error types. (#17)

- Allow extracting undefined variables from blocks and function definitions
  via a `CompilerExt` trait. (#18)

- Implement order comparisons in the interpreter as syntactic sugar for `cmp`
  function. See the crate docs for more details. (#23)

### Changed

- Update dependencies re-exported through the public interfaces, such as
  `nom_locate` and `num-complex`. (#22)

## 0.1.0 - 2020-05-31

The initial release of `arithmetic-eval`.
