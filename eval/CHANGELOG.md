# Changelog

All notable changes to this project (the `arithmetic-eval` crate) will be
documented in this file. The project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Allow converting `ExecutableModule`s and related types (such as `ModuleImport`s)
  to versions with a static lifetime. (#26)

- Implement `while` as a native function. (#26)

- Add IDs for executable modules. These IDs can be used to locate code spans for errors
  and interpreted functions. (#31)

- Decouple `ExecutableModule` compilation and execution. This allows distinguishing
  between errors on these two steps and gives more control over module lifecycle. (#32)

- Make complex number support optional. (#39)

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
