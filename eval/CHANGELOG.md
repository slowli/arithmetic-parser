# Changelog

All notable changes to this project (the `arithmetic-eval` crate) will be
documented in this file. The project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
