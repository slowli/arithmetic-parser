# Type Inference for Arithmetics

[![Build Status](https://github.com/slowli/arithmetic-parser/workflows/Rust/badge.svg?branch=master)](https://github.com/slowli/arithmetic-parser/actions)
[![License: Apache-2.0](https://img.shields.io/crates/l/arithmetic-eval)](https://github.com/slowli/arithmetic-parser/blob/master/LICENSE)
![rust 1.44+ required](https://img.shields.io/badge/rust-1.44+-blue.svg)

**Links:** [![crate docs (master)](https://img.shields.io/badge/master-yellow.svg?label=docs)](https://slowli.github.io/arithmetic-parser/arithmetic_typing/)

Hindley–Milner type inference for arithmetic expressions parsed
by the [`arithmetic-parser`] crate.

This crate allows parsing type annotations as a part of grammars, and to infer
and check types for expressions / statements produced by `arithmetic-parser`.
Type inference is *partially* compatible with the interpreter from [`arithmetic-eval`];
if the inference algorithm succeeds on a certain expression / statement / block,
it will execute successfully, but not all successfully executing items pass type inference.

## Usage

Add this to your `Crate.toml`:

```toml
[dependencies]
arithmetic-typing = "0.2.0"
```

Please see the crate docs and [examples](examples) for the examples of usage.

## License

Licensed under the [Apache-2.0 license](LICENSE).

[`arithmetic-parser`]: https://crates.io/crates/arithmetic-parser
[`arithmetic-eval`]: https://crates.io/crates/arithmetic-eval
