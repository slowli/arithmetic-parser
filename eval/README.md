# Simple Arithmetic Interpreter

[![Build Status](https://github.com/slowli/arithmetic-parser/workflows/Rust/badge.svg?branch=master)](https://github.com/slowli/arithmetic-parser/actions)
[![License: Apache-2.0](https://img.shields.io/crates/l/arithmetic-eval)](https://github.com/slowli/arithmetic-parser/blob/master/LICENSE)
![rust 1.42+ required](https://img.shields.io/badge/rust-1.42+-blue.svg) 

**Links:** [![Docs on docs.rs](https://docs.rs/arithmetic-eval/badge.svg)](https://docs.rs/arithmetic-eval/)
[![crate docs (master)](https://img.shields.io/badge/master-yellow.svg?label=docs)](https://slowli.github.io/arithmetic-parser/arithmetic_eval/) 
[![changelog](https://img.shields.io/badge/-changelog-orange)](CHANGELOG.md)

The library provides a simple interpreter, which can be used for *some* grammars
recognized by [`arithmetic-parser`] (e.g., real-valued and complex-valued arithmetic).
The interpreter provides support for native functions,
which allows to overcome some syntax limitations (e.g., the lack of control flow
can be solved with native `if` / `loop` functions).

In general, the interpreter is quite opinionated on how to interpret language features.
The primary goal is to be intuitive for simple grammars (such as the aforementioned
real-valued arithmetic), even if this comes at the cost of rendering the interpreter
unusable for other grammars. Universality is explicitly not the design goal.

The interpreter is quite slow – 1–2 orders of magnitude slower than native
floating-point arithmetic.

## Usage

Add this to your `Crate.toml`:

```toml
[dependencies]
arithmetic-eval = "0.2.0-beta.1"
```

Please see the crate docs for the examples of usage.

## License

Licensed under the [Apache-2.0 license](LICENSE).

[`arithmetic-parser`]: https://docs.rs/crates/arithmetic-parser
