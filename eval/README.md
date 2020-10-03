# Simple Arithmetic Interpreter

[![Build Status](https://github.com/slowli/arithmetic-parser/workflows/Rust/badge.svg?branch=master)](https://github.com/slowli/arithmetic-parser/actions)
[![License: Apache-2.0](https://img.shields.io/github/license/slowli/arithmetic-parser.svg)](https://github.com/slowli/arithmetic-parser/blob/master/LICENSE)
![rust 1.41.0+ required](https://img.shields.io/badge/rust-1.41.0+-blue.svg) 

**Documentation:** [![Docs.rs](https://docs.rs/arithmetic-eval/badge.svg)](https://docs.rs/arithmetic-eval/)
[![crate docs (master)](https://img.shields.io/badge/master-yellow.svg?label=docs)](https://slowli.github.io/arithmetic-parser/arithmetic_eval/) 

The library provides a simple interpreter, which can be used for *some* grammars
recognized by [`arithmetic-parser`] (e.g., real-valued arithmetic).
The interpreter provides support for native functions,
which allows to overcome some syntax limitations (e.g., the lack of control flow
can be solved with native `if` / `loop` functions).

The interpreter is quite slow – 1–2 orders of magnitude slower than native
floating-point arithmetic.

## License

Licensed under the [Apache-2.0 license](LICENSE).

[`arithmetic-parser`]: https://docs.rs/crates/arithmetic-parser
