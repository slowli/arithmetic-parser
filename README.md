# Flexible Arithmetic Parser and Interpreter

[![Build Status](https://github.com/slowli/arithmetic-parser/workflows/Rust/badge.svg?branch=master)](https://github.com/slowli/arithmetic-parser/actions)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue)](https://github.com/slowli/arithmetic-parser#license)

This repository contains a versatile parser for arithmetic expressions
which allows customizing literal definitions, type annotations and several other aspects of parsing.
The repository also contains several auxiliary crates (for example, a simple interpreter). 

## Contents

- [`arithmetic-parser`](parser) is the core parsing library.
- [`arithmetic-eval`](eval) is a simple interpreter, which could be used on parsed expressions
  in *some* cases. See the crate docs for more details on its limitations.
- [`arithmetic-typing`](typing) is Hindley–Milner type inference for parsed expressions.
- [`arithmetic-parser-cli`](cli) is the CLI / REPL for the library.

## License

All code in this repository is licensed under either of [Apache License, Version 2.0](LICENSE-APACHE)
or [MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in `arithmetic-*` crates by you, as defined in the Apache-2.0 license,
shall be dual licensed as above, without any additional terms or conditions. 
