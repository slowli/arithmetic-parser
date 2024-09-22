# Flexible Arithmetic Parser and Interpreter

[![Build Status](https://github.com/slowli/arithmetic-parser/actions/workflows/ci.yml/badge.svg)](https://github.com/slowli/arithmetic-parser/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue)](https://github.com/slowli/arithmetic-parser#license)

This repository contains a versatile parser for arithmetic expressions
which allows customizing literal definitions, type annotations and several other aspects of parsing.
The repository also contains several auxiliary crates (for example, a simple interpreter).

## Contents

- [`arithmetic-parser`](parser) is the core parsing library.
- [`arithmetic-eval`](eval) is a simple interpreter that could be used on parsed expressions
  in *some* cases. See the crate docs for more details on its limitations.
- [`arithmetic-typing`](typing) is Hindley–Milner type inference for parsed expressions.
- [`arithmetic-parser-cli`](cli) is the CLI / REPL for the library.

## Why?

- The parser is designed to be reusable and customizable for simple scripting use cases.
  For example, it's used to dynamically define and process complex-valued functions
  in a [Julia set renderer](https://github.com/slowli/julia-set-rs).
  Customization specifically extends to literals; e.g., it is possible
  to have a single numeric literal / primitive type.
- Interpreter and type inference are natural complementary tools for the parser
  that allow evaluating parsed ASTs and reasoning about their correctness. Again,
  it is possible to fully customize primitive types and their mapping from literals,
  as well as semantics of arithmetic ops and (in case of typing) constraints they put on
  operands.
- Type inference is a challenging (and therefore interesting!) problem given the requirements
  (being able to work without any explicit type annotations).

## Project status 🚧

Early-stage; quite a bit of functionality is lacking, especially in interpreter and typing.
As an example, method resolution is a mess (methods are just syntax sugar for functions).

## Alternatives / similar tools

- Scripting languages like [Rhai](https://rhai.rs/book/) and [Gluon](https://gluon-lang.org/)
  are significantly more mature and have a sizable standard library, but are less customizable.
  E.g., there is a pre-determined set of primitive types with unchangeable semantics and type constraints.
  Rhai also does not have parser / interpreter separation or type inference support,
  and Gluon's syntax is a bit academic at times.

## Contributing

All contributions are welcome! See [the contributing guide](CONTRIBUTING.md) to help
you get involved.

## License

All code in this repository is licensed under either of [Apache License, Version 2.0](LICENSE-APACHE)
or [MIT license](LICENSE-MIT) at your option.
