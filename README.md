# Flexible Arithmetic Parser and Interpreter

[![Build Status](https://github.com/slowli/arithmetic-parser/workflows/Rust/badge.svg?branch=master)](https://github.com/slowli/arithmetic-parser/actions)
[![License: Apache-2.0](https://img.shields.io/github/license/slowli/arithmetic-parser.svg)](https://github.com/slowli/arithmetic-parser/blob/master/LICENSE)

This repository contains a versatile parser for arithmetic expressions
which allows customizing literal definitions, type annotations and several other aspects of parsing.
The repository also contains several auxiliary crates (for example, a simple interpreter). 

## Contents

- [`arithmetic-parser`](parser) is the core parsing library.
- [`arithmetic-eval`](eval) is a simple interpreter, which could be used on parsed expressions
  in *some* cases. See the crate docs for more details on its limitations.
- [`arithmetic-parser-cli`](cli) is the CLI / REPL for the library.

## License

All code in this repository is licensed under the [Apache-2.0 license](LICENSE).
