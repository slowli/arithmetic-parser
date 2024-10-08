# CLI / REPL for Arithmetic Parser

[![Build Status](https://github.com/slowli/arithmetic-parser/actions/workflows/ci.yml/badge.svg)](https://github.com/slowli/arithmetic-parser/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue)](https://github.com/slowli/arithmetic-parser#license)

CLI and REPL for parsing and evaluating arithmetic expressions
that uses [`arithmetic-parser`](../parser) and [`arithmetic-eval`](../eval) internally.
Supports integer, modular, real and complex-valued arithmetic.
Each arithmetic is supplied with all standard functions from the `arithmetic-eval` crate
(`map`, `assert` and so on) and some functions / constants specific to the number type.

![REPL example](tests/snapshots/repl/basics.svg)

## Usage

**Tip.** Run the binary with `--help` flag to find out more details.

### Parsing

Use the `--ast` flag to output the AST of the expression. The AST is output
in the standard Rust debug format.

### Evaluating

Without the `--ast` or `--interactive` flags, the command evaluates
the provided expression in the selected arithmetic.

### REPL

With the `--interactive` / `-i` flag, the command works as REPL, allowing
to iteratively evaluate expressions.

### Minimum supported Rust version

The crate supports the latest stable Rust version. It may support previous stable Rust versions,
but this is not guaranteed.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE)
or [MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in `arithmetic-parser-cli` by you, as defined in the Apache-2.0 license,
shall be dual licensed as above, without any additional terms or conditions.
