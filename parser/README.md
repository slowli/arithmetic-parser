# Flexible Arithmetic Parser

[![Build Status](https://github.com/slowli/arithmetic-parser/workflows/CI/badge.svg?branch=master)](https://github.com/slowli/arithmetic-parser/actions)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue)](https://github.com/slowli/arithmetic-parser#license)
![rust 1.60+ required](https://img.shields.io/badge/rust-1.60+-blue.svg)
![no_std supported](https://img.shields.io/badge/no__std-tested-green.svg)

**Links:** [![Docs.rs](https://img.shields.io/docsrs/arithmetic-parser)](https://docs.rs/arithmetic-parser/)
[![crate docs (master)](https://img.shields.io/badge/master-yellow.svg?label=docs)](https://slowli.github.io/arithmetic-parser/arithmetic_parser/) 
[![changelog](https://img.shields.io/badge/-changelog-orange)](CHANGELOG.md)

A versatile parser for arithmetic expressions which allows customizing literal definitions,
type annotations and several other aspects of parsing.

## Usage

Add this to your `Crate.toml`:

```toml
[dependencies]
arithmetic-parser = "0.3.0"
```

The parser is overall similar to Rust. It supports variables, literals, comments,
arithmetic and boolean operations, parentheses, function calls, tuples and tuple destructuring,
function definitions, blocks, methods, and type annotations.
In other words, the parser forms a foundation of a minimalistic scripting language,
while leaving certain aspects up to user (most of all, specification of literals).

See the crate docs for more details on the supported syntax features.

### Code sample

Here is an example of code parsed with the grammar with real-valued literals
and the only supported type `Num`:

```text
// This is a comment.
x = 1 + 2.5 * 3 + sin(a^3 / b^2);

// Function declarations have syntax similar to Rust closures.
some_function = |x| {
    r = min(rand(), 0.5);
    r * x
};

// Objects are similar to JavaScript, except they require
// a preceding hash `#`, like in Rhai (https://rhai.rs/).
other_function = |a, b: Num| #{ sum: a + b, diff: a - b };

// Object destructuring is supported as well.
{ sum, diff: Num } = other_function(
    x,
    // Blocks are supported and have a similar syntax to Rust.
    some_function({ x = x - 0.5; x }),
);

// Tuples have syntax similar to Rust (besides spread syntax
// in destructuring, which is similar to one in JavaScript).
(x, ...tail) = (1, 2).map(some_function);
```

## Implementation details

The parser is based on the [`nom`](https://docs.rs/nom/) crate. The core trait of the library,
`Grammar`, is designed in such a way that switching optional features
should not induce run-time overhead; the unused parsing code paths should be removed during
compilation.

## See also

- [`arithmetic-eval`] is a simple interpreter that could be used on parsed ASTs.
- [`arithmetic-typing`] is a type checker / inference tool for parsed ASTs.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE)
or [MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in `arithmetic-parser` by you, as defined in the Apache-2.0 license,
shall be dual licensed as above, without any additional terms or conditions.

[`arithmetic-eval`]: https://crates.io/crates/arithmetic-eval
[`arithmetic-typing`]: https://crates.io/crates/arithmetic-typing
