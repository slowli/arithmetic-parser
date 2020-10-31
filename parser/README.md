# Flexible Arithmetic Parser

[![Build Status](https://github.com/slowli/arithmetic-parser/workflows/Rust/badge.svg?branch=master)](https://github.com/slowli/arithmetic-parser/actions)
[![License: Apache-2.0](https://img.shields.io/crates/l/arithmetic-parser)](https://github.com/slowli/arithmetic-parser/blob/master/LICENSE)
![rust 1.42+ required](https://img.shields.io/badge/rust-1.42+-blue.svg) 

**Links:** [![Docs.rs](https://docs.rs/arithmetic-parser/badge.svg)](https://docs.rs/arithmetic-parser/)
[![crate docs (master)](https://img.shields.io/badge/master-yellow.svg?label=docs)](https://slowli.github.io/arithmetic-parser/arithmetic_parser/) 
[![changelog](https://img.shields.io/badge/-changelog-orange)](CHANGELOG.md)

A versatile parser for arithmetic expressions which allows customizing literal definitions,
type annotations and several other aspects of parsing.

## Features

The parser is overall similar to Rust. It supports variables, literals, comments,
arithmetic and boolean operations, parentheses, function calls, tuples and tuple destructuring,
function definitions, blocks, methods, and type annotations.

See the crate docs for more details on the supported features.

### Code Sample

Here is an example of code parsed with the grammar with real-valued literals
and the only supported type `Num`:

```text
# This is a comment.
x = 1 + 2.5 * 3 + sin(a^3 / b^2);
# Function declarations have syntax similar to Rust closures.
some_function = |a, b: Num| (a + b, a - b);
other_function = |x| {
    r = min(rand(), 0.5);
    r * x
};
# Tuples and blocks are supported and have a similar syntax to Rust.
(y, z: Num) = some_function({ x = x - 0.5; x }, x);
other_function(y - z)
```

## Implementation Details 

The parser is based on the [`nom`](https://docs.rs/nom/) crate. The core trait of the library,
`Grammar`, is designed in such a way that switching [optional features](#optional-features)
should not induce run-time overhead; the unused parsing code paths should be removed during
compilation.

## License

Licensed under the [Apache-2.0 license](LICENSE).
