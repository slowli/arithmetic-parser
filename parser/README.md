# Flexible Arithmetic Parser

[![Build Status](https://github.com/slowli/arithmetic-parser/workflows/Rust/badge.svg?branch=master)](https://github.com/slowli/arithmetic-parser/actions)
[![License: Apache-2.0](https://img.shields.io/crates/l/arithmetic-parser)](https://github.com/slowli/arithmetic-parser/blob/master/LICENSE)
![rust 1.41.0+ required](https://img.shields.io/badge/rust-1.41.0+-blue.svg) 

**Links:** [![Docs.rs](https://docs.rs/arithmetic-parser/badge.svg)](https://docs.rs/arithmetic-parser/)
[![crate docs (master)](https://img.shields.io/badge/master-yellow.svg?label=docs)](https://slowli.github.io/arithmetic-parser/arithmetic_parser/) 
[![changelog](https://img.shields.io/badge/-changelog-orange)](CHANGELOG.md)

A versatile parser for arithmetic expressions which allows customizing literal definitions,
type annotations and several other aspects of parsing.

## Supported Features

- **Variables.** A variable name is defined similar to Rust and other programming languages,
  as a sequence of alphanumeric chars and underscores that does not start with a digit.
- **Literals.** The parser for literals is user-provided, thus allowing to apply the library
  to different domains (e.g., finite group arithmetic).
- Python-like **comments** staring with `#`.
- Basic **arithmetic operations**: `+`, `-` (binary and unary), `*`, `/`, `^` (power).
  The parser outputs AST with nodes organized according to the operation priority.
- **Boolean operations**: `==`, `!=`, `&&`, `||`, `!`.
- **Function calls**: `foo(1.0, x)`.
- **Parentheses** which predictably influence operation priority.

The parser supports both complete and streaming (incomplete) modes; the latter is useful
for REPLs and similar applications.

### Optional Features

These features can be switched on or off when defining a grammar.

- **Tuples.** A tuple is two or more elements separated by commas, such as `(x, y)`
  or `(1, 2 * x)`. Tuples are parsed both as lvalues and rvalues.
- **Function definitions.** A definition looks like a closure definition in Rust, e.g.,
  `|x| x - 10` or `|x, y| { z = max(x, y); (z - x, z - y) }`. A definition may be
  assigned to a variable (which is the way to define named functions).
- **Blocks.** A block is several `;`-delimited statements enclosed in `{}` braces,
  e.g, `{ z = max(x, y); (z - x, z - y) }`. The blocks can be used in all contexts
  instead of a simple expression; for example, `min({ z = 5; z - 1 }, 3)`.
- **Methods.** Method call is a function call separated from the receiver with a `.` char;
  for example, `foo.bar(2, x)`. 
- **Type annotations.** A type annotation in the form `var: Type` can be present
  in the lvalues or in the function argument definitions. The parser for type annotations
  is user-defined.
- **Order comparisons,** that is, `>`, `<`, `>=`, and `<=` boolean ops.
  (The reason is that these ops do not make sense for some grammars, 
  such as for modular arithmetic.)

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
