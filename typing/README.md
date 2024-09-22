# Type Inference for Arithmetic Grammars

[![Build Status](https://github.com/slowli/arithmetic-parser/workflows/CI/badge.svg?branch=master)](https://github.com/slowli/arithmetic-parser/actions)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue)](https://github.com/slowli/arithmetic-parser#license)
![rust 1.70+ required](https://img.shields.io/badge/rust-1.70+-blue.svg)

**Links:** [![crate docs (master)](https://img.shields.io/badge/master-yellow.svg?label=docs)](https://slowli.github.io/arithmetic-parser/arithmetic_typing/)

Hindley–Milner type inference for arithmetic expressions parsed
by the [`arithmetic-parser`] crate.

This crate allows parsing type annotations as a part of grammars, and inferring /
checking types for ASTs produced by `arithmetic-parser`.
Type inference is *partially* compatible with the interpreter from [`arithmetic-eval`];
if the inference algorithm succeeds on a certain expression / statement / block,
it will execute successfully, but not all successfully executing items pass type inference.

## Usage

Add this to your `Crate.toml`:

```toml
[dependencies]
arithmetic-typing = "0.4.0-beta.1"
```

### Quick overview

The type system supports all major constructions from [`arithmetic-parser`],
such as tuples, objects, and functional types. Functions and arithmetic operations
can place constraints on involved types, which are similar to Rust traits
(except *much* more limited). There is an equivalent for dynamic typing / trait objects
as well. Finally, the `any` type can be used to circumvent type system limitations.

The type system is generic with respect to primitive types. This allows customizing
processing of arithmetic ops and constraints, quite similar to `Arithmetic`s
in the [`arithmetic-eval`] crate.

For simple scripts, type inference may be successful without any annotations.
In the examples below, the only annotation is added to *test* type inference,
rather than to drive it:

```text
minmax: ([Num; N]) -> { max: Num, min: Num } = 
    |xs| xs.fold(#{ min: INF, max: -INF }, |acc, x| #{
         min: if(x < acc.min, x, acc.min),
         max: if(x > acc.max, x, acc.max),
    });
assert_eq((3, 7, 2, 4).minmax().min, 2);
assert_eq((5, -4, 6, 9, 1).minmax(), #{ min: -4, max: 9 });
```

```text
INF_PT = #{ x: INF, y: INF };

min_point: ([{ x: Num, y: Num }; N]) -> { x: Num, y: Num } = 
    |points| points
        .map(|pt| (pt, pt.x * pt.x + pt.y * pt.y))
        .fold(
            #{ min_r: INF, pt: INF_PT },
            |acc, (pt, r)| if(r < acc.min_r, #{ min_r: r, pt }, acc),
        )
        .pt;

assert_eq(
    array(10, |x| #{ x, y: 10 - x }).min_point(),
    #{ x: 5, y: 5 }
);
```

Please see the crate docs and [examples](examples) for info on type notation
and more examples of usage.

## Missing or incomplete features

- Sum / tagged union types
- Type constraints beyond simplest ones
- Specifying type vars in type annotations (beyond simplest cases)
- Type aliases

## See also

- [`arithmetic-eval`] is a simple interpreter that could be used on ASTs
  consumed by this crate.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE)
or [MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in `arithmetic-typing` by you, as defined in the Apache-2.0 license,
shall be dual licensed as above, without any additional terms or conditions.

[`arithmetic-parser`]: https://crates.io/crates/arithmetic-parser
[`arithmetic-eval`]: https://crates.io/crates/arithmetic-eval
