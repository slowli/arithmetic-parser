# Simple Arithmetic Interpreter

[![Build Status](https://github.com/slowli/arithmetic-parser/workflows/Rust/badge.svg?branch=master)](https://github.com/slowli/arithmetic-parser/actions)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue)](https://github.com/slowli/arithmetic-parser#license)
![rust 1.44+ required](https://img.shields.io/badge/rust-1.44+-blue.svg) 

**Links:** [![Docs on docs.rs](https://docs.rs/arithmetic-eval/badge.svg)](https://docs.rs/arithmetic-eval/)
[![crate docs (master)](https://img.shields.io/badge/master-yellow.svg?label=docs)](https://slowli.github.io/arithmetic-parser/arithmetic_eval/) 
[![changelog](https://img.shields.io/badge/-changelog-orange)](CHANGELOG.md)

This library provides a simple interpreter, which can be used for some grammars
recognized by [`arithmetic-parser`], e.g., integer-, real-, complex-valued and modular arithmetic.
(Both built-in integer types and big integers from [`num-bigint`] are supported.)
The interpreter provides support for native functions,
which allows to overcome some syntax limitations (e.g., the lack of control flow
can be solved with native `if` / `loop` functions). Native functions and opaque reference types
allow effectively embedding the interpreter into larger Rust applications.

The interpreter is somewhat opinionated on how to interpret language features
(e.g., in terms of arithmetic ops for tuple / object arguments).
On the other hand, handling primitive types is fully customizable, just like their parsing
in `arithmetic-parser`.
The primary goal is to be intuitive for simple grammars (such as the aforementioned
real-valued arithmetic).

The interpreter is quite slow – 1–2 orders of magnitude slower than native arithmetic.

## Usage

Add this to your `Crate.toml`:

```toml
[dependencies]
arithmetic-eval = "0.3.0"
```

### Script samples

A simple script relying entirely on standard functions.

```text
minmax = |xs| xs.fold(#{ min: INF, max: -INF }, |acc, x| #{
     min: if(x < acc.min, x, acc.min),
     max: if(x > acc.max, x, acc.max),
});
assert_eq((3, 7, 2, 4).minmax().min, 2);
assert_eq((5, -4, 6, 9, 1).minmax(), #{ min: -4, max: 9 });
```

Recursive quick sort implementation:

```text
quick_sort = |xs, quick_sort| {
    // We need to pass a function as an arg since the top-level fn
    // is undefined at this point.
  
    if(xs == (), || (), || {
        (pivot, ...rest) = xs;
        lesser_part = rest.filter(|x| x < pivot).quick_sort(quick_sort);
        greater_part = rest.filter(|x| x >= pivot).quick_sort(quick_sort);
        lesser_part.push(pivot).merge(greater_part)
    })()
};
// Shortcut to get rid of an awkward fn signature.
sort = |xs| xs.quick_sort(quick_sort);

assert_eq((1, 7, -3, 2, -1, 4, 2).sort(), (-3, -1, 1, 2, 2, 4, 7));

// Generate a larger array to sort. `rand_num` is a custom native function
// that generates random numbers in the specified range.
xs = array(1000, |_| rand_num(0, 100)).sort();
// Check that elements in `xs` are monotonically non-decreasing.
{ sorted } = xs.fold(
    #{ prev: -1, sorted: true },
    |{ prev, sorted }, x| #{
        prev: x,
        sorted: sorted && prev <= x
    },
);
assert(sorted);
```

Please see the crate docs and [examples](examples) for more examples.

## See also

- [`arithmetic-typing`] is a type checker / inference tool for ASTs evaluated
  by this crate.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE)
or [MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in `arithmetic-eval` by you, as defined in the Apache-2.0 license,
shall be dual licensed as above, without any additional terms or conditions.

[`arithmetic-parser`]: https://crates.io/crates/arithmetic-parser
[`arithmetic-typing`]: https://crates.io/crates/arithmetic-typing
[`num-bigint`]: https://crates.io/crates/num-bigint
[Schnorr signatures]: https://en.wikipedia.org/wiki/Schnorr_signature
