# Simple Arithmetic Interpreter

[![Build Status](https://github.com/slowli/arithmetic-parser/workflows/CI/badge.svg?branch=master)](https://github.com/slowli/arithmetic-parser/actions)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue)](https://github.com/slowli/arithmetic-parser#license)
![rust 1.65+ required](https://img.shields.io/badge/rust-1.65+-blue.svg)
![no_std supported](https://img.shields.io/badge/no__std-tested-green.svg)

**Links:** [![Docs on docs.rs](https://img.shields.io/docsrs/arithmetic-eval)](https://docs.rs/arithmetic-eval/)
[![crate docs (master)](https://img.shields.io/badge/master-yellow.svg?label=docs)](https://slowli.github.io/arithmetic-parser/arithmetic_eval/) 
[![changelog](https://img.shields.io/badge/-changelog-orange)](CHANGELOG.md)

This library provides a simple interpreter, which can be used for some grammars
recognized by [`arithmetic-parser`], e.g., integer-, real-, complex-valued and modular arithmetic.
(Both built-in integer types and big integers from [`num-bigint`] are supported.)
The interpreter provides support for native functions,
which allows to overcome some syntax limitations (e.g., the lack of control flow
can be solved with native `if` / `while` functions). Native functions and opaque reference types
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
minmax = |...xs| xs.fold(#{ min: INF, max: -INF }, |acc, x| #{
     min: if(x < acc.min, x, acc.min),
     max: if(x > acc.max, x, acc.max),
});
assert_eq(minmax(3, 7, 2, 4).min, 2);
assert_eq(minmax(5, -4, 6, 9, 1), #{ min: -4, max: 9 });
```

Recursive quick sort implementation:

```text
sort = defer(|sort| {
    // `defer` allows to define a function recursively
    |xs| {
        if(xs == (), || (), || {
            (pivot, ...rest) = xs;
            lesser_part = sort(rest.filter(|x| x < pivot));
            greater_part = sort(rest.filter(|x| x >= pivot));
            lesser_part.push(pivot).merge(greater_part)
        })()
    }
});

assert_eq(sort((1, 7, -3, 2, -1, 4, 2)), (-3, -1, 1, 2, 2, 4, 7));

// Generate a larger array to sort. `rand_num` is a custom native function
// that generates random numbers in the specified range.
xs = sort(array(1000, |_| rand_num(0, 100)));
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

Defining a type:

```text
Vector = defer(|Self| impl(#{
    len: |{ x, y }| sqrt(x * x + y * y),
    add: |self, other| Self(self + other),
    to_unit_scale: |self| if(self.x == 0 && self.y == 0,
        || self,
        || Self(self / self.len()),
    )(),
}));

assert_eq(
    Vector(#{ x: 3, y: 4 }).add(Vector(#{ x: -1, y: 1 })),
    Vector(#{ x: 2, y: 5 }),
);
scaled = Vector(#{ x: 3, y: -4 }).to_unit_scale();
assert_close(scaled.len(), 1);
assert_close(scaled.x, 0.6);
assert_close(scaled.y, -0.8);
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
