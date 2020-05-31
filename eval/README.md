# Simple Arithmetic Interpreter

The library provides a simple interpreter, which can be used for *some* grammars
recognized by [`arithmetic-parser`] (e.g., real-valued arithmetic).
The interpreter provides support for native functions,
which allows to overcome some syntax limitations (e.g., the lack of control flow
can be solved with native `if` / `loop` functions).

The interpreter is quite slow – 1–2 orders of magnitude slower than native
floating-point arithmetic.

[`arithmetic-parser`]: https://docs.rs/crates/arithmetic-parser
