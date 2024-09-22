//! Parser for arithmetic expressions with flexible definition of literals and support
//! of type annotations.
//!
//! Overall, parsed grammars are similar to Rust syntax,
//! [with a few notable differences](#differences-with-rust).
//!
//! # Supported syntax features
//!
//! - **Variables.** A variable name is defined similar to Rust and other programming languages,
//!   as a sequence of alphanumeric chars and underscores that does not start with a digit.
//! - **Literals.** The parser for literals is user-provided, thus allowing to apply the library
//!   to different domains (e.g., finite group arithmetic).
//! - `//` and `/* .. */` **comments**.
//! - Basic **arithmetic operations**: `+`, `-` (binary and unary), `*`, `/`, `^` (power).
//!   The parser outputs AST with nodes organized according to the operation priority.
//! - **Function calls**: `foo(1.0, x)`.
//! - **Parentheses** which predictably influence operation priority.
//!
//! The parser supports both complete and streaming (incomplete) modes; the latter is useful
//! for REPLs and similar applications.
//!
//! ## Optional syntax features
//!
//! These features can be switched on or off when defining a [`Parse`](grammars::Parse) impl
//! by declaring the corresponding [`Features`](grammars::Features).
//!
//! - **Tuples.** A tuple is two or more elements separated by commas, such as `(x, y)`
//!   or `(1, 2 * x)`. Tuples are parsed both as lvalues and rvalues.
//! - **Tuple destructuring.** Using a tuple as an lvalue, for example, `(x, y, z) = foo`.
//!   The "rest" syntax is also supported, either named or unnamed: `(head, ...tail) = foo`,
//!   `(a, ..., b, c) = foo`.
//! - **Function definitions.** A definition looks like a closure definition in Rust, e.g.,
//!   `|x| x - 10` or `|x, y| { z = max(x, y); (z - x, z - y) }`. A definition may be
//!   assigned to a variable (which is the way to define named functions).
//! - **Destructuring for function args.** Similar to tuple destructuring, it is possible to
//!   destructure and group args in function definitions, for example, `|(x, y), ...zs| { }`.
//! - **Blocks.** A block is several `;`-delimited statements enclosed in `{}` braces,
//!   e.g, `{ z = max(x, y); (z - x, z - y) }`. The blocks can be used in all contexts
//!   instead of a simple expression; for example, `min({ z = 5; z - 1 }, 3)`.
//! - **Objects.** Object is a mapping of string fields to values. Objects are defined via
//!   *object expressions*, which look similar to struct initialization in Rust or object
//!   initialization in JavaScript; for example, `#{ x: 1, y }`. (Note the `#` char at the start
//!   of the block; it is used to distinguish object expressions from blocks.)
//! - **Methods.** Method call is a function call separated from the receiver with a `.` char
//!   (e.g., `foo.bar(2, x)`).
//! - **Type annotations.** A type annotation in the form `var: Type` can be present
//!   in the lvalues or in the function argument definitions. The parser for type annotations
//!   is user-defined.
//! - **Boolean operations**: `==`, `!=`, `&&`, `||`, `!`.
//! - **Order comparisons,** that is, `>`, `<`, `>=`, and `<=` boolean ops.
//!
//! ## Differences with Rust
//!
//! *(within shared syntax constructs; of course, Rust is much more expressive)*
//!
//! - No keyword for assigning a variable (i.e., no `let` / `let mut`). There are no
//!   keywords in general.
//! - Functions are only defined via the closure syntax.
//! - There is "rest" destructuting for tuples and function arguments.
//! - Type annotations are placed within tuple elements, for example, `(x: Num, _) = y`.
//! - Object expressions are enclosed in `#{ ... }`, similarly to [Rhai](https://rhai.rs/).
//! - Object field access and method calls accept arbitrary block-enclosed "names" in addition to
//!   simple names. E.g., `xs.{Array.len}()` is a valid method call with `xs` receiver, `Array.len`
//!   method name and an empty args list.
//!
//! # Crate features
//!
//! ## `std`
//!
//! *(On by default)*
//!
//! Enables support of types from `std`, such as the `Error` trait, and propagates to dependencies.
//!
//! ## `num-complex`
//!
//! *(Off by default)*
//!
//! Implements [`NumLiteral`](grammars::NumLiteral) for floating-point complex numbers
//! (`Complex32` and `Complex64`).
//!
//! ## `num-bigint`
//!
//! *(Off by default)*
//!
//! Implements [`NumLiteral`](grammars::NumLiteral) for `BigInt` and `BigUint` from the `num-bigint` crate.
//!
//! # Examples
//!
//! Using a grammar for arithmetic on real values.
//!
//! ```
//! # use assert_matches::assert_matches;
//! use arithmetic_parser::{
//!     grammars::{F32Grammar, Parse, Untyped},
//!     NomResult, Statement, Expr, FnDefinition, LvalueLen,
//! };
//!
//! const PROGRAM: &str = "
//!     // This is a comment.
//!     x = 1 + 2.5 * 3 + sin(a^3 / b^2 /* another comment */);
//!     // Function declarations have syntax similar to Rust closures.
//!     some_function = |a, b| (a + b, a - b);
//!     other_function = |x| {
//!         r = min(rand(), 0.5);
//!         r * x
//!     };
//!     // Tuples and blocks are supported and have a similar syntax to Rust.
//!     (y, z) = some_function({ x = x - 0.5; x }, x);
//!     other_function(y - z)
//! ";
//!
//! # fn main() -> anyhow::Result<()> {
//! let block = Untyped::<F32Grammar>::parse_statements(PROGRAM)?;
//! // First statement is an assignment.
//! assert_matches!(
//!     block.statements[0].extra,
//!     Statement::Assignment { ref lhs, .. } if *lhs.fragment() == "x"
//! );
//! // The RHS of the second statement is a function.
//! let some_function = match &block.statements[1].extra {
//!     Statement::Assignment { rhs, .. } => &rhs.extra,
//!     _ => panic!("Unexpected parsing result"),
//! };
//! // This function has a single argument and a single statement in the body.
//! assert_matches!(
//!     some_function,
//!     Expr::FnDefinition(FnDefinition { ref args, ref body, .. })
//!         if args.extra.len() == LvalueLen::Exact(2)
//!             && body.statements.is_empty()
//!             && body.return_value.is_some()
//! );
//! # Ok(())
//! # }
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![doc(html_root_url = "https://docs.rs/arithmetic-parser/0.4.0-beta.1")]
#![warn(missing_docs, missing_debug_implementations)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::module_name_repetitions
)]

// Polyfill for `alloc` types.
mod alloc {
    #[cfg(not(feature = "std"))]
    extern crate alloc as std;

    pub use std::{boxed::Box, format, string::String, vec, vec::Vec};
}

pub use crate::{
    ast::{
        Block, Destructure, DestructureRest, Expr, ExprType, FnDefinition, Lvalue, LvalueLen,
        LvalueType, ObjectDestructure, ObjectDestructureField, ObjectExpr, SpannedExpr,
        SpannedLvalue, SpannedStatement, Statement, StatementType,
    },
    error::{Context, Error, ErrorKind, UnsupportedType},
    ops::{BinaryOp, Op, OpPriority, UnaryOp},
    parser::is_valid_variable_name,
    spans::{with_span, InputSpan, LocatedSpan, Location, NomResult, Spanned},
};

mod ast;
mod error;
pub mod grammars;
mod ops;
mod parser;
mod spans;
