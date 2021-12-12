//! Simple interpreter for ASTs produced by [`arithmetic-parser`].
//!
//! # How it works
//!
//! 1. A `Block` of statements is *compiled* into an [`ExecutableModule`]. Internally,
//!   compilation processes the AST of the block and transforms it into a non-recusrive form.
//!   An [`ExecutableModule`] may require *imports* (such as [`NativeFn`]s or constant [`Value`]s),
//!   which can be taken from a [`VariableMap`] (e.g., an [`Environment`]).
//! 2. [`ExecutableModule`] can then be executed, for the return value and/or for the
//!   changes at the top-level variable scope. There are two major variables influencing
//!   the execution outcome. An [arithmetic](crate::arith) is used to define arithmetic ops
//!   (`+`, unary and binary `-`, `*`, `/`, `^`) and comparisons (`==`, `!=`, `>`, `<`, `>=`, `<=`).
//!   Imports may be redefined at this stage as well.
//!
//! # Type system
//!
//! [`Value`]s have 5 major types:
//!
//! - **Primitive values** corresponding to literals in the parsed `Block`
//! - **Boolean values**
//! - **Functions,** which are further subdivided into native functions (defined in the Rust code)
//!   and interpreted ones (defined within a module)
//! - [**Tuples / arrays**](#tuples).
//! - [**Objects**](#objects).
//!
//! Besides these types, there is an auxiliary one: [`OpaqueRef`], which represents a
//! reference-counted native value, which can be returned from native functions or provided to
//! them as an arg, but is otherwise opaque from the point of view of the interpreted code
//! (cf. `anyref` in WASM).
//!
//! # Semantics
//!
//! - All variables are immutable. Re-declaring a var shadows the previous declaration.
//! - Functions are first-class (in fact, a function is just a variant of the [`Value`] enum).
//! - Functions can capture variables (including other functions). All captures are by value.
//! - Arithmetic operations are defined on primitive values, tuples and objects. Ops on primitives are defined
//!   via an [`Arithmetic`]. With tuples and objects, operations are performed per element / field.
//!   Binary operations require tuples of the same size / objects of the same shape,
//!   or a tuple / object and a primitive value.
//!   As an example, `(1, 2) + 3` and `#{ x: 2, y: 3 } / #{ x: 4, y: 5 }` are valid,
//!   but `(1, 2) * (3, 4, 5)` isn't.
//! - Methods are considered syntactic sugar for functions, with the method receiver considered
//!   the first function argument. For example, `(1, 2).map(sin)` is equivalent to
//!   `map((1, 2), sin)`.
//! - No type checks are performed before evaluation.
//! - Type annotations and type casts are completely ignored.
//!   This means that the interpreter may execute  code that is incorrect with annotations
//!   (e.g., assignment of a tuple to a variable which is annotated to have a numeric type).
//!
//! ## Value comparisons
//!
//! Equality comparisons (`==`, `!=`) are defined on all types of values.
//!
//! - For bool values, the comparisons work as expected.
//! - For functions, the equality is determined by the pointer (2 functions are equal
//!   iff they alias each other).
//! - `OpaqueRef`s either use the [`PartialEq`] impl of the underlying type or
//!   the pointer equality, depending on how the reference was created; see [`OpaqueRef`] docs
//!   for more details.
//! - Equality for primitive values is determined by the [`Arithmetic`].
//! - Tuples are equal if they contain the same number of elements and elements are pairwise
//!   equal.
//! - Different types of values are always non-equal.
//!
//! Order comparisons (`>`, `<`, `>=`, `<=`) are defined for primitive values only and use
//! [`OrdArithmetic`].
//!
//! ## Tuples
//!
//! - Tuples are created using a [`Tuple` expression], e.g., `(x, 1, 5)`.
//! - Indexing for tuples is performed via [`FieldAccess`] with a numeric field name: `xs.0`.
//!   Thus, the index is always a "compile-time" constant. An error is raised if the index
//!   is out of bounds or the receiver is not a tuple.
//! - Tuples can be destructured using a [`Destructure`] LHS of an assignment, e.g.,
//!   `(x, y, ...) = (1, 2, 3, 4)`. An error will be raised if the destructured value is
//!   not a tuple, or has an incompatible length.
//!
//! ## Objects
//!
//! - Objects can be created using [object expressions], which are similar to ones in JavaScript.
//!   For example, `#{ x: 1, y: (2, 3) }` will create an object with two fields:
//!   `x` equal to 1 and `y` equal to `(2, 3)`. Similar to Rust / modern JavaScript, shortcut
//!   field initialization is available: `#{ x, y }` will take vars `x` and `y` from the context.
//! - Object fields can be accessed via [`FieldAccess`] with a field name that is a valid
//!   variable name. No other values have such fields. An error will be raised if the object
//!   does not have the specified field.
//! - Objects can be destructured using an [`ObjectDestructure`] LHS of an assignment, e.g.,
//!   `{ x, y } = obj`. An error will be raised if the destructured value is not an object
//!   or does not have the specified fields. Destructuring is not exhaustive; i.e.,
//!   the destructured object may have extra fields.
//! - Functional fields are permitted. Similar to Rust, to call a function field, it must
//!   be enclosed in parentheses: `(obj.run)(arg0, arg1)`.
//!
//! # Crate features
//!
//! - `std`. Enables support of types from `std`, such as the `Error` trait, and propagates
//!   to dependencies. Importantly, `std` is necessary for floating-point arithmetics.
//! - `complex`. Implements [`Number`] for floating-point complex numbers from
//!   the [`num-complex`] crate (i.e., `Complex32` and `Complex64`). Enables complex number parsing
//!   in `arithmetic-parser`.
//! - `bigint`. Implements `Number` and a couple of other helpers for big integers
//!   from the [`num-bigint`] crate (i.e., `BigInt` and `BigUint`). Enables big integer parsing
//!   in `arithmetic-parser`.
//!
//! [`Arithmetic`]: crate::arith::Arithmetic
//! [`OrdArithmetic`]: crate::arith::OrdArithmetic
//! [`arithmetic-parser`]: https://crates.io/crates/arithmetic-parser
//! [`num-complex`]: https://crates.io/crates/num-complex
//! [`num-bigint`]: https://crates.io/crates/num-bigint
//! [`Tuple` expression]: arithmetic_parser::Expr::Tuple
//! [`Destructure`]: arithmetic_parser::Destructure
//! [`ObjectDestructure`]: arithmetic_parser::ObjectDestructure
//! [`FieldAccess`]: arithmetic_parser::Expr::FieldAccess
//! [object expressions]: arithmetic_parser::Expr::Object
//!
//! # Examples
//!
//! ```
//! use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
//! use arithmetic_eval::{
//!     Assertions, Comparisons, Environment, Prelude, Value, VariableMap,
//! };
//!
//! # fn main() -> anyhow::Result<()> {
//! let program = r#"
//!     // The interpreter supports all parser features, including
//!     // function definitions, tuples and blocks.
//!     order = |x, y| (min(x, y), max(x, y));
//!     assert_eq(order(0.5, -1), (-1, 0.5));
//!     (_, M) = order(3^2, { x = 3; x + 5 });
//!     M
//! "#;
//! let program = Untyped::<F32Grammar>::parse_statements(program)?;
//!
//! let mut env = Environment::new();
//! // Add some native functions to the environment.
//! env.extend(Prelude.iter());
//! env.extend(Assertions.iter());
//! env.extend(Comparisons.iter());
//!
//! // To execute statements, we first compile them into a module.
//! let module = env.compile_module("test", &program)?;
//! // Then, the module can be run.
//! assert_eq!(module.run()?, Value::Prim(9.0));
//! # Ok(())
//! # }
//! ```
//!
//! Using objects:
//!
//! ```
//! # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
//! # use arithmetic_eval::{Assertions, Environment, Prelude, Value, VariableMap};
//! # fn main() -> anyhow::Result<()> {
//! let program = r#"
//!     minmax = |...xs| xs.fold(#{ min: INF, max: -INF }, |acc, x| #{
//!          min: if(x < acc.min, x, acc.min),
//!          max: if(x > acc.max, x, acc.max),
//!     });
//!     assert_eq(minmax(3, 7, 2, 4).min, 2);
//!     assert_eq(minmax(5, -4, 6, 9, 1), #{ min: -4, max: 9 });
//! "#;
//! let program = Untyped::<F32Grammar>::parse_statements(program)?;
//!
//! let mut env = Environment::new();
//! env.extend(Prelude.iter().chain(Assertions.iter()));
//! env.insert("INF", Value::Prim(f32::INFINITY))
//!     .insert_prototypes(Prelude.prototypes());
//! let module = env.compile_module("minmax", &program)?;
//! module.run_in_env(&mut env)?;
//! # Ok(())
//! # }
//! ```
//!
//! More complex examples are available in the `examples` directory of the crate.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_root_url = "https://docs.rs/arithmetic-eval/0.3.0")]
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
    extern crate alloc;

    #[cfg(not(feature = "std"))]
    pub use alloc::{
        borrow::ToOwned,
        boxed::Box,
        format,
        rc::Rc,
        string::{String, ToString},
        vec,
        vec::Vec,
    };

    #[cfg(feature = "std")]
    pub use std::{
        borrow::ToOwned,
        boxed::Box,
        format,
        rc::Rc,
        string::{String, ToString},
        vec,
        vec::Vec,
    };
}

// FIXME: organize this mess
pub use self::{
    compiler::CompilerExt,
    error::{Error, ErrorKind, EvalResult},
    executable::{
        ExecutableModule, ExecutableModuleBuilder, IndexedId, ModuleId, ModuleImports, WildcardId,
        WithArithmetic,
    },
    values::{
        Assertions, CallContext, Comparisons, Environment, Filler, Function, InterpretedFn,
        NativeFn, Object, OpaqueRef, Prelude, Prototype, SpannedValue, StandardPrototypes, Tuple,
        Value, ValueType, VariableMap,
    },
};

pub mod arith;
mod compiler;
pub mod error;
mod executable;
pub mod fns;
mod values;

/// Marker trait for possible literals.
///
/// This trait is somewhat of a crutch, necessary to ensure that [function wrappers] can accept
/// number arguments and distinguish them from other types (booleans, vectors, tuples, etc.).
///
/// [function wrappers]: crate::fns::FnWrapper
pub trait Number: Clone + 'static {}

impl Number for i8 {}
impl Number for u8 {}
impl Number for i16 {}
impl Number for u16 {}
impl Number for i32 {}
impl Number for u32 {}
impl Number for i64 {}
impl Number for u64 {}
impl Number for i128 {}
impl Number for u128 {}

impl Number for f32 {}
impl Number for f64 {}

#[cfg(feature = "num-complex")]
impl Number for num_complex::Complex32 {}
#[cfg(feature = "num-complex")]
impl Number for num_complex::Complex64 {}

#[cfg(feature = "num-bigint")]
impl Number for num_bigint::BigInt {}
#[cfg(feature = "num-bigint")]
impl Number for num_bigint::BigUint {}
