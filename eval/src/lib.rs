//! Simple interpreter for ASTs produced by [`arithmetic-parser`].
//!
//! # Assumptions
//!
//! - There is only one numeric type, which is complete w.r.t. all arithmetic operations.
//!   This is expressed via type constraints on relevant types via the [`Number`] trait.
//! - Arithmetic operations are assumed to be infallible; panics during their execution
//!   are **not** caught by the interpreter.
//! - Grammar literals are directly parsed to the aforementioned numeric type.
//!
//! These assumptions do not hold for some grammars parsed by the crate. For example, finite
//! cyclic groups have two types (scalars and group elements) and thus cannot be effectively
//! interpreted.
//!
//! # Semantics
//!
//! - All variables are immutable. Re-declaring a var shadows the previous declaration.
//! - Functions are first-class (in fact, a function is just a variant of the [`Value`] enum).
//! - Functions can capture variables (including other functions). All captures are by value.
//! - Arithmetic operations are defined on primitive vars and tuples. With tuples, operations
//!   are performed per-element. Binary operations require tuples of the same size,
//!   or a tuple and a primitive value. As an example, `(1, 2) + 3` and `(2, 3) / (4, 5)` are valid,
//!   but `(1, 2) * (3, 4, 5)` isn't.
//! - Methods are considered syntactic sugar for functions, with the method receiver considered
//!   the first function argument. For example, `(1, 2).map(sin)` is equivalent to
//!   `map((1, 2), sin)`.
//! - No type checks are performed before evaluation.
//! - Type annotations are completely ignored. This means that the interpreter may execute
//!   code that is incorrect with annotations (e.g., assignment of a tuple to a variable which
//!   is annotated to have a numeric type).
//! - Order comparisons (`>`, `<`, `>=`, `<=`) are desugared as follows. First, the `cmp` function
//!   is called with LHS and RHS as args (in this order). The result is then interpreted as
//!   [`Ordering`](core::cmp::Ordering) (-1 is `Less`, 1 is `Greater`, 0 is `Equal`;
//!   anything else leads to an error).
//!   Finally, the `Ordering` is used to compute the original comparison operation. For example,
//!   if `cmp(x, y) == -1`, then `x < y` and `x <= y` will return `true`, and `x > y` will
//!   return `false`.
//!
//! [`arithmetic-parser`]: https://crates.io/crates/arithmetic-parser
//!
//! # Examples
//!
//! ```
//! use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
//! use arithmetic_eval::{fns, Comparisons, Environment, Prelude, Value, VariableMap};
//!
//! # fn main() -> anyhow::Result<()> {
//! let program = r#"
//!     // The interpreter supports all parser features, including
//!     // function definitions, tuples and blocks.
//!     order = |x, y| (min(x, y), max(x, y));
//!     assert(order(0.5, -1) == (-1, 0.5));
//!     (_, M) = order(3^2, { x = 3; x + 5 });
//!     M
//! "#;
//! let program = Untyped::<F32Grammar>::parse_statements(program)?;
//!
//! let mut env = Environment::new();
//! // Add some native functions to the environment.
//! env.extend(Prelude.iter());
//! env.extend(Comparisons.iter());
//!
//! // To execute statements, we first compile them into a module.
//! let module = env.compile_module("test", &program)?;
//! // Then, the module can be run.
//! assert_eq!(module.run()?, Value::Number(9.0));
//! # Ok(())
//! # }
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![doc(html_root_url = "https://docs.rs/arithmetic-eval/0.2.0-beta.1")]
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

pub use self::{
    compiler::CompilerExt,
    env::Environment,
    error::{Error, ErrorKind, EvalResult},
    executable::{ExecutableModule, ExecutableModuleBuilder, ModuleImports},
    module_id::{IndexedId, ModuleId, WildcardId},
    values::{CallContext, Function, InterpretedFn, NativeFn, SpannedValue, Value, ValueType},
    variable_map::{Comparisons, Prelude, VariableMap},
};

use num_traits::Pow;

use core::ops;

use arithmetic_parser::grammars::NumLiteral;

mod compiler;
mod env;
pub mod error;
mod executable;
pub mod fns;
mod module_id;
mod values;
mod variable_map;

/// Number with fully defined arithmetic operations.
pub trait Number: NumLiteral + ops::Neg<Output = Self> + Pow<Self, Output = Self> {}

impl Number for f32 {}
impl Number for f64 {}
#[cfg(feature = "num-complex")]
impl Number for num_complex::Complex32 {}
#[cfg(feature = "num-complex")]
impl Number for num_complex::Complex64 {}
