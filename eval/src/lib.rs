//! Simple interpreter for ASTs produced by [`arithmetic-parser`].
//!
//! # Assumptions
//!
//! - There is only one numeric type, which is complete w.r.t. all arithmetic operations.
//!   This is expressed via type constraints, in [`Interpreter`].
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
//!   [`Ordering`] (-1 is `Less`, 1 is `Greater`, 0 is `Equal`; anything else leads to an error).
//!   Finally, the `Ordering` is used to compute the original comparison operation. For example,
//!   if `cmp(x, y) == -1`, then `x < y` and `x <= y` will return `true`, and `x > y` will
//!   return `false`.
//!
//! [`arithmetic-parser`]: https://crates.io/crates/arithmetic-parser
//! [`Interpreter`]: struct.Interpreter.html
//! [`Value`]: enum.Value.html
//! [`Ordering`]: https://doc.rust-lang.org/std/cmp/enum.Ordering.html
//!
//! # Examples
//!
//! ```
//! use arithmetic_parser::{grammars::F32Grammar, Grammar, GrammarExt};
//! use arithmetic_eval::{fns, Comparisons, Environment, Prelude, Value, VariableMap};
//!
//! # fn main() -> anyhow::Result<()> {
//! let program = r#"
//!     ## The interpreter supports all parser features, including
//!     ## function definitions, tuples and blocks.
//!     order = |x, y| (min(x, y), max(x, y));
//!     assert(order(0.5, -1) == (-1, 0.5));
//!     (_, M) = order(3^2, { x = 3; x + 5 });
//!     M
//! "#;
//! let program = F32Grammar::parse_statements(program)?;
//!
//! let mut env = Environment::new();
//! // Add some native functions to the environment.
//! env.extend(&Prelude).extend(&Comparisons);
//!
//! // To execute statements, we first compile them into a module.
//! let module = env.compile_module("test", &program)?;
//! // Then, the module can be run.
//! assert_eq!(module.run()?, Value::Number(9.0));
//! # Ok(())
//! # }
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
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
    error::{Error, ErrorKind, EvalResult},
    executable::{ExecutableModule, ExecutableModuleBuilder, ModuleImports},
    module_id::{IndexedId, ModuleId, WildcardId},
    values::{CallContext, Function, InterpretedFn, NativeFn, SpannedValue, Value, ValueType},
    variable_map::{Comparisons, Prelude, VariableMap},
};

use hashbrown::HashMap;
use num_complex::{Complex32, Complex64};
use num_traits::Pow;

use core::ops;

use arithmetic_parser::{grammars::NumLiteral, Grammar};

mod compiler;
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
impl Number for Complex32 {}
impl Number for Complex64 {}

/// Environment containing named variables.
///
/// Note that the environment implements [`Index`] / [`IndexMut`] traits, which allows to eloquently
/// access or modify environment.
///
/// [`Index`]: https://doc.rust-lang.org/std/ops/trait.Index.html
/// [`IndexMut`]: https://doc.rust-lang.org/std/ops/trait.IndexMut.html
#[derive(Debug)]
pub struct Environment<'a, T: Grammar> {
    variables: HashMap<String, Value<'a, T>>,
}

impl<T: Grammar> Default for Environment<'_, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Grammar> Clone for Environment<'_, T> {
    fn clone(&self) -> Self {
        Self {
            variables: self.variables.clone(),
        }
    }
}

impl<T: Grammar> PartialEq for Environment<'_, T>
where
    T::Lit: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.variables == other.variables
    }
}

impl<'a, T: Grammar> Environment<'a, T> {
    /// Creates a new environment.
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }

    /// Adds all values from the specified container into the interpreter.
    pub fn extend<V>(&mut self, source: &V) -> &mut Self
    where
        V: VariableMap<'a, T> + ?Sized,
    {
        for (name, value) in source.variables() {
            self.insert(name, value);
        }
        self
    }

    /// Gets a variable by name.
    pub fn get(&self, name: &str) -> Option<&Value<'a, T>> {
        self.variables.get(name)
    }

    /// Gets a mutable reference to the variable by name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut Value<'a, T>> {
        self.variables.get_mut(name)
    }

    /// Iterates over variables.
    pub fn variables(&self) -> impl Iterator<Item = (&str, &Value<'a, T>)> + '_ {
        self.variables
            .iter()
            .map(|(name, value)| (name.as_str(), value))
    }

    /// Inserts a variable with the specified name.
    pub fn insert(&mut self, name: &str, value: Value<'a, T>) -> &mut Self {
        self.variables.insert(name.to_owned(), value);
        self
    }

    /// Inserts a native function with the specified name.
    pub fn insert_native_fn(
        &mut self,
        name: &str,
        native_fn: impl NativeFn<T> + 'static,
    ) -> &mut Self {
        self.insert(name, Value::native_fn(native_fn))
    }

    /// Inserts a [wrapped function] with the specified name.
    ///
    /// Calling this method is equivalent to [`wrap`]ping a function and calling
    /// [`insert_native_fn()`](#method.insert_native_fn) on it. Thanks to type inference magic,
    /// the Rust compiler will usually be able to extract the `Args` type param
    /// from the function definition, provided that type of function arguments and its return type
    /// are defined explicitly or can be unequivocally inferred from the declaration.
    ///
    /// [wrapped function]: fns/struct.FnWrapper.html
    /// [`wrap`]: fns/fn.wrap.html
    pub fn insert_wrapped_fn<Args, F>(&mut self, name: &str, fn_to_wrap: F) -> &mut Self
    where
        fns::FnWrapper<Args, F>: NativeFn<T> + 'static,
    {
        let wrapped = fns::wrap::<Args, _>(fn_to_wrap);
        self.insert(name, Value::native_fn(wrapped))
    }
}

impl<'a, T: Grammar> ops::Index<&str> for Environment<'a, T> {
    type Output = Value<'a, T>;

    fn index(&self, index: &str) -> &Self::Output {
        self.get(index)
            .unwrap_or_else(|| panic!("Variable `{}` is not defined", index))
    }
}

impl<'a, T: Grammar> ops::IndexMut<&str> for Environment<'a, T> {
    fn index_mut(&mut self, index: &str) -> &mut Self::Output {
        self.get_mut(index)
            .unwrap_or_else(|| panic!("Variable `{}` is not defined", index))
    }
}
