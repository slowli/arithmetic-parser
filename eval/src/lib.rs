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
//! use arithmetic_eval::{fns, Interpreter, Value};
//!
//! const MIN: fns::Binary<f32> =
//!     fns::Binary::new(|x, y| if x < y { x } else { y });
//! const MAX: fns::Binary<f32> =
//!     fns::Binary::new(|x, y| if x > y { x } else { y });
//!
//! let mut context = Interpreter::with_prelude();
//! // Add some native functions to the interpreter.
//! context
//!     .insert_native_fn("min", MIN)
//!     .insert_native_fn("max", MAX);
//!
//! let program = r#"
//!     ## The interpreter supports all parser features, including
//!     ## function definitions, tuples and blocks.
//!     order = |x, y| (min(x, y), max(x, y));
//!     assert(order(0.5, -1) == (-1, 0.5));
//!     (_, M) = order(3^2, { x = 3; x + 5 });
//!     M
//! "#;
//! let program = F32Grammar::parse_statements(program).unwrap();
//! let ret = context.evaluate(&program).unwrap();
//! assert_eq!(ret, Value::Number(9.0));
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
    error::{Error, ErrorKind, EvalResult, InterpreterError},
    executable::{ExecutableModule, ExecutableModuleBuilder, ModuleImports},
    module_id::{IndexedId, ModuleId, WildcardId},
    values::{CallContext, Function, InterpretedFn, NativeFn, SpannedValue, Value, ValueType},
    variable_map::{Comparisons, Prelude, VariableMap},
};

use num_complex::{Complex32, Complex64};
use num_traits::Pow;

use core::ops;

use crate::{
    compiler::{CompilationOptions, Compiler},
    error::{Backtrace, ErrorWithBacktrace},
    executable::Env,
};
use arithmetic_parser::{grammars::NumLiteral, Block, Grammar, StripCode};

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

/// Simple interpreter for arithmetic expressions.
#[derive(Debug)]
pub struct Interpreter<'a, T: Grammar> {
    env: Env<'a, T>,
}

impl<T: Grammar> Default for Interpreter<'_, T>
where
    T::Lit: Number,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Grammar> Clone for Interpreter<'_, T> {
    fn clone(&self) -> Self {
        Self {
            env: self.env.clone(),
        }
    }
}

impl<'a, T> Interpreter<'a, T>
where
    T: Grammar,
{
    /// Creates a new interpreter.
    pub fn new() -> Self {
        Self { env: Env::new() }
    }

    /// Adds all values from the specified container into the interpreter.
    pub fn with<V>(mut self, source: &V) -> Self
    where
        V: VariableMap<'a, T> + ?Sized,
    {
        for (name, value) in source.variables() {
            self.insert_var(name, value);
        }
        self
    }

    /// Gets a variable by name.
    pub fn get_var(&self, name: &str) -> Option<&Value<'a, T>> {
        self.env.get_var(name)
    }

    /// Iterates over variables.
    pub fn variables(&self) -> impl Iterator<Item = (&str, &Value<'a, T>)> + '_ {
        self.env.variables()
    }

    /// Inserts a variable with the specified name.
    pub fn insert_var(&mut self, name: &str, value: Value<'a, T>) -> &mut Self {
        self.env.push_var(name, value);
        self
    }

    /// Inserts a native function with the specified name.
    pub fn insert_native_fn(
        &mut self,
        name: &str,
        native_fn: impl NativeFn<T> + 'static,
    ) -> &mut Self {
        self.insert_var(name, Value::native_fn(native_fn))
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
        self.insert_var(name, Value::native_fn(wrapped))
    }
}

impl<'a, T> Interpreter<'a, T>
where
    T: Grammar,
    T::Lit: Number,
{
    /// Shortcut for `new().with(&Prelude)`.
    pub fn with_prelude() -> Self {
        Self::new().with(&Prelude)
    }

    fn evaluate_module(
        &mut self,
        module: &ExecutableModule<'a, T>,
    ) -> Result<Value<'a, T>, ErrorWithBacktrace<'a>> {
        let mut backtrace = Backtrace::default();
        let result = self
            .env
            .execute(module.inner(), Some(&mut backtrace))
            .map_err(|err| ErrorWithBacktrace::new(err, backtrace));
        self.env.compress();
        result
    }

    /// Evaluates a list of statements.
    pub fn evaluate(
        &mut self,
        block: &Block<'a, T>,
    ) -> Result<Value<'a, T>, InterpreterError<'a, 'a>> {
        let module =
            Compiler::compile_module(WildcardId, &self.env, block, CompilationOptions::Embedded)
                .map_err(InterpreterError::Compile)?;
        self.evaluate_module(&module)
            .map_err(InterpreterError::Evaluate)
    }

    /// Compiles the provided block, returning the compiled module and its imports.
    /// The imports can the be changed to run the module with different params.
    pub fn compile<F: ModuleId>(
        &self,
        id: F,
        program: &Block<'a, T>,
    ) -> Result<ExecutableModule<'a, T>, Error<'a>> {
        let options = CompilationOptions::Standalone {
            create_imports: false,
        };
        Compiler::compile_module(id, &self.env, program, options)
    }
}

impl<T> Interpreter<'static, T>
where
    T: Grammar,
    T::Lit: Number,
{
    /// Evaluates a named block of statements.
    ///
    /// Unlike [`evaluate`], this method strips code spans from the compiled statements right away.
    /// This allows to operate an `Interpreter` with a static lifetime (i.e., not tied to
    /// the lifetime of the code). As a downside, if code spans are required in certain situations
    /// (e.g., for error reporting), they should be handled separately.
    ///
    /// [`evaluate`]: #method.evaluate
    pub fn evaluate_named_block<'bl, F: ModuleId>(
        &mut self,
        id: F,
        block: &Block<'bl, T>,
    ) -> Result<Value<'static, T>, InterpreterError<'bl, 'static>> {
        let module = Compiler::compile_module(id, &self.env, block, CompilationOptions::Embedded)
            .map_err(InterpreterError::Compile)?;
        self.evaluate_module(&module.strip_code())
            .map_err(InterpreterError::Evaluate)
    }
}
