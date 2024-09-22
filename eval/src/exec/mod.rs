//! [`ExecutableModule`] and related types.

use core::fmt;

use arithmetic_parser::{grammars::Grammar, Block};

pub use self::module_id::{IndexedId, ModuleId, WildcardId};
pub(crate) use self::{
    command::{Atom, Command, CompiledExpr, FieldName, LocatedAtom},
    registers::{Executable, ExecutableFn, Operations, Registers},
};
pub use crate::compiler::CompilerExt;
use crate::{
    alloc::Arc,
    compiler::{Captures, Compiler},
    env::Environment,
    error::{Backtrace, Error, ErrorKind, ErrorWithBacktrace},
    Value,
};

mod command;
mod module_id;
mod registers;

/// Executable module together with its imports.
///
/// An `ExecutableModule` is a result of compiling a `Block` of statements. A module can *import*
/// [`Value`]s, such as [commonly used functions](crate::fns). Importing is performed
/// when building the module.
///
/// After the module is created, it can be associated with an environment via [`Self::with_env()`]
/// and [`run`](WithEnvironment::run()).
/// If the last statement of the block is an expression (that is, not terminated with a `;`),
/// it is the result of the execution; otherwise, the result is [`Value::void()`].
///
/// In some cases (e.g., when building a REPL) it is useful to get not only the outcome
/// of the module execution, but the intermediate results as well. Use [`Self::with_mutable_env()`]
/// for such cases.
///
/// `ExecutableModule`s are generic with respect to the primitive value type, just like [`Value`].
///
/// # Examples
///
/// ## Basic usage
///
/// ```
/// use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// use arithmetic_eval::{env, fns, Environment, ExecutableModule, Value};
/// # use std::collections::HashSet;
///
/// # fn main() -> anyhow::Result<()> {
/// let module = Untyped::<F32Grammar>::parse_statements(
///     "xs.fold(-INFINITY, max)",
/// )?;
/// let module = ExecutableModule::new("test", &module)?;
///
/// let mut env = Environment::new();
/// env.insert("INFINITY", Value::Prim(f32::INFINITY))
///     .insert("xs", Value::void())
///     .extend(env::Prelude::iter().chain(env::Comparisons::iter()));
///
/// // With the original imports, the returned value is `-INFINITY`.
/// assert_eq!(module.with_env(&env)?.run()?, Value::Prim(f32::NEG_INFINITY));
///
/// // Imports can be changed. Let's check that `xs` is indeed an import.
/// assert!(module.is_import("xs"));
/// // It's possible to iterate over imports, too.
/// let imports = module.import_names().collect::<HashSet<_>>();
/// assert!(imports.is_superset(&HashSet::from_iter(vec!["max", "fold"])));
/// # drop(imports); // necessary to please the borrow checker
///
/// // Change the `xs` import and run the module again.
/// let array = [1.0, -3.0, 2.0, 0.5].iter().copied()
///     .map(Value::Prim)
///     .collect();
/// env.insert("xs", Value::Tuple(array));
/// assert_eq!(module.with_env(&env)?.run()?, Value::Prim(2.0));
/// # Ok(())
/// # }
/// ```
///
/// ## Reusing a module
///
/// The same module can be run with multiple imports:
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{Environment, ExecutableModule, Value};
/// # fn main() -> anyhow::Result<()> {
/// let block = Untyped::<F32Grammar>::parse_statements("x + y")?;
/// let module = ExecutableModule::new("test", &block)?;
///
/// let mut env = Environment::new();
/// env.insert("x", Value::Prim(3.0)).insert("y", Value::Prim(5.0));
/// assert_eq!(module.with_env(&env)?.run()?, Value::Prim(8.0));
///
/// env.insert("x", Value::Prim(-1.0));
/// assert_eq!(module.with_env(&env)?.run()?, Value::Prim(4.0));
/// # Ok(())
/// # }
/// ```
///
/// ## Behavior on errors
///
/// [`Self::with_mutable_env()`] modifies the environment even if an error occurs during execution:
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{env::Assertions, Environment, ExecutableModule, Value};
/// # fn main() -> anyhow::Result<()> {
/// let module = Untyped::<F32Grammar>::parse_statements("x = 5; assert_eq(x, 4);")?;
/// let module = ExecutableModule::new("test", &module)?;
///
/// let mut env = Environment::new();
/// env.extend(Assertions::iter());
/// assert!(module.with_mutable_env(&mut env)?.run().is_err());
/// assert_eq!(env["x"], Value::Prim(5.0));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct ExecutableModule<T> {
    inner: Executable<T>,
    captures: Captures,
}

#[cfg(test)]
static_assertions::assert_impl_all!(ExecutableModule<f32>: Send, Sync);

impl<T: Clone + fmt::Debug> ExecutableModule<T> {
    /// Creates a new module.
    pub fn new<G, Id>(id: Id, block: &Block<'_, G>) -> Result<Self, Error>
    where
        Id: ModuleId,
        G: Grammar<Lit = T>,
    {
        Compiler::compile_module(id, block)
    }
}

impl<T> ExecutableModule<T> {
    pub(crate) fn from_parts(inner: Executable<T>, captures: Captures) -> Self {
        Self { inner, captures }
    }

    /// Gets the identifier of this module.
    pub fn id(&self) -> &Arc<dyn ModuleId> {
        self.inner.id()
    }

    /// Returns a shared reference to imports of this module.
    pub fn import_names(&self) -> impl Iterator<Item = &str> + '_ {
        self.captures.iter().map(|(name, _)| name)
    }

    /// Checks if the specified variable is an import.
    pub fn is_import(&self, name: &str) -> bool {
        self.captures.contains(name)
    }

    /// Combines this module with the specified [`Environment`]. The environment must contain
    /// all module imports; otherwise, an error will be raised.
    ///
    /// # Errors
    ///
    /// Returns an error if the environment does not contain all variables imported by this module.
    pub fn with_env<'s>(
        &'s self,
        env: &'s Environment<T>,
    ) -> Result<WithEnvironment<'s, T>, Error> {
        self.check_imports(env)?;
        Ok(WithEnvironment {
            module: self,
            env: Reference::Shared(env),
        })
    }

    fn check_imports(&self, env: &Environment<T>) -> Result<(), Error> {
        for (name, span) in self.captures.iter() {
            if !env.contains(name) {
                let err = ErrorKind::Undefined(name.into());
                return Err(Error::new(self.inner.id().clone(), span, err));
            }
        }
        Ok(())
    }

    /// Analogue of [`Self::with_env()`] that modifies the provided [`Environment`]
    /// when the module is [run](WithEnvironment::run()).
    ///
    /// # Errors
    ///
    /// Returns an error if the environment does not contain all variables imported by this module.
    pub fn with_mutable_env<'s>(
        &'s self,
        env: &'s mut Environment<T>,
    ) -> Result<WithEnvironment<'s, T>, Error> {
        self.check_imports(env)?;
        Ok(WithEnvironment {
            module: self,
            env: Reference::Mutable(env),
        })
    }
}

impl<T: 'static + Clone> ExecutableModule<T> {
    fn run_with_registers(
        &self,
        registers: &mut Registers<T>,
        operations: Operations<'_, T>,
    ) -> Result<Value<T>, ErrorWithBacktrace> {
        let mut backtrace = Backtrace::default();
        registers
            .execute(&self.inner, operations, Some(&mut backtrace))
            .map_err(|err| ErrorWithBacktrace::new(err, backtrace))
    }
}

#[derive(Debug)]
enum Reference<'a, T> {
    Shared(&'a T),
    Mutable(&'a mut T),
}

impl<T> AsRef<T> for Reference<'_, T> {
    fn as_ref(&self) -> &T {
        match self {
            Self::Shared(shared) => shared,
            Self::Mutable(mutable) => mutable,
        }
    }
}

/// Container for an [`ExecutableModule`] together with an [`Environment`].
#[derive(Debug)]
pub struct WithEnvironment<'env, T> {
    module: &'env ExecutableModule<T>,
    env: Reference<'env, Environment<T>>,
}

impl<T: 'static + Clone> WithEnvironment<'_, T> {
    /// Runs the module in the previously provided [`Environment`].
    ///
    /// If a mutable reference was provided to the environment, the environment is modified
    /// to reflect top-level assignments in the module (both new and reassigned variables).
    /// If an error occurs, the assignments are performed up until the error (i.e., the environment
    /// is **not** rolled back on error).
    ///
    /// # Errors
    ///
    /// Returns an error if module execution fails.
    pub fn run(self) -> Result<Value<T>, ErrorWithBacktrace> {
        let mut registers = Registers::from(&self.module.captures);
        registers.update_from_env(self.env.as_ref());
        let result = self
            .module
            .run_with_registers(&mut registers, self.env.as_ref().operations());

        if let Reference::Mutable(env) = self.env {
            registers.update_env(env);
        }
        result
    }
}
