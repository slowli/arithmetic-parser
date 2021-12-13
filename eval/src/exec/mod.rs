//! [`ExecutableModule`] and related types.

use crate::{
    compiler::{Compiler, ImportSpans},
    env::Environment,
    error::{Backtrace, Error, ErrorKind, ErrorWithBacktrace},
    Value,
};
use arithmetic_parser::{grammars::Grammar, Block, MaybeSpanned, StripCode};

mod command;
mod module_id;
mod registers;

pub use self::module_id::{IndexedId, ModuleId, WildcardId};
pub(crate) use self::{
    command::{Atom, Command, CompiledExpr, FieldName, SpannedAtom},
    registers::{Executable, ExecutableFn, Operations, Registers},
};
pub use crate::compiler::CompilerExt;

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
/// The lifetime of a module depends on the lifetime of the code, but this dependency
/// can be eliminated via [`StripCode`] implementation.
///
/// # Examples
///
/// ## Basic usage
///
/// ```
/// use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// use arithmetic_eval::{env, fns, Environment, ExecutableModule, Value};
/// # use core::iter::FromIterator;
/// # use hashbrown::HashSet;
///
/// # fn main() -> anyhow::Result<()> {
/// let module = Untyped::<F32Grammar>::parse_statements("fold(xs, -INFINITY, max)")?;
/// let module = ExecutableModule::new("test", &module)?;
///
/// let mut env = Environment::new();
/// env.extend(env::Prelude.iter());
/// env.extend(env::Comparisons.iter());
/// env.insert("INFINITY", Value::Prim(f32::INFINITY)).insert("xs", Value::void());
///
/// // With the original imports, the returned value is `-INFINITY`.
/// assert_eq!(module.with_env(&env)?.run()?, Value::Prim(f32::NEG_INFINITY));
///
/// // Imports can be changed. Let's check that `xs` is indeed an import.
/// assert!(module.is_import("fold"));
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
/// # use core::iter::FromIterator;
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
/// `run_in_env` modifies the environment even if an error occurs during execution:
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{env::Assertions, Environment, ExecutableModule, Value};
/// # use core::iter::FromIterator;
/// # fn main() -> anyhow::Result<()> {
/// let module = Untyped::<F32Grammar>::parse_statements("x = 5; assert_eq(x, 4);")?;
/// let module = ExecutableModule::new("test", &module)?;
///
/// let mut env = Environment::new();
/// env.extend(Assertions.iter());
/// assert!(module.with_mutable_env(&mut env)?.run().is_err());
/// assert_eq!(env["x"], Value::Prim(5.0));
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct ExecutableModule<'a, T> {
    inner: Executable<'a, T>,
    imports: ModuleImports<'a, T>,
}

impl<T: Clone> Clone for ExecutableModule<'_, T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            imports: self.imports.clone(),
        }
    }
}

impl<T: 'static + Clone> StripCode for ExecutableModule<'_, T> {
    type Stripped = ExecutableModule<'static, T>;

    fn strip_code(self) -> Self::Stripped {
        ExecutableModule {
            inner: self.inner.strip_code(),
            imports: self.imports.strip_code(),
        }
    }
}

impl<'a, T> ExecutableModule<'a, T> {
    /// Creates a new module.
    pub fn new<G, Id>(id: Id, block: &Block<'a, G>) -> Result<Self, Error<'a>>
    where
        Id: ModuleId,
        G: Grammar<'a, Lit = T>,
    {
        Compiler::compile_module(id, block)
    }

    pub(crate) fn from_parts(
        inner: Executable<'a, T>,
        imports: Registers<'a, T>,
        import_spans: ImportSpans<'a>,
    ) -> Self {
        Self {
            inner,
            imports: ModuleImports {
                inner: imports,
                spans: import_spans,
            },
        }
    }

    /// Gets the identifier of this module.
    pub fn id(&self) -> &dyn ModuleId {
        self.inner.id()
    }

    /// Returns a shared reference to imports of this module.
    pub fn import_names(&self) -> impl Iterator<Item = &str> + '_ {
        self.imports.inner.variables().map(|(name, _)| name)
    }

    /// Checks if the specified variable is an import.
    pub fn is_import(&self, name: &str) -> bool {
        self.imports.inner.variables_map().contains_key(name)
    }

    /// Combines this module with the specified [`Environment`]. The environment must contain
    /// all module imports; otherwise, an error will be raised.
    ///
    /// # Errors
    ///
    /// Returns an error if the environment does not contain all variables imported by this module.
    pub fn with_env<'s>(
        &'s self,
        env: &'s Environment<'a, T>,
    ) -> Result<WithEnvironment<'s, 'a, T>, Error<'a>> {
        self.check_imports(env)?;
        Ok(WithEnvironment {
            module: self,
            env: Reference::Shared(env),
        })
    }

    fn check_imports(&self, env: &Environment<'a, T>) -> Result<(), Error<'a>> {
        for (name, span) in self.imports.spanned_iter() {
            if !env.contains(name) {
                let err = ErrorKind::Undefined(name.into());
                return Err(Error::new(self.inner.id(), span, err));
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
        env: &'s mut Environment<'a, T>,
    ) -> Result<WithEnvironment<'s, 'a, T>, Error<'a>> {
        self.check_imports(env)?;
        Ok(WithEnvironment {
            module: self,
            env: Reference::Mutable(env),
        })
    }
}

impl<'a, T: 'static + Clone> ExecutableModule<'a, T> {
    fn run_with_registers(
        &self,
        registers: &mut Registers<'a, T>,
        operations: Operations<'_, T>,
    ) -> Result<Value<'a, T>, ErrorWithBacktrace<'a>> {
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

/// Container for an [`ExecutableModule`] together with an [`OrdArithmetic`].
#[derive(Debug)]
pub struct WithEnvironment<'env, 'a, T> {
    module: &'env ExecutableModule<'a, T>,
    env: Reference<'env, Environment<'a, T>>,
}

impl<'a, T: 'static + Clone> WithEnvironment<'_, 'a, T> {
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
    pub fn run(self) -> Result<Value<'a, T>, ErrorWithBacktrace<'a>> {
        let mut registers = self.module.imports.inner.clone();
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

/// Imports of an [`ExecutableModule`].
#[derive(Debug, Clone)]
struct ModuleImports<'a, T> {
    inner: Registers<'a, T>,
    spans: ImportSpans<'a>,
}

impl<'a, T> ModuleImports<'a, T> {
    fn spanned_iter(&self) -> impl Iterator<Item = (&str, &MaybeSpanned<'a>)> + '_ {
        let iter = self.inner.variables_map().iter();
        iter.map(move |(name, idx)| (name.as_str(), &self.spans[*idx]))
    }
}

impl<T: 'static + Clone> StripCode for ModuleImports<'_, T> {
    type Stripped = ModuleImports<'static, T>;

    fn strip_code(self) -> Self::Stripped {
        ModuleImports {
            inner: self.inner.strip_code(),
            spans: self.spans.into_iter().map(StripCode::strip_code).collect(),
        }
    }
}
