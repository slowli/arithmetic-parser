//! Executables output by a `Compiler` and related types.

use core::{fmt, ops};

use crate::{
    alloc::{Box, String, ToOwned},
    arith::{OrdArithmetic, StdArithmetic},
    compiler::{Compiler, ImportSpans},
    error::{Backtrace, ErrorWithBacktrace},
    Environment, Error, ErrorKind, Value, VariableMap,
};
use arithmetic_parser::{grammars::Grammar, Block, StripCode};

mod command;
mod module_id;
mod registers;

pub use self::module_id::{IndexedId, ModuleId, WildcardId};
pub(crate) use self::{
    command::{Atom, Command, CompiledExpr, SpannedAtom},
    registers::{Executable, ExecutableFn, Registers},
};

/// Executable module together with its imports.
///
/// An `ExecutableModule` is a result of compiling a `Block` of statements. A module can *import*
/// [`Value`]s, such as [commonly used functions](crate::fns). Importing is performed
/// when building the module.
///
/// After the module is created, it can be [`run`](Self::run). If the last statement of the block
/// is an expression (that is, not terminated with a `;`), it is the result of the execution;
/// otherwise, the result is [`Value::void()`]. It is possible to run a module multiple times
/// and to change imports by using [`Self::set_import()`].
///
/// In some cases (e.g., when building a REPL) it is useful to get not only the outcome
/// of the module execution, but the intermediate results as well. Use [`Self::run_in_env()`]
/// for such cases.
///
/// # Examples
///
/// ## Basic usage
///
/// ```
/// use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// use arithmetic_eval::{fns, Comparisons, ExecutableModule, Prelude, Value};
/// # use core::{f32, iter::FromIterator};
/// # use hashbrown::HashSet;
///
/// # fn main() -> anyhow::Result<()> {
/// let module = Untyped::<F32Grammar>::parse_statements("xs.fold(-INFINITY, max)")?;
/// let mut module = ExecutableModule::builder("test", &module)?
///     .with_imports_from(&Prelude)
///     .with_imports_from(&Comparisons)
///     .with_import("INFINITY", Value::Number(f32::INFINITY))
///     // Set remaining imports to a fixed value.
///     .set_imports(|_| Value::void());
///
/// // With the original imports, the returned value is `-INFINITY`.
/// assert_eq!(module.run()?, Value::Number(f32::NEG_INFINITY));
///
/// // Imports can be changed. Let's check that `xs` is indeed an import.
/// assert!(module.imports().contains("xs"));
/// // ...or even
/// assert!(module.imports()["fold"].is_function());
/// // It's possible to iterate over imports, too.
/// let imports = module.imports().iter()
///     .map(|(name, _)| name)
///     .collect::<HashSet<_>>();
/// assert!(imports.is_superset(&HashSet::from_iter(vec!["max", "fold"])));
/// # drop(imports); // necessary to please the borrow checker
///
/// // Change the `xs` import and run the module again.
/// let array = [1.0, -3.0, 2.0, 0.5].iter().copied()
///     .map(Value::Number)
///     .collect();
/// module.set_import("xs", Value::Tuple(array));
/// assert_eq!(module.run()?, Value::Number(2.0));
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
/// let mut module = ExecutableModule::builder("test", &block)?
///     .with_import("x", Value::Number(3.0))
///     .with_import("y", Value::Number(5.0))
///     .build();
/// assert_eq!(module.run()?, Value::Number(8.0));
///
/// let mut env = Environment::from_iter(module.imports());
/// env.insert("x", Value::Number(-1.0));
/// assert_eq!(module.run_in_env(&mut env)?, Value::Number(4.0));
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
/// # use arithmetic_eval::{Environment, ExecutableModule, Prelude, Value};
/// # use core::iter::FromIterator;
/// # fn main() -> anyhow::Result<()> {
/// let module = Untyped::<F32Grammar>::parse_statements("x = 5; assert_eq(x, 4);")?;
/// let module = ExecutableModule::builder("test", &module)?
///     .with_imports_from(&Prelude)
///     .build();
///
/// let mut env = Environment::from_iter(module.imports());
/// assert!(module.run_in_env(&mut env).is_err());
/// assert_eq!(env["x"], Value::Number(5.0));
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
    pub(crate) fn from_parts(inner: Executable<'a, T>, imports: Registers<'a, T>) -> Self {
        Self {
            inner,
            imports: ModuleImports { inner: imports },
        }
    }

    /// Gets the identifier of this module.
    pub fn id(&self) -> &dyn ModuleId {
        self.inner.id()
    }

    /// Sets the value of an imported variable.
    ///
    /// # Panics
    ///
    /// Panics if the variable with the specified name is not an import. Check
    /// that the import exists beforehand via [`imports().contains()`] if this is
    /// unknown at compile time.
    ///
    /// [`imports().contains()`]: ModuleImports::contains()
    pub fn set_import(&mut self, name: &str, value: Value<'a, T>) -> &mut Self {
        self.imports.inner.set_var(name, value);
        self
    }

    /// Returns shared reference to imports of this module.
    pub fn imports(&self) -> &ModuleImports<'a, T> {
        &self.imports
    }

    /// Combines this module with the specified `arithmetic`.
    pub fn with_arithmetic<'s>(
        &'s self,
        arithmetic: &'s dyn OrdArithmetic<T>,
    ) -> WithArithmetic<'s, 'a, T> {
        WithArithmetic {
            module: self,
            arithmetic,
        }
    }
}

impl<'a, T: Clone + fmt::Debug> ExecutableModule<'a, T> {
    /// Starts building a new module.
    pub fn builder<G, Id>(
        id: Id,
        block: &Block<'a, G>,
    ) -> Result<ExecutableModuleBuilder<'a, T>, Error<'a>>
    where
        Id: ModuleId,
        G: Grammar<Lit = T>,
    {
        let (module, import_spans) = Compiler::compile_module(id, block)?;
        Ok(ExecutableModuleBuilder::new(module, import_spans))
    }

    fn run_with_registers(
        &self,
        registers: &mut Registers<'a, T>,
        arithmetic: &dyn OrdArithmetic<T>,
    ) -> Result<Value<'a, T>, ErrorWithBacktrace<'a>> {
        let mut backtrace = Backtrace::default();
        registers
            .execute(&self.inner, arithmetic, Some(&mut backtrace))
            .map_err(|err| ErrorWithBacktrace::new(err, backtrace))
    }
}

impl<'a, T: Clone + fmt::Debug> ExecutableModule<'a, T>
where
    StdArithmetic: OrdArithmetic<T>,
{
    /// Runs the module with the current values of imports. This is a read-only operation;
    /// neither the imports, nor other module state are modified by it.
    pub fn run(&self) -> Result<Value<'a, T>, ErrorWithBacktrace<'a>> {
        self.with_arithmetic(&StdArithmetic).run()
    }

    /// Runs the module with the specified [`Environment`]. The environment may contain some of
    /// module imports; they will be used to override imports defined in the module.
    ///
    /// On execution, the environment is modified to reflect assignments in the topmost scope
    /// of the module. The modification takes place regardless of whether or not the execution
    /// succeeds. That is, if an error occurs, all preceding assignments in the topmost scope
    /// still take place. See [the relevant example](#behavior-on-errors).
    pub fn run_in_env(
        &self,
        env: &mut Environment<'a, T>,
    ) -> Result<Value<'a, T>, ErrorWithBacktrace<'a>> {
        self.with_arithmetic(&StdArithmetic).run_in_env(env)
    }
}

/// Container for an [`ExecutableModule`] together with an [`OrdArithmetic`].
#[derive(Debug)]
pub struct WithArithmetic<'r, 'a, T> {
    module: &'r ExecutableModule<'a, T>,
    arithmetic: &'r dyn OrdArithmetic<T>,
}

impl<T> Clone for WithArithmetic<'_, '_, T> {
    fn clone(&self) -> Self {
        Self {
            module: self.module,
            arithmetic: self.arithmetic,
        }
    }
}

impl<T> Copy for WithArithmetic<'_, '_, T> {}

impl<'a, T> WithArithmetic<'_, 'a, T>
where
    T: Clone + fmt::Debug,
{
    /// Runs the module with the previously provided [`OrdArithmetic`] and the current values
    /// of imports.
    ///
    /// See [`ExecutableModule::run()`] for more details.
    pub fn run(self) -> Result<Value<'a, T>, ErrorWithBacktrace<'a>> {
        let mut registers = self.module.imports.inner.clone();
        self.module
            .run_with_registers(&mut registers, self.arithmetic)
    }

    /// Runs the module with the specified [`Environment`]. The environment may contain some of
    /// module imports; they will be used to override imports defined in the module.
    ///
    /// See [`ExecutableModule::run_in_env()`] for more details.
    pub fn run_in_env(
        self,
        env: &mut Environment<'a, T>,
    ) -> Result<Value<'a, T>, ErrorWithBacktrace<'a>> {
        let mut registers = self.module.imports.inner.clone();
        registers.update_from_env(env);

        let result = self
            .module
            .run_with_registers(&mut registers, self.arithmetic);
        registers.update_env(env);
        result
    }
}

/// Imports of an [`ExecutableModule`].
///
/// Note that imports implement [`Index`](ops::Index) trait, which allows to eloquently
/// get imports by name.
#[derive(Debug)]
pub struct ModuleImports<'a, T> {
    inner: Registers<'a, T>,
}

impl<T: Clone> Clone for ModuleImports<'_, T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T: 'static + Clone> StripCode for ModuleImports<'_, T> {
    type Stripped = ModuleImports<'static, T>;

    fn strip_code(self) -> Self::Stripped {
        ModuleImports {
            inner: self.inner.strip_code(),
        }
    }
}

impl<'a, T> ModuleImports<'a, T> {
    /// Checks if the imports contain a variable with the specified name.
    pub fn contains(&self, name: &str) -> bool {
        self.inner.variables_map().contains_key(name)
    }

    /// Gets the current value of the import with the specified name, or `None` if the import
    /// is not defined.
    pub fn get(&self, name: &str) -> Option<&Value<'a, T>> {
        self.inner.get_var(name)
    }

    /// Iterates over imported variables.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value<'a, T>)> + '_ {
        self.inner.variables()
    }
}

impl<'a, T> ops::Index<&str> for ModuleImports<'a, T> {
    type Output = Value<'a, T>;

    fn index(&self, index: &str) -> &Self::Output {
        self.inner
            .get_var(index)
            .unwrap_or_else(|| panic!("Import `{}` is not defined", index))
    }
}

impl<'a, T: Clone + 'a> IntoIterator for ModuleImports<'a, T> {
    type Item = (String, Value<'a, T>);
    type IntoIter = Box<dyn Iterator<Item = Self::Item> + 'a>;

    fn into_iter(self) -> Self::IntoIter {
        Box::new(self.inner.into_variables())
    }
}

impl<'a, 'r, T> IntoIterator for &'r ModuleImports<'a, T> {
    type Item = (&'r str, &'r Value<'a, T>);
    type IntoIter = Box<dyn Iterator<Item = Self::Item> + 'r>;

    fn into_iter(self) -> Self::IntoIter {
        Box::new(self.iter())
    }
}

/// Builder for an `ExecutableModule`.
///
/// The builder can be created via [`ExecutableModule::builder()`]. See [`ExecutableModule`] docs
/// for the examples of usage.
#[derive(Debug)]
pub struct ExecutableModuleBuilder<'a, T> {
    module: ExecutableModule<'a, T>,
    undefined_imports: ImportSpans<'a>,
}

impl<'a, T> ExecutableModuleBuilder<'a, T> {
    fn new(module: ExecutableModule<'a, T>, undefined_imports: ImportSpans<'a>) -> Self {
        Self {
            module,
            undefined_imports,
        }
    }

    /// Checks if all necessary imports are defined for this module.
    pub fn has_undefined_imports(&self) -> bool {
        self.undefined_imports.is_empty()
    }

    /// Iterates over the names of undefined imports.
    pub fn undefined_imports(&self) -> impl Iterator<Item = &str> + '_ {
        self.undefined_imports.keys().map(String::as_str)
    }

    /// Adds a single import. If the specified variable is not an import, does nothing.
    pub fn with_import(mut self, name: &str, value: Value<'a, T>) -> Self {
        if self.module.imports.contains(name) {
            self.module.set_import(name, value);
        }
        self.undefined_imports.remove(name);
        self
    }

    /// Sets undefined imports from the specified source. Imports defined previously and present
    /// in the source are **not** overridden.
    pub fn with_imports_from<V>(mut self, source: &V) -> Self
    where
        V: VariableMap<'a, T> + ?Sized,
    {
        let module = &mut self.module;
        self.undefined_imports.retain(|var_name, _| {
            source.get_variable(var_name).map_or(true, |value| {
                module.set_import(var_name, value);
                false
            })
        });
        self
    }

    /// Tries to build this module.
    ///
    /// # Errors
    ///
    /// Fails if this module has at least one undefined import. In this case, the returned error
    /// highlights one of such imports.
    pub fn try_build(self) -> Result<ExecutableModule<'a, T>, Error<'a>> {
        if let Some((var_name, span)) = self.undefined_imports.iter().next() {
            let err = ErrorKind::Undefined(var_name.to_owned());
            Err(Error::new(self.module.id(), span, err))
        } else {
            Ok(self.module)
        }
    }

    /// A version of [`Self::try_build()`] that panics if there are undefined imports.
    pub fn build(self) -> ExecutableModule<'a, T> {
        self.try_build().unwrap()
    }

    /// Sets the undefined imports using the provided closure and returns the resulting module.
    /// The closure is called with the name of each undefined import and should return
    /// the corresponding [`Value`].
    pub fn set_imports<F>(mut self, mut setter: F) -> ExecutableModule<'a, T>
    where
        F: FnMut(&str) -> Value<'a, T>,
    {
        for var_name in self.undefined_imports.keys() {
            self.module.set_import(var_name, setter(var_name));
        }
        self.module
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{compiler::Compiler, WildcardId};

    use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};

    #[test]
    fn cloning_module() {
        let block = "y = x + 2 * (x + 1) + 1; y";
        let block = Untyped::<F32Grammar>::parse_statements(block).unwrap();
        let (mut module, _) = Compiler::compile_module(WildcardId, &block).unwrap();

        let mut module_copy = module.clone();
        module_copy.set_import("x", Value::Number(10.0));
        let value = module_copy.run().unwrap();
        assert_eq!(value, Value::Number(33.0));

        module.set_import("x", Value::Number(5.0));
        let value = module.run().unwrap();
        assert_eq!(value, Value::Number(18.0));
    }
}
