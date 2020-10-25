//! Executables output by a `Compiler` and related types.

use core::ops;

use crate::{
    compiler::{Compiler, ImportSpans},
    error::{Backtrace, ErrorWithBacktrace},
    Environment, Error, ErrorKind, ModuleId, Number, Value, VariableMap,
};
use arithmetic_parser::{Block, Grammar, LvalueLen, MaybeSpanned, StripCode};

mod command;
mod env;

pub(crate) use self::{
    command::{Atom, Command, ComparisonOp, CompiledExpr, SpannedAtom},
    env::{Executable, Registers},
};

#[derive(Debug)]
pub(crate) struct ExecutableFn<'a, T: Grammar> {
    pub inner: Executable<'a, T>,
    pub def_span: MaybeSpanned<'a>,
    pub arg_count: LvalueLen,
}

impl<T: Grammar> ExecutableFn<'_, T> {
    pub fn to_stripped_code(&self) -> ExecutableFn<'static, T> {
        ExecutableFn {
            inner: self.inner.clone().strip_code(),
            def_span: self.def_span.strip_code(),
            arg_count: self.arg_count,
        }
    }
}

impl<T: Grammar> StripCode for ExecutableFn<'_, T> {
    type Stripped = ExecutableFn<'static, T>;

    fn strip_code(self) -> Self::Stripped {
        ExecutableFn {
            inner: self.inner.strip_code(),
            def_span: self.def_span.strip_code(),
            arg_count: self.arg_count,
        }
    }
}

/// Executable module together with its imports.
///
/// # Examples
///
/// ```
/// use arithmetic_parser::{grammars::F32Grammar, GrammarExt};
/// use arithmetic_eval::{fns, Comparisons, ExecutableModule, Prelude, Value, ValueType};
/// # use std::{collections::HashSet, f32, iter::FromIterator};
///
/// # fn main() -> anyhow::Result<()> {
/// let module = F32Grammar::parse_statements("xs.fold(-INFINITY, max)")?;
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
/// let imports: HashSet<_> = module.imports().iter().map(|(name, _)| name).collect();
/// assert!(imports.is_superset(&HashSet::from_iter(vec!["max", "fold", "xs"])));
///
/// // Change the `xs` import and run the module again.
/// let array = [1.0, -3.0, 2.0, 0.5].iter().copied().map(Value::Number).collect();
/// module.set_import("xs", Value::Tuple(array));
/// assert_eq!(module.run()?, Value::Number(2.0));
/// # Ok(())
/// # }
/// ```
///
/// The same module can be run with multiple imports:
///
/// ```
/// # use arithmetic_parser::{grammars::F32Grammar, GrammarExt};
/// # use arithmetic_eval::{Environment, ExecutableModule, Value};
/// # use core::iter::FromIterator;
/// # fn main() -> anyhow::Result<()> {
/// let block = F32Grammar::parse_statements("x + y")?;
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
#[derive(Debug)]
pub struct ExecutableModule<'a, T: Grammar> {
    inner: Executable<'a, T>,
    imports: ModuleImports<'a, T>,
}

impl<T: Grammar> Clone for ExecutableModule<'_, T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            imports: self.imports.clone(),
        }
    }
}

impl<T: Grammar> StripCode for ExecutableModule<'_, T> {
    type Stripped = ExecutableModule<'static, T>;

    fn strip_code(self) -> Self::Stripped {
        ExecutableModule {
            inner: self.inner.strip_code(),
            imports: self.imports.strip_code(),
        }
    }
}

impl<'a, T: Grammar> ExecutableModule<'a, T> {
    pub(crate) fn from_parts(inner: Executable<'a, T>, imports: Registers<'a, T>) -> Self {
        Self {
            inner,
            imports: ModuleImports { inner: imports },
        }
    }

    /// Starts building a new module.
    pub fn builder<Id>(
        id: Id,
        block: &Block<'a, T>,
    ) -> Result<ExecutableModuleBuilder<'a, T>, Error<'a>>
    where
        Id: ModuleId,
    {
        let (module, import_spans) = Compiler::compile_module(id, &Registers::new(), block)?;
        Ok(ExecutableModuleBuilder::new(module, import_spans))
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
    /// [`imports().contains()`]: struct.ModuleImports.html#method.contains
    pub fn set_import(&mut self, name: &str, value: Value<'a, T>) -> &mut Self {
        self.imports.inner.set_var(name, value);
        self
    }

    /// Returns shared reference to imports of this module.
    pub fn imports(&self) -> &ModuleImports<'a, T> {
        &self.imports
    }
}

impl<'a, T: Grammar> ExecutableModule<'a, T>
where
    T::Lit: Number,
{
    /// Runs the module with the current values of imports. The imports are not modified.
    pub fn run(&self) -> Result<Value<'a, T>, ErrorWithBacktrace<'a>> {
        let mut registers = self.imports.inner.clone();
        self.run_with_registers(&mut registers)
    }

    /// Runs the module with the specified `Environment`. The environment may contain some of
    /// module imports; they will be used to override imports defined in the module.
    /// On execution, the environment is modified to reflect assignments in the topmost scope
    /// of the module.
    pub fn run_in_env(
        &self,
        env: &mut Environment<'a, T>,
    ) -> Result<Value<'a, T>, ErrorWithBacktrace<'a>> {
        let mut registers = self.imports.inner.clone();
        registers.update_from_env(env);

        let result = self.run_with_registers(&mut registers);
        registers.update_env(env);
        result
    }

    fn run_with_registers(
        &self,
        registers: &mut Registers<'a, T>,
    ) -> Result<Value<'a, T>, ErrorWithBacktrace<'a>> {
        let mut backtrace = Backtrace::default();
        registers
            .execute(&self.inner, Some(&mut backtrace))
            .map_err(|err| ErrorWithBacktrace::new(err, backtrace))
    }
}

/// Imports of an [`ExecutableModule`].
///
/// Note that imports implement [`Index`] trait, which allows to eloquently get imports by name.
///
/// [`ExecutableModule`]: struct.ExecutableModule.html
/// [`Index`]: https://doc.rust-lang.org/std/ops/trait.Index.html
#[derive(Debug)]
pub struct ModuleImports<'a, T: Grammar> {
    inner: Registers<'a, T>,
}

impl<T: Grammar> Clone for ModuleImports<'_, T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T: Grammar> StripCode for ModuleImports<'_, T> {
    type Stripped = ModuleImports<'static, T>;

    fn strip_code(self) -> Self::Stripped {
        ModuleImports {
            inner: self.inner.strip_code(),
        }
    }
}

impl<'a, T: Grammar> ModuleImports<'a, T> {
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

impl<'a, T: Grammar> ops::Index<&str> for ModuleImports<'a, T> {
    type Output = Value<'a, T>;

    fn index(&self, index: &str) -> &Self::Output {
        self.inner
            .get_var(index)
            .unwrap_or_else(|| panic!("Import `{}` is not defined", index))
    }
}

impl<'a, T: Grammar> IntoIterator for ModuleImports<'a, T> {
    type Item = (String, Value<'a, T>);
    // FIXME: use a more specific type.
    type IntoIter = Box<dyn Iterator<Item = Self::Item> + 'a>;

    fn into_iter(self) -> Self::IntoIter {
        Box::new(self.inner.into_variables())
    }
}

impl<'a, 'r, T: Grammar> IntoIterator for &'r ModuleImports<'a, T> {
    type Item = (&'r str, &'r Value<'a, T>);
    // FIXME: use a more specific type.
    type IntoIter = Box<dyn Iterator<Item = Self::Item> + 'r>;

    fn into_iter(self) -> Self::IntoIter {
        Box::new(self.iter())
    }
}

/// Builder for an `ExecutableModule`.
#[derive(Debug)]
pub struct ExecutableModuleBuilder<'a, T: Grammar> {
    module: ExecutableModule<'a, T>,
    undefined_imports: ImportSpans<'a>,
}

impl<'a, T: Grammar> ExecutableModuleBuilder<'a, T> {
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

    /// Adds a single import. If the specified variable is not an import, does nothing.
    pub fn with_import<V>(mut self, name: &str, value: V) -> Self
    where
        V: Into<Value<'a, T>>,
    {
        if self.module.imports.contains(name) {
            self.module.set_import(name, value.into());
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

    /// A version of [`try_build()`] that panics if there are undefined imports.
    ///
    /// [`try_build()`]: #method.try_build
    pub fn build(self) -> ExecutableModule<'a, T> {
        self.try_build().unwrap()
    }

    /// Sets the undefined imports using the provided closure and returns the resulting module.
    /// The closure is called with the name of each undefined import and should return
    /// the corresponding `Value`.
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

    use arithmetic_parser::{grammars::F32Grammar, GrammarExt};

    #[test]
    fn cloning_module() {
        let mut env = Registers::new();
        env.push_var("x", Value::<F32Grammar>::Number(5.0));

        let block = "y = x + 2 * (x + 1) + 1; y";
        let block = F32Grammar::parse_statements(block).unwrap();
        let (module, _) = Compiler::compile_module(WildcardId, &env, &block).unwrap();

        let mut module_copy = module.clone();
        module_copy.set_import("x", Value::Number(10.0));
        let value = module_copy.run().unwrap();
        assert_eq!(value, Value::Number(33.0));
        let value = module.run().unwrap();
        assert_eq!(value, Value::Number(18.0));
    }
}
