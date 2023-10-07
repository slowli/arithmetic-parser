//! `Function` and closely related types.

use core::fmt;

use crate::{
    alloc::{HashMap, Rc, String, ToOwned, Vec},
    arith::OrdArithmetic,
    error::{Backtrace, CodeInModule, Error, ErrorKind},
    exec::{ExecutableFn, ModuleId, Operations},
    fns::ValueCell,
    Environment, EvalResult, SpannedValue, Value,
};
use arithmetic_parser::{LvalueLen, MaybeSpanned};

/// Context for native function calls.
#[derive(Debug)]
pub struct CallContext<'r, T> {
    call_span: CodeInModule,
    backtrace: Option<&'r mut Backtrace>,
    operations: Operations<'r, T>,
}

impl<'r, T> CallContext<'r, T> {
    /// Creates a mock call context with the specified module ID and call span.
    /// The provided [`Environment`] is used to extract an [`OrdArithmetic`] implementation.
    pub fn mock(
        module_id: &dyn ModuleId,
        call_span: MaybeSpanned,
        env: &'r Environment<T>,
    ) -> Self {
        Self {
            call_span: CodeInModule::new(module_id, call_span),
            backtrace: None,
            operations: env.operations(),
        }
    }

    pub(crate) fn new(
        call_span: CodeInModule,
        backtrace: Option<&'r mut Backtrace>,
        operations: Operations<'r, T>,
    ) -> Self {
        Self {
            call_span,
            backtrace,
            operations,
        }
    }

    #[allow(clippy::needless_option_as_deref)] // false positive
    pub(crate) fn backtrace(&mut self) -> Option<&mut Backtrace> {
        self.backtrace.as_deref_mut()
    }

    pub(crate) fn arithmetic(&self) -> &'r dyn OrdArithmetic<T> {
        self.operations.arithmetic
    }

    /// Returns the call span of the currently executing function.
    pub fn call_span(&self) -> &CodeInModule {
        &self.call_span
    }

    /// Applies the call span to the specified `value`.
    pub fn apply_call_span<U>(&self, value: U) -> MaybeSpanned<U> {
        self.call_span.code().copy_with_extra(value)
    }

    /// Creates an error spanning the call site.
    pub fn call_site_error(&self, error: ErrorKind) -> Error {
        Error::from_parts(self.call_span.clone(), error)
    }

    /// Checks argument count and returns an error if it doesn't match.
    pub fn check_args_count(
        &self,
        args: &[SpannedValue<T>],
        expected_count: impl Into<LvalueLen>,
    ) -> Result<(), Error> {
        let expected_count = expected_count.into();
        if expected_count.matches(args.len()) {
            Ok(())
        } else {
            Err(self.call_site_error(ErrorKind::ArgsLenMismatch {
                def: expected_count,
                call: args.len(),
            }))
        }
    }
}

/// Function on zero or more [`Value`]s.
///
/// Native functions are defined in the Rust code and then can be used from the interpreted
/// code. See [`fns`](crate::fns) module docs for different ways to define native functions.
pub trait NativeFn<T> {
    /// Executes the function on the specified arguments.
    fn evaluate(
        &self,
        args: Vec<SpannedValue<T>>,
        context: &mut CallContext<'_, T>,
    ) -> EvalResult<T>;
}

impl<T, F> NativeFn<T> for F
where
    F: 'static + Fn(Vec<SpannedValue<T>>, &mut CallContext<'_, T>) -> EvalResult<T>,
{
    fn evaluate(
        &self,
        args: Vec<SpannedValue<T>>,
        context: &mut CallContext<'_, T>,
    ) -> EvalResult<T> {
        self(args, context)
    }
}

impl<T> fmt::Debug for dyn NativeFn<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("NativeFn").finish()
    }
}

impl<T> dyn NativeFn<T> {
    /// Extracts a data pointer from this trait object reference.
    pub(crate) fn data_ptr(&self) -> *const () {
        // `*const dyn Trait as *const ()` extracts the data pointer,
        // see https://github.com/rust-lang/rust/issues/27751. This is seemingly
        // the simplest way to extract the data pointer; `TraitObject` in `std::raw` is
        // a more future-proof alternative, but it is unstable.
        (self as *const dyn NativeFn<T>).cast()
    }
}

/// Function defined within the interpreter.
#[derive(Debug, Clone)]
pub struct InterpretedFn<T> {
    definition: Rc<ExecutableFn<T>>,
    captures: Vec<Value<T>>,
    capture_names: Vec<String>,
}

impl<T> InterpretedFn<T> {
    pub(crate) fn new(
        definition: Rc<ExecutableFn<T>>,
        captures: Vec<Value<T>>,
        capture_names: Vec<String>,
    ) -> Self {
        Self {
            definition,
            captures,
            capture_names,
        }
    }

    /// Returns ID of the module defining this function.
    pub fn module_id(&self) -> &dyn ModuleId {
        self.definition.inner.id()
    }

    /// Returns the number of arguments for this function.
    pub fn arg_count(&self) -> LvalueLen {
        self.definition.arg_count
    }

    /// Returns values captured by this function.
    pub fn captures(&self) -> HashMap<&str, &Value<T>> {
        self.capture_names
            .iter()
            .zip(&self.captures)
            .map(|(name, val)| (name.as_str(), val))
            .collect()
    }
}

impl<T: 'static + Clone> InterpretedFn<T> {
    /// Evaluates this function with the provided arguments and the execution context.
    pub fn evaluate(
        &self,
        args: Vec<SpannedValue<T>>,
        ctx: &mut CallContext<'_, T>,
    ) -> EvalResult<T> {
        if !self.arg_count().matches(args.len()) {
            let err = ErrorKind::ArgsLenMismatch {
                def: self.arg_count(),
                call: args.len(),
            };
            return Err(ctx.call_site_error(err));
        }

        let args = args.into_iter().map(|arg| arg.extra).collect();
        let captures: Result<Vec<_>, _> = self
            .captures
            .iter()
            .zip(&self.capture_names)
            .map(|(capture, name)| Self::deref_capture(capture, name))
            .collect();
        let captures = captures.map_err(|err| ctx.call_site_error(err))?;

        self.definition.inner.call_function(captures, args, ctx)
    }

    fn deref_capture(capture: &Value<T>, name: &str) -> Result<Value<T>, ErrorKind> {
        Ok(match capture {
            Value::Ref(opaque_ref) => {
                if let Some(cell) = opaque_ref.downcast_ref::<ValueCell<T>>() {
                    cell.get()
                        .cloned()
                        .ok_or_else(|| ErrorKind::Uninitialized(name.to_owned()))?
                } else {
                    capture.clone()
                }
            }
            _ => capture.clone(),
        })
    }
}

/// Function definition. Functions can be either native (defined in the Rust code) or defined
/// in the interpreter.
#[derive(Debug)]
pub enum Function<T> {
    /// Native function.
    Native(Rc<dyn NativeFn<T>>),
    /// Interpreted function.
    Interpreted(Rc<InterpretedFn<T>>),
}

impl<T> Clone for Function<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Native(function) => Self::Native(Rc::clone(function)),
            Self::Interpreted(function) => Self::Interpreted(Rc::clone(function)),
        }
    }
}

impl<T: fmt::Display> fmt::Display for Function<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Native(_) => formatter.write_str("(native fn)"),
            Self::Interpreted(function) => {
                formatter.write_str("(interpreted fn @ ")?;
                let location =
                    CodeInModule::new(function.module_id(), function.definition.def_span);
                location.fmt_location(formatter)?;
                formatter.write_str(")")
            }
        }
    }
}

impl<T: PartialEq> PartialEq for Function<T> {
    fn eq(&self, other: &Self) -> bool {
        self.is_same_function(other)
    }
}

impl<T> Function<T> {
    /// Creates a native function.
    pub fn native(function: impl NativeFn<T> + 'static) -> Self {
        Self::Native(Rc::new(function))
    }

    /// Checks if the provided function is the same as this one.
    pub fn is_same_function(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Native(this), Self::Native(other)) => this.data_ptr() == other.data_ptr(),
            (Self::Interpreted(this), Self::Interpreted(other)) => Rc::ptr_eq(this, other),
            _ => false,
        }
    }

    pub(crate) fn def_span(&self) -> Option<CodeInModule> {
        match self {
            Self::Native(_) => None,
            Self::Interpreted(function) => Some(CodeInModule::new(
                function.module_id(),
                function.definition.def_span,
            )),
        }
    }
}

impl<T: 'static + Clone> Function<T> {
    /// Evaluates the function on the specified arguments.
    pub fn evaluate(
        &self,
        args: Vec<SpannedValue<T>>,
        ctx: &mut CallContext<'_, T>,
    ) -> EvalResult<T> {
        match self {
            Self::Native(function) => function.evaluate(args, ctx),
            Self::Interpreted(function) => function.evaluate(args, ctx),
        }
    }
}
