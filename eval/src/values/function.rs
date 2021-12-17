//! `Function` and closely related types.

use hashbrown::HashMap;

use core::fmt;

use crate::{
    alloc::{Rc, String, ToOwned, Vec},
    arith::OrdArithmetic,
    error::{Backtrace, CodeInModule, Error, ErrorKind},
    exec::{ExecutableFn, ModuleId, Operations},
    fns::ValueCell,
    values::StandardPrototypes,
    Environment, EvalResult, Prototype, SpannedValue, Value,
};
use arithmetic_parser::{LvalueLen, MaybeSpanned, StripCode};

/// Context for native function calls.
#[derive(Debug)]
pub struct CallContext<'r, 'a, T> {
    call_span: CodeInModule<'a>,
    backtrace: Option<&'r mut Backtrace<'a>>,
    operations: Operations<'r, T>,
}

impl<'r, 'a, T> CallContext<'r, 'a, T> {
    /// Creates a mock call context with the specified module ID and call span.
    /// The provided [`Environment`] is used to extract an [`OrdArithmetic`] implementation
    /// and [`Prototype`]s for standard types.
    pub fn mock(
        module_id: &dyn ModuleId,
        call_span: MaybeSpanned<'a>,
        env: &'r Environment<'a, T>,
    ) -> Self {
        Self {
            call_span: CodeInModule::new(module_id, call_span),
            backtrace: None,
            operations: env.operations(),
        }
    }

    pub(crate) fn new(
        call_span: CodeInModule<'a>,
        backtrace: Option<&'r mut Backtrace<'a>>,
        operations: Operations<'r, T>,
    ) -> Self {
        Self {
            call_span,
            backtrace,
            operations,
        }
    }

    #[allow(clippy::needless_option_as_deref)] // false positive
    pub(crate) fn backtrace(&mut self) -> Option<&mut Backtrace<'a>> {
        self.backtrace.as_deref_mut()
    }

    pub(crate) fn arithmetic(&self) -> &'r dyn OrdArithmetic<T> {
        self.operations.arithmetic
    }

    pub(crate) fn prototypes(&self) -> Option<&'r StandardPrototypes<T>> {
        self.operations.prototypes
    }

    /// Retrieves value prototype using standard prototypes if necessary.
    pub(crate) fn get_prototype(&self, value: &Value<'a, T>) -> Prototype<'a, T> {
        let prototype = match value {
            Value::Object(object) => object.prototype(),
            Value::Tuple(tuple) => tuple.prototype(),
            _ => None,
        };
        prototype.cloned().unwrap_or_else(|| {
            self.operations
                .prototypes
                .and_then(|prototypes| prototypes.get(value.value_type()).cloned())
                .unwrap_or_default()
        })
    }

    /// Returns the call span of the currently executing function.
    pub fn call_span(&self) -> &CodeInModule<'a> {
        &self.call_span
    }

    /// Applies the call span to the specified `value`.
    pub fn apply_call_span<U>(&self, value: U) -> MaybeSpanned<'a, U> {
        self.call_span.code().copy_with_extra(value)
    }

    /// Creates an error spanning the call site.
    pub fn call_site_error(&self, error: ErrorKind) -> Error<'a> {
        Error::from_parts(self.call_span.clone(), error)
    }

    /// Checks argument count and returns an error if it doesn't match.
    pub fn check_args_count(
        &self,
        args: &[SpannedValue<'a, T>],
        expected_count: impl Into<LvalueLen>,
    ) -> Result<(), Error<'a>> {
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
    fn evaluate<'a>(
        &self,
        args: Vec<SpannedValue<'a, T>>,
        context: &mut CallContext<'_, 'a, T>,
    ) -> EvalResult<'a, T>;
}

impl<T, F: 'static> NativeFn<T> for F
where
    F: for<'a> Fn(Vec<SpannedValue<'a, T>>, &mut CallContext<'_, 'a, T>) -> EvalResult<'a, T>,
{
    fn evaluate<'a>(
        &self,
        args: Vec<SpannedValue<'a, T>>,
        context: &mut CallContext<'_, 'a, T>,
    ) -> EvalResult<'a, T> {
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
#[derive(Debug)]
pub struct InterpretedFn<'a, T> {
    definition: Rc<ExecutableFn<'a, T>>,
    captures: Vec<Value<'a, T>>,
    capture_names: Vec<String>,
}

impl<T: Clone> Clone for InterpretedFn<'_, T> {
    fn clone(&self) -> Self {
        Self {
            definition: Rc::clone(&self.definition),
            captures: self.captures.clone(),
            capture_names: self.capture_names.clone(),
        }
    }
}

impl<T: 'static + Clone> StripCode for InterpretedFn<'_, T> {
    type Stripped = InterpretedFn<'static, T>;

    fn strip_code(self) -> Self::Stripped {
        InterpretedFn {
            definition: Rc::new(self.definition.to_stripped_code()),
            captures: self
                .captures
                .into_iter()
                .map(StripCode::strip_code)
                .collect(),
            capture_names: self.capture_names,
        }
    }
}

impl<'a, T> InterpretedFn<'a, T> {
    pub(crate) fn new(
        definition: Rc<ExecutableFn<'a, T>>,
        captures: Vec<Value<'a, T>>,
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
    pub fn captures(&self) -> HashMap<&str, &Value<'a, T>> {
        self.capture_names
            .iter()
            .zip(&self.captures)
            .map(|(name, val)| (name.as_str(), val))
            .collect()
    }
}

impl<T: 'static + Clone> InterpretedFn<'_, T> {
    fn to_stripped_code(&self) -> InterpretedFn<'static, T> {
        self.clone().strip_code()
    }
}

impl<'a, T: 'static + Clone> InterpretedFn<'a, T> {
    /// Evaluates this function with the provided arguments and the execution context.
    pub fn evaluate(
        &self,
        args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a, T>,
    ) -> EvalResult<'a, T> {
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

    fn deref_capture(capture: &Value<'a, T>, name: &str) -> Result<Value<'a, T>, ErrorKind> {
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
pub enum Function<'a, T> {
    /// Native function.
    Native(Rc<dyn NativeFn<T>>),
    /// Interpreted function.
    Interpreted(Rc<InterpretedFn<'a, T>>),
    /// Value prototype.
    Prototype(Prototype<'a, T>),
}

impl<T> Clone for Function<'_, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Native(function) => Self::Native(Rc::clone(function)),
            Self::Interpreted(function) => Self::Interpreted(Rc::clone(function)),
            Self::Prototype(proto) => Self::Prototype(proto.clone()),
        }
    }
}

impl<T: fmt::Display> fmt::Display for Function<'_, T> {
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
            Self::Prototype(proto) => {
                write!(formatter, "(prototype {})", proto.as_object())
            }
        }
    }
}

impl<T: PartialEq> PartialEq for Function<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        if let (Self::Prototype(this_proto), Self::Prototype(other_proto)) = (self, other) {
            *this_proto == *other_proto
        } else {
            self.is_same_function(other)
        }
    }
}

impl<T: 'static + Clone> StripCode for Function<'_, T> {
    type Stripped = Function<'static, T>;

    fn strip_code(self) -> Self::Stripped {
        match self {
            Self::Native(function) => Function::Native(function),
            Self::Interpreted(function) => {
                Function::Interpreted(Rc::new(function.to_stripped_code()))
            }
            Self::Prototype(proto) => Function::Prototype(proto.strip_code()),
        }
    }
}

impl<'a, T> Function<'a, T> {
    /// Creates a native function.
    pub fn native(function: impl NativeFn<T> + 'static) -> Self {
        Self::Native(Rc::new(function))
    }

    /// Checks if the provided function is the same as this one.
    pub fn is_same_function(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Native(this), Self::Native(other)) => this.data_ptr() == other.data_ptr(),
            (Self::Interpreted(this), Self::Interpreted(other)) => Rc::ptr_eq(this, other),
            (Self::Prototype(this), Self::Prototype(other)) => this.is_same(other),
            _ => false,
        }
    }

    pub(crate) fn def_span(&self) -> Option<CodeInModule<'a>> {
        match self {
            Self::Native(_) | Self::Prototype(_) => None,
            Self::Interpreted(function) => Some(CodeInModule::new(
                function.module_id(),
                function.definition.def_span,
            )),
        }
    }
}

impl<'a, T: 'static + Clone> Function<'a, T> {
    /// Evaluates the function on the specified arguments.
    pub fn evaluate(
        &self,
        args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a, T>,
    ) -> EvalResult<'a, T> {
        match self {
            Self::Native(function) => function.evaluate(args, ctx),
            Self::Interpreted(function) => function.evaluate(args, ctx),
            Self::Prototype(proto) => proto.evaluate(args, ctx),
        }
    }
}
