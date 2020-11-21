//! Values used by the interpreter.

use hashbrown::HashMap;

use core::{any::Any, cmp::Ordering, fmt};

use crate::{
    alloc::{vec, Rc, String, ToOwned, Vec},
    arith::{Arithmetic, Compare, OrdArithmetic},
    error::{AuxErrorInfo, Backtrace, CodeInModule, TupleLenMismatchContext},
    executable::ExecutableFn,
    fns, Error, ErrorKind, EvalResult, ModuleId,
};
use arithmetic_parser::{BinaryOp, LvalueLen, MaybeSpanned, Op, StripCode, UnaryOp};

mod env;
mod variable_map;

pub use self::{
    env::Environment,
    variable_map::{Comparisons, Prelude, VariableMap},
};

/// Opaque context for native calls.
#[derive(Debug)]
pub struct CallContext<'r, 'a, T> {
    call_span: CodeInModule<'a>,
    backtrace: Option<&'r mut Backtrace<'a>>,
    arithmetic: &'r dyn OrdArithmetic<T>,
}

impl<'r, 'a, T> CallContext<'r, 'a, T> {
    /// Creates a mock call context with the specified module ID and call span.
    pub fn mock<A>(module_id: &dyn ModuleId, call_span: MaybeSpanned<'a>, arithmetic: &'r A) -> Self
    where
        A: Arithmetic<T> + Compare<T>,
    {
        Self {
            call_span: CodeInModule::new(module_id, call_span),
            backtrace: None,
            arithmetic,
        }
    }

    pub(crate) fn new(
        call_span: CodeInModule<'a>,
        backtrace: Option<&'r mut Backtrace<'a>>,
        arithmetic: &'r dyn OrdArithmetic<T>,
    ) -> Self {
        Self {
            call_span,
            backtrace,
            arithmetic,
        }
    }

    pub(crate) fn backtrace(&mut self) -> Option<&mut Backtrace<'a>> {
        self.backtrace.as_deref_mut()
    }

    pub(crate) fn arithmetic(&self) -> &'r dyn OrdArithmetic<T> {
        self.arithmetic
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
        self as *const dyn NativeFn<T> as *const ()
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

impl<'a, T: Clone> InterpretedFn<'a, T> {
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
        self.definition
            .inner
            .call_function(self.captures.clone(), args, ctx)
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
}

impl<T> Clone for Function<'_, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Native(function) => Self::Native(Rc::clone(&function)),
            Self::Interpreted(function) => Self::Interpreted(Rc::clone(&function)),
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
            _ => false,
        }
    }

    pub(crate) fn def_span(&self) -> Option<CodeInModule<'a>> {
        match self {
            Self::Native(_) => None,
            Self::Interpreted(function) => Some(CodeInModule::new(
                function.module_id(),
                function.definition.def_span,
            )),
        }
    }
}

impl<'a, T: Clone> Function<'a, T> {
    /// Evaluates the function on the specified arguments.
    pub fn evaluate(
        &self,
        args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a, T>,
    ) -> EvalResult<'a, T> {
        match self {
            Self::Native(function) => function.evaluate(args, ctx),
            Self::Interpreted(function) => function.evaluate(args, ctx),
        }
    }
}

/// Possible high-level types of [`Value`]s.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ValueType {
    /// Number.
    Number,
    /// Boolean value.
    Bool,
    /// Function value.
    Function,
    /// Tuple of a specific size.
    Tuple(usize),
    /// Array (a tuple of arbitrary size).
    ///
    /// This variant is never returned from [`Value::value_type()`]; at the same time, it is
    /// used for error reporting etc.
    Array,
    /// Opaque reference to a value.
    Ref,
}

impl fmt::Display for ValueType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Number => formatter.write_str("number"),
            Self::Bool => formatter.write_str("boolean value"),
            Self::Function => formatter.write_str("function"),
            Self::Tuple(1) => write!(formatter, "tuple with 1 element"),
            Self::Tuple(size) => write!(formatter, "tuple with {} elements", size),
            Self::Array => formatter.write_str("array"),
            Self::Ref => formatter.write_str("reference"),
        }
    }
}

/// Opaque reference to a native value.
///
/// The references cannot be created by interpreted code, but can be used as function args
/// or return values of native functions. References are [`Rc`]'d, thus can easily be cloned.
///
/// References are comparable among each other:
///
/// - If the wrapped value implements [`PartialEq`], this implementation will be used
///   for comparison.
/// - If `PartialEq` is not implemented, the comparison is by the `Rc` pointer.
pub struct OpaqueRef {
    value: Rc<dyn Any>,
    // FIXME: do we need a `Debug` impl in the "vtable" as well?
    dyn_eq: fn(&dyn Any, &dyn Any) -> bool,
}

impl OpaqueRef {
    /// Creates a reference to `value` that implements equality comparison.
    ///
    /// Prefer using this method to [`Self::with_identity_eq()`] if the wrapped type implements
    /// [`PartialEq`].
    pub fn new<T: Any + PartialEq>(value: T) -> Self {
        Self {
            value: Rc::new(value),
            dyn_eq: |this, other| {
                let this_cast = this.downcast_ref::<T>().unwrap();
                other
                    .downcast_ref::<T>()
                    .map_or(false, |other_cast| other_cast == this_cast)
            },
        }
    }

    /// Creates a reference to `value` with the identity comparison: values are considered
    /// equal iff they point to the same data.
    ///
    /// Prefer [`Self::new()`] when possible.
    pub fn with_identity_eq<T: Any>(value: T) -> Self {
        Self {
            value: Rc::new(value),
            dyn_eq: |this, other| {
                let this_data = this as *const dyn Any as *const ();
                let other_data = other as *const dyn Any as *const ();
                this_data == other_data
            },
        }
    }

    /// Tries to downcast this reference to a specific type.
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.value.downcast_ref()
    }
}

impl Clone for OpaqueRef {
    fn clone(&self) -> Self {
        Self {
            value: Rc::clone(&self.value),
            dyn_eq: self.dyn_eq,
        }
    }
}

impl PartialEq for OpaqueRef {
    fn eq(&self, other: &Self) -> bool {
        (self.dyn_eq)(self.value.as_ref(), other.value.as_ref())
    }
}

impl fmt::Debug for OpaqueRef {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("OpaqueRef")
            .field(&self.value.as_ref())
            .finish()
    }
}

/// Values produced by expressions during their interpretation.
#[derive(Debug)]
#[non_exhaustive]
pub enum Value<'a, T> {
    /// Primitive value: a single literal.
    Number(T),
    /// Boolean value.
    Bool(bool),
    /// Function.
    Function(Function<'a, T>),
    /// Tuple of zero or more values.
    Tuple(Vec<Value<'a, T>>),
    /// Opaque reference to a native variable.
    Ref(OpaqueRef),
}

/// Value together with a span that has produced it.
pub type SpannedValue<'a, T> = MaybeSpanned<'a, Value<'a, T>>;

impl<'a, T> Value<'a, T> {
    /// Creates a value for a native function.
    pub fn native_fn(function: impl NativeFn<T> + 'static) -> Self {
        Self::Function(Function::Native(Rc::new(function)))
    }

    /// Creates a [wrapped function](fns::FnWrapper).
    ///
    /// Calling this method is equivalent to [`wrap`](fns::wrap)ping a function and calling
    /// [`Self::native_fn()`] on it. Thanks to type inference magic, the Rust compiler
    /// will usually be able to extract the `Args` type param from the function definition,
    /// provided that type of function arguments and its return type are defined explicitly
    /// or can be unequivocally inferred from the declaration.
    pub fn wrapped_fn<Args, F>(fn_to_wrap: F) -> Self
    where
        fns::FnWrapper<Args, F>: NativeFn<T> + 'static,
    {
        let wrapped = fns::wrap::<Args, _>(fn_to_wrap);
        Self::native_fn(wrapped)
    }

    /// Creates a value for an interpreted function.
    pub(crate) fn interpreted_fn(function: InterpretedFn<'a, T>) -> Self {
        Self::Function(Function::Interpreted(Rc::new(function)))
    }

    /// Creates a void value (an empty tuple).
    pub fn void() -> Self {
        Self::Tuple(vec![])
    }

    /// Creates a reference to a native variable.
    pub fn any(value: impl Any + PartialEq) -> Self {
        Self::Ref(OpaqueRef::new(value))
    }

    /// Returns the type of this value.
    pub fn value_type(&self) -> ValueType {
        match self {
            Self::Number(_) => ValueType::Number,
            Self::Bool(_) => ValueType::Bool,
            Self::Function(_) => ValueType::Function,
            Self::Tuple(elements) => ValueType::Tuple(elements.len()),
            Self::Ref(_) => ValueType::Ref,
        }
    }

    /// Checks if this value is void (an empty tuple).
    pub fn is_void(&self) -> bool {
        matches!(self, Self::Tuple(tuple) if tuple.is_empty())
    }

    /// Checks if this value is a function.
    pub fn is_function(&self) -> bool {
        matches!(self, Self::Function(_))
    }
}

impl<T: Clone> Clone for Value<'_, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Number(lit) => Self::Number(lit.clone()),
            Self::Bool(bool) => Self::Bool(*bool),
            Self::Function(function) => Self::Function(function.clone()),
            Self::Tuple(tuple) => Self::Tuple(tuple.clone()),
            Self::Ref(reference) => Self::Ref(reference.clone()),
        }
    }
}

impl<T: 'static + Clone> StripCode for Value<'_, T> {
    type Stripped = Value<'static, T>;

    fn strip_code(self) -> Self::Stripped {
        match self {
            Self::Number(lit) => Value::Number(lit),
            Self::Bool(bool) => Value::Bool(bool),
            Self::Function(function) => Value::Function(function.strip_code()),
            Self::Tuple(tuple) => {
                Value::Tuple(tuple.into_iter().map(StripCode::strip_code).collect())
            }
            Self::Ref(reference) => Value::Ref(reference),
        }
    }
}

impl<'a, T: Clone> From<&Value<'a, T>> for Value<'a, T> {
    fn from(reference: &Value<'a, T>) -> Self {
        reference.to_owned()
    }
}

impl<T: PartialEq> PartialEq for Value<'_, T> {
    fn eq(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Self::Number(this), Self::Number(other)) => this == other,
            (Self::Bool(this), Self::Bool(other)) => this == other,
            (Self::Tuple(this), Self::Tuple(other)) => this == other,
            (Self::Function(this), Self::Function(other)) => this.is_same_function(other),
            (Self::Ref(this), Self::Ref(other)) => this == other,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum OpSide {
    Lhs,
    Rhs,
}

#[derive(Debug)]
struct BinaryOpError {
    inner: ErrorKind,
    side: Option<OpSide>,
}

impl BinaryOpError {
    fn new(op: BinaryOp) -> Self {
        Self {
            inner: ErrorKind::UnexpectedOperand { op: Op::Binary(op) },
            side: None,
        }
    }

    fn tuple(op: BinaryOp, lhs: usize, rhs: usize) -> Self {
        Self {
            inner: ErrorKind::TupleLenMismatch {
                lhs: lhs.into(),
                rhs,
                context: TupleLenMismatchContext::BinaryOp(op),
            },
            side: Some(OpSide::Lhs),
        }
    }

    fn with_side(mut self, side: OpSide) -> Self {
        self.side = Some(side);
        self
    }

    fn with_error_kind(mut self, error_kind: ErrorKind) -> Self {
        self.inner = error_kind;
        self
    }

    fn span<'a>(
        self,
        module_id: &dyn ModuleId,
        total_span: MaybeSpanned<'a>,
        lhs_span: MaybeSpanned<'a>,
        rhs_span: MaybeSpanned<'a>,
    ) -> Error<'a> {
        let main_span = match self.side {
            Some(OpSide::Lhs) => lhs_span,
            Some(OpSide::Rhs) => rhs_span,
            None => total_span,
        };

        let aux_info = if let ErrorKind::TupleLenMismatch { rhs, .. } = self.inner {
            Some(AuxErrorInfo::UnbalancedRhs(rhs))
        } else {
            None
        };

        let mut err = Error::new(module_id, &main_span, self.inner);
        if let Some(aux_info) = aux_info {
            err = err.with_span(&rhs_span, aux_info);
        }
        err
    }
}

impl<'a, T: Clone> Value<'a, T> {
    fn try_binary_op_inner(
        self,
        rhs: Self,
        op: BinaryOp,
        arithmetic: &dyn OrdArithmetic<T>,
    ) -> Result<Self, BinaryOpError> {
        match (self, rhs) {
            (Self::Number(this), Self::Number(other)) => {
                let op_result = match op {
                    BinaryOp::Add => arithmetic.add(this, other),
                    BinaryOp::Sub => arithmetic.sub(this, other),
                    BinaryOp::Mul => arithmetic.mul(this, other),
                    BinaryOp::Div => arithmetic.div(this, other),
                    BinaryOp::Power => arithmetic.pow(this, other),
                    _ => unreachable!(),
                };
                op_result
                    .map(Self::Number)
                    .map_err(|e| BinaryOpError::new(op).with_error_kind(ErrorKind::Arithmetic(e)))
            }

            (this @ Self::Number(_), Self::Tuple(other)) => {
                let output: Result<Vec<_>, _> = other
                    .into_iter()
                    .map(|y| this.clone().try_binary_op_inner(y, op, arithmetic))
                    .collect();
                output.map(Self::Tuple)
            }

            (Self::Tuple(this), other @ Self::Number(_)) => {
                let output: Result<Vec<_>, _> = this
                    .into_iter()
                    .map(|x| x.try_binary_op_inner(other.clone(), op, arithmetic))
                    .collect();
                output.map(Self::Tuple)
            }

            (Self::Tuple(this), Self::Tuple(other)) => {
                if this.len() == other.len() {
                    let output: Result<Vec<_>, _> = this
                        .into_iter()
                        .zip(other)
                        .map(|(x, y)| x.try_binary_op_inner(y, op, arithmetic))
                        .collect();
                    output.map(Self::Tuple)
                } else {
                    Err(BinaryOpError::tuple(op, this.len(), other.len()))
                }
            }

            (Self::Number(_), _) | (Self::Tuple(_), _) => {
                Err(BinaryOpError::new(op).with_side(OpSide::Rhs))
            }
            _ => Err(BinaryOpError::new(op).with_side(OpSide::Lhs)),
        }
    }

    #[inline]
    pub(crate) fn try_binary_op(
        module_id: &dyn ModuleId,
        total_span: MaybeSpanned<'a>,
        lhs: MaybeSpanned<'a, Self>,
        rhs: MaybeSpanned<'a, Self>,
        op: BinaryOp,
        arithmetic: &dyn OrdArithmetic<T>,
    ) -> Result<Self, Error<'a>> {
        let lhs_span = lhs.with_no_extra();
        let rhs_span = rhs.with_no_extra();
        lhs.extra
            .try_binary_op_inner(rhs.extra, op, arithmetic)
            .map_err(|e| e.span(module_id, total_span, lhs_span, rhs_span))
    }

    pub(crate) fn try_neg(self, arithmetic: &dyn OrdArithmetic<T>) -> Result<Self, ErrorKind> {
        match self {
            Self::Number(val) => arithmetic
                .neg(val)
                .map(Self::Number)
                .map_err(ErrorKind::Arithmetic),

            Self::Tuple(tuple) => {
                let res: Result<Vec<_>, _> = tuple
                    .into_iter()
                    .map(|elem| Value::try_neg(elem, arithmetic))
                    .collect();
                res.map(Self::Tuple)
            }

            _ => Err(ErrorKind::UnexpectedOperand {
                op: UnaryOp::Neg.into(),
            }),
        }
    }

    pub(crate) fn try_not(self) -> Result<Self, ErrorKind> {
        match self {
            Self::Bool(val) => Ok(Self::Bool(!val)),
            Self::Tuple(tuple) => {
                let res: Result<Vec<_>, _> = tuple.into_iter().map(Value::try_not).collect();
                res.map(Self::Tuple)
            }

            _ => Err(ErrorKind::UnexpectedOperand {
                op: UnaryOp::Not.into(),
            }),
        }
    }

    // **NB.** Must match `PartialEq` impl for `Value`!
    pub(crate) fn eq_by_arithmetic(&self, rhs: &Self, arithmetic: &dyn OrdArithmetic<T>) -> bool {
        match (self, rhs) {
            (Self::Number(this), Self::Number(other)) => arithmetic.eq(this, other),
            (Self::Bool(this), Self::Bool(other)) => this == other,
            (Self::Tuple(this), Self::Tuple(other)) => {
                if this.len() == other.len() {
                    this.iter()
                        .zip(other.iter())
                        .all(|(x, y)| x.eq_by_arithmetic(y, arithmetic))
                } else {
                    false
                }
            }
            (Self::Function(this), Self::Function(other)) => this.is_same_function(other),
            (Self::Ref(this), Self::Ref(other)) => this == other,
            _ => false,
        }
    }

    pub(crate) fn compare(
        module_id: &dyn ModuleId,
        lhs: &MaybeSpanned<'a, Self>,
        rhs: &MaybeSpanned<'a, Self>,
        op: BinaryOp,
        arithmetic: &dyn OrdArithmetic<T>,
    ) -> Result<Self, Error<'a>> {
        // We only know how to compare numbers.
        let lhs_number = match &lhs.extra {
            Value::Number(number) => number,
            _ => return Err(Error::new(module_id, &lhs, ErrorKind::CannotCompare)),
        };
        let rhs_number = match &rhs.extra {
            Value::Number(number) => number,
            _ => return Err(Error::new(module_id, &rhs, ErrorKind::CannotCompare)),
        };

        let maybe_ordering = arithmetic.partial_cmp(lhs_number, rhs_number);
        let cmp_result = maybe_ordering.map_or(false, |ordering| match op {
            BinaryOp::Gt => ordering == Ordering::Greater,
            BinaryOp::Lt => ordering == Ordering::Less,
            BinaryOp::Ge => ordering != Ordering::Less,
            BinaryOp::Le => ordering != Ordering::Greater,
            _ => unreachable!(),
        });
        Ok(Value::Bool(cmp_result))
    }

    pub(crate) fn try_and(
        module_id: &dyn ModuleId,
        lhs: &MaybeSpanned<'a, Self>,
        rhs: &MaybeSpanned<'a, Self>,
    ) -> Result<Self, Error<'a>> {
        match (&lhs.extra, &rhs.extra) {
            (Value::Bool(this), Value::Bool(other)) => Ok(Value::Bool(*this && *other)),
            (Value::Bool(_), _) => {
                let err = ErrorKind::UnexpectedOperand {
                    op: BinaryOp::And.into(),
                };
                Err(Error::new(module_id, &rhs, err))
            }
            _ => {
                let err = ErrorKind::UnexpectedOperand {
                    op: BinaryOp::And.into(),
                };
                Err(Error::new(module_id, &lhs, err))
            }
        }
    }

    pub(crate) fn try_or(
        module_id: &dyn ModuleId,
        lhs: &MaybeSpanned<'a, Self>,
        rhs: &MaybeSpanned<'a, Self>,
    ) -> Result<Self, Error<'a>> {
        match (&lhs.extra, &rhs.extra) {
            (Value::Bool(this), Value::Bool(other)) => Ok(Value::Bool(*this || *other)),
            (Value::Bool(_), _) => {
                let err = ErrorKind::UnexpectedOperand {
                    op: BinaryOp::Or.into(),
                };
                Err(Error::new(module_id, &rhs, err))
            }
            _ => {
                let err = ErrorKind::UnexpectedOperand {
                    op: BinaryOp::Or.into(),
                };
                Err(Error::new(module_id, &lhs, err))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use core::cmp::Ordering;

    #[test]
    fn opaque_ref_equality() {
        let value = Value::<f32>::any(Ordering::Less);
        let same_value = Value::<f32>::any(Ordering::Less);
        assert_eq!(value, same_value);
        assert_eq!(value, value.clone());
        let other_value = Value::<f32>::any(Ordering::Greater);
        assert_ne!(value, other_value);
    }
}
