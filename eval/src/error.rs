//! Evaluation errors.

pub use arithmetic_parser::UnsupportedType;

use derive_more::Display;
use hashbrown::HashSet;

use core::fmt;

use crate::{
    alloc::{format, vec, Box, String, ToOwned, ToString, Vec},
    fns::FromValueError,
    ModuleId, Value, ValueType,
};
use arithmetic_parser::{
    BinaryOp, CodeFragment, LocatedSpan, LvalueLen, MaybeSpanned, Op, StripCode, UnaryOp,
};

/// Arithmetic errors raised by [`Arithmetic`] operations on primitive values.
///
/// [`Arithmetic`]: crate::arith::Arithmetic
#[derive(Debug)]
#[non_exhaustive]
pub enum ArithmeticError {
    /// Integer overflow or underflow.
    IntegerOverflow,
    /// Division by zero.
    DivisionByZero,
    /// Exponent of [`Arithmetic::pow()`] cannot be converted to `usize`, for example because
    /// it is too large or negative.
    ///
    /// [`Arithmetic::pow()`]: crate::arith::Arithmetic::pow()
    InvalidExponent,
    /// Integer used as a denominator in [`Arithmetic::div()`] has no multiplicative inverse.
    ///
    /// [`Arithmetic::div()`]: crate::arith::Arithmetic::div()
    NoInverse,
    /// Invalid operation with a custom error message.
    ///
    /// This error may be used by [`Arithmetic`](crate::arith::Arithmetic) implementations
    /// as a catch-all fallback.
    InvalidOp(anyhow::Error),
}

impl ArithmeticError {
    /// Creates a new invalid operation error with the specified `message`.
    pub fn invalid_op(message: impl Into<String>) -> Self {
        Self::InvalidOp(anyhow::Error::msg(message.into()))
    }
}

impl fmt::Display for ArithmeticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IntegerOverflow => formatter.write_str("Integer overflow or underflow"),
            Self::DivisionByZero => formatter.write_str("Integer division by zero"),
            Self::InvalidExponent => formatter.write_str("Exponent is too large or negative"),
            Self::NoInverse => formatter.write_str("Integer has no multiplicative inverse"),
            Self::InvalidOp(e) => write!(formatter, "Invalid operation: {}", e),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ArithmeticError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidOp(e) => Some(e.as_ref()),
            _ => None,
        }
    }
}

/// Context for [`ErrorKind::TupleLenMismatch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TupleLenMismatchContext {
    /// An error has occurred when evaluating a binary operation.
    BinaryOp(BinaryOp),
    /// An error has occurred during assignment.
    Assignment,
}

impl fmt::Display for TupleLenMismatchContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BinaryOp(op) => write!(formatter, "{}", op),
            Self::Assignment => formatter.write_str("assignment"),
        }
    }
}

/// Context for [`ErrorKind::RepeatedAssignment`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepeatedAssignmentContext {
    /// A duplicated variable is in function args.
    FnArgs,
    /// A duplicated variable is in an lvalue of an assignment.
    Assignment,
}

impl fmt::Display for RepeatedAssignmentContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::FnArgs => "function args",
            Self::Assignment => "assignment",
        })
    }
}

/// Kinds of errors that can occur when compiling or interpreting expressions and statements.
#[derive(Debug, Display)]
#[non_exhaustive]
pub enum ErrorKind {
    /// Mismatch between length of tuples in a binary operation or assignment.
    #[display(
        fmt = "Mismatch between length of tuples in {}: LHS has {} element(s), whereas RHS has {}",
        context,
        lhs,
        rhs
    )]
    TupleLenMismatch {
        /// Length of a tuple on the left-hand side.
        lhs: LvalueLen,
        /// Length of a tuple on the right-hand side.
        rhs: usize,
        /// Context in which the error has occurred.
        context: TupleLenMismatchContext,
    },

    /// Field set differs between LHS and RHS, which are both objects.
    #[display(
        fmt = "Cannot perform {} on objects: LHS has fields {:?}, whereas RHS has fields {:?}",
        op,
        lhs_fields,
        rhs_fields
    )]
    FieldsMismatch {
        /// Fields in LHS.
        lhs_fields: HashSet<String>,
        /// Fields in RHS.
        rhs_fields: HashSet<String>,
        /// Operation being performed.
        op: BinaryOp,
    },

    /// Mismatch between the number of arguments in the function definition and its call.
    #[display(
        fmt = "Mismatch between the number of arguments in the function definition and its call: \
            definition requires {} arg(s), call has {}",
        def,
        call
    )]
    ArgsLenMismatch {
        /// Number of args at the function definition.
        def: LvalueLen,
        /// Number of args at the function call.
        call: usize,
    },

    /// Cannot destructure a non-tuple variable.
    #[display(fmt = "Cannot destructure a non-tuple variable")]
    CannotDestructure,

    /// Repeated assignment to the same variable in function args or tuple destructuring.
    #[display(fmt = "Repeated assignment to the same variable in {}", context)]
    RepeatedAssignment {
        /// Context in which the error has occurred.
        context: RepeatedAssignmentContext,
    },

    /// Repeated field in object initialization (e.g., `#{ x: 1, x: 2 }`) or destructure
    /// (e.g., `{ x, x }`).
    #[display(fmt = "Repeated object field")]
    RepeatedField,

    /// Variable with the enclosed name is not defined.
    #[display(fmt = "Variable `{}` is not defined", _0)]
    Undefined(String),

    /// Field name is invalid.
    #[display(fmt = "`{}` is not a valid field name", _0)]
    InvalidFieldName(String),

    /// Value is not callable (i.e., it is not a function).
    #[display(fmt = "Value is not callable")]
    CannotCall,
    /// Value cannot be indexed (i.e., it is not a tuple).
    #[display(fmt = "Value cannot be indexed")]
    CannotIndex,
    /// A field cannot be accessed for the value (i.e., it is not an object).
    #[display(fmt = "Fields cannot be accessed for the object")]
    CannotAccessFields,

    /// Cannot use a value of the specified type as a method receiver.
    #[display(fmt = "Method cannot be called on {} since it has no prototype", _0)]
    NoPrototype(ValueType),

    /// Index is out of bounds for the indexed tuple.
    #[display(
        fmt = "Attempting to get element {} from tuple with length {}",
        index,
        len
    )]
    IndexOutOfBounds {
        /// Index.
        index: usize,
        /// Actual tuple length.
        len: usize,
    },
    /// Object does not have a required field.
    #[display(fmt = "Object does not have field {}", field)]
    NoField {
        /// Missing field.
        field: String,
        /// Available fields in the object in no particular order.
        available_fields: Vec<String>,
    },

    /// Generic error during execution of a native function.
    #[display(fmt = "Failed executing native function: {}", _0)]
    NativeCall(String),

    /// Error while converting arguments for [`FnWrapper`](crate::fns::FnWrapper).
    #[display(
        fmt = "Failed converting arguments for native function wrapper: {}",
        _0
    )]
    Wrapper(FromValueError),

    /// Unexpected operand type for the specified operation.
    #[display(fmt = "Unexpected operand type for {}", op)]
    UnexpectedOperand {
        /// Operation which failed.
        op: Op,
    },

    /// Value cannot be compared to other values. Only primitive values can be compared; other value types
    /// cannot.
    #[display(fmt = "Value cannot be compared to other values")]
    CannotCompare,

    /// Construct not supported by the interpreter.
    #[display(fmt = "Unsupported {}", _0)]
    Unsupported(UnsupportedType),

    /// [`Arithmetic`](crate::arith::Arithmetic) error, such as division by zero.
    #[display(fmt = "Arithmetic error: {}", _0)]
    Arithmetic(ArithmeticError),
}

impl ErrorKind {
    /// Creates a native error.
    pub fn native(message: impl Into<String>) -> Self {
        Self::NativeCall(message.into())
    }

    /// Creates an error for an lvalue type not supported by the interpreter.
    pub fn unsupported<T: Into<UnsupportedType>>(ty: T) -> Self {
        Self::Unsupported(ty.into())
    }

    /// Returned shortened error cause.
    pub fn to_short_string(&self) -> String {
        match self {
            Self::TupleLenMismatch { context, .. } => {
                format!("Mismatch between length of tuples in {}", context)
            }
            Self::FieldsMismatch { op, .. } => {
                format!("Mismatch between object shapes during {}", op)
            }
            Self::ArgsLenMismatch { .. } => {
                "Mismatch between the number of arguments in the function definition and its call"
                    .to_owned()
            }
            Self::CannotDestructure => "Cannot destructure a non-tuple variable".to_owned(),
            Self::RepeatedAssignment { context } => {
                format!("Repeated assignment to the same variable in {}", context)
            }
            Self::RepeatedField => "Repeated object field".to_owned(),
            Self::Undefined(name) => format!("Variable `{}` is not defined", name),
            Self::InvalidFieldName(name) => format!("`{}` is not a valid field name", name),
            Self::CannotCall => "Value is not callable".to_owned(),
            Self::CannotIndex => "Value cannot be indexed".to_owned(),
            Self::CannotAccessFields => "Value has no fields".to_owned(),
            Self::IndexOutOfBounds { len, .. } => {
                format!("Index out of bounds for tuple with length {}", len)
            }
            Self::NoField { field, .. } => format!("Object does not have field {}", field),
            Self::NoPrototype(ty) => format!("No prototype set for {}", ty),
            Self::NativeCall(message) => message.clone(),
            Self::Wrapper(err) => err.to_string(),
            Self::UnexpectedOperand { op } => format!("Unexpected operand type for {}", op),
            Self::CannotCompare => "Value is not comparable".to_owned(),
            Self::Unsupported(_) => "Grammar construct not supported".to_owned(),
            Self::Arithmetic(_) => "Arithmetic error".to_owned(),
        }
    }

    /// Returns a short description of the spanned information.
    pub fn main_span_info(&self) -> String {
        match self {
            Self::TupleLenMismatch { context, lhs, .. } => {
                format!("LHS of {} with {} element(s)", context, lhs)
            }
            Self::FieldsMismatch { lhs_fields, .. } => {
                format!("LHS with fields {:?}", lhs_fields)
            }
            Self::ArgsLenMismatch { call, .. } => format!("Called with {} arg(s) here", call),
            Self::CannotDestructure => "Failed destructuring".to_owned(),
            Self::RepeatedAssignment { .. } => "Re-assigned variable".to_owned(),
            Self::RepeatedField => "Repeated object field".to_owned(),
            Self::Undefined(_) => "Undefined variable occurrence".to_owned(),
            Self::InvalidFieldName(_) => "Invalid field".to_owned(),
            Self::CannotIndex | Self::IndexOutOfBounds { .. } => "Indexing operation".to_owned(),
            Self::CannotAccessFields | Self::NoField { .. } => "Field access".to_owned(),
            Self::NoPrototype(_) => "Method access".to_owned(),
            Self::CannotCall | Self::NativeCall(_) | Self::Wrapper(_) => "Failed call".to_owned(),
            Self::UnexpectedOperand { .. } => "Operand of wrong type".to_owned(),
            Self::CannotCompare => "Cannot be compared".to_owned(),
            Self::Unsupported(ty) => format!("Unsupported {}", ty),
            Self::Arithmetic(e) => e.to_string(),
        }
    }

    /// Returns information helping fix the error.
    pub fn help(&self) -> Option<String> {
        Some(match self {
            Self::TupleLenMismatch { context, .. } => format!(
                "If both args of {} are tuples, the number of elements in them must agree",
                context
            ),
            Self::FieldsMismatch { op, .. } => format!(
                "If both args of {} are objects, their field names must be the same",
                op
            ),
            Self::CannotDestructure => "Only tuples can be destructured".to_owned(),
            Self::RepeatedAssignment { context } => format!(
                "In {}, all assigned variables must have different names",
                context
            ),
            Self::RepeatedField => "Field names in objects must be unique".to_owned(),
            Self::InvalidFieldName(_) => "Field names must be `usize`s or identifiers".to_owned(),
            Self::CannotCall => "Only functions are callable, i.e., can be used as `fn_name` \
                in `fn_name(...)` expressions"
                .to_owned(),
            Self::CannotIndex => "Only tuples can be indexed".to_owned(),
            Self::CannotAccessFields => "Only objects have fields".to_owned(),

            Self::UnexpectedOperand { op: Op::Binary(op) } if op.is_arithmetic() => {
                "Operands of binary arithmetic ops must be primitive values or tuples / objects \
                 consisting of primitive values"
                    .to_owned()
            }
            Self::UnexpectedOperand { op: Op::Binary(op) } if op.is_comparison() => {
                "Operands of comparison ops must be primitive values".to_owned()
            }
            Self::UnexpectedOperand {
                op: Op::Binary(BinaryOp::And),
            }
            | Self::UnexpectedOperand {
                op: Op::Binary(BinaryOp::Or),
            } => "Operands of binary boolean ops must be boolean".to_owned(),
            Self::UnexpectedOperand {
                op: Op::Unary(UnaryOp::Neg),
            } => "Operand of negation must be primitive values or tuples / objects \
                  consisting of primitive values"
                .to_owned(),
            Self::UnexpectedOperand {
                op: Op::Unary(UnaryOp::Not),
            } => "Operand of boolean negation must be boolean".to_owned(),

            Self::CannotCompare => {
                "Only primitive values can be compared; complex values cannot".to_owned()
            }

            _ => return None,
        })
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ErrorKind {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Wrapper(error) => Some(error),
            Self::Arithmetic(error) => Some(error),
            _ => None,
        }
    }
}

/// Auxiliary information about an evaluation error.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum AuxErrorInfo {
    /// Function arguments declaration for [`ErrorKind::ArgsLenMismatch`].
    FnArgs,

    /// Previous variable assignment for [`ErrorKind::RepeatedAssignment`].
    PrevAssignment,

    /// Rvalue containing an invalid assignment for [`ErrorKind::CannotDestructure`]
    /// or [`ErrorKind::TupleLenMismatch`].
    Rvalue,

    /// RHS of a binary operation on differently sized tuples.
    UnbalancedRhsTuple(usize),
    /// RHS of a binary operation on differently shaped objects.
    UnbalancedRhsObject(HashSet<String>),

    /// Invalid argument.
    InvalidArg,

    /// String representation of an argument value (e.g., for a failed equality assertion).
    ArgValue(String),
}

impl AuxErrorInfo {
    pub(crate) fn arg_value<T: fmt::Display>(value: &Value<'_, T>) -> Self {
        Self::ArgValue(value.to_string())
    }
}

impl fmt::Display for AuxErrorInfo {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FnArgs => formatter.write_str("Function arguments declared here"),
            Self::PrevAssignment => formatter.write_str("Previous declaration"),
            Self::Rvalue => formatter.write_str("RHS containing the invalid assignment"),
            Self::UnbalancedRhsTuple(size) => {
                write!(formatter, "RHS with the {}-element tuple", size)
            }
            Self::UnbalancedRhsObject(fields) => {
                write!(formatter, "RHS object with fields {:?}", fields)
            }
            Self::InvalidArg => formatter.write_str("Invalid argument"),
            Self::ArgValue(val) => write!(formatter, "Has value: {}", val),
        }
    }
}

/// Evaluation error together with one or more relevant code spans.
///
/// Use the [`StripCode`] implementation to convert an `Error` to the `'static` lifetime, e.g.,
/// before boxing it into `Box<dyn std::error::Error>` or `anyhow::Error`.
/// If the error is wrapped into a [`Result`], you can do this via the `StripResultExt` trait
/// defined in the `arithmetic-parser` crate:
///
/// ```
/// # use arithmetic_parser::{grammars::{F64Grammar, Parse, Untyped}, StripResultExt};
/// # use arithmetic_eval::{Environment, Error, ExecutableModule, VariableMap, WildcardId};
/// fn compile_code(code: &str) -> anyhow::Result<ExecutableModule<'_, f64>> {
///     let block = Untyped::<F64Grammar>::parse_statements(code).strip_err()?;
///
///     // Without `strip_err()` call, the code below won't compile:
///     // `Error<'_>` in general cannot be boxed into `anyhow::Error`,
///     // only `Error<'static>` can.
///     Ok(Environment::new().compile_module(WildcardId, &block).strip_err()?)
/// }
/// ```
#[derive(Debug)]
pub struct Error<'a> {
    kind: ErrorKind,
    main_span: CodeInModule<'a>,
    aux_spans: Vec<CodeInModule<'a, AuxErrorInfo>>,
}

impl<'a> Error<'a> {
    pub(crate) fn new<Span, T>(
        module_id: &dyn ModuleId,
        main_span: &LocatedSpan<Span, T>,
        kind: ErrorKind,
    ) -> Self
    where
        Span: Copy + Into<CodeFragment<'a>>,
    {
        Self {
            kind,
            main_span: CodeInModule::new(
                module_id,
                main_span.with_no_extra().map_fragment(Into::into),
            ),
            aux_spans: vec![],
        }
    }

    pub(crate) fn from_parts(main_span: CodeInModule<'a>, kind: ErrorKind) -> Self {
        Self {
            kind,
            main_span,
            aux_spans: vec![],
        }
    }

    /// Adds an auxiliary span to this error. The `span` must be in the same module
    /// as the main span.
    pub fn with_span<T>(mut self, span: &MaybeSpanned<'a, T>, info: AuxErrorInfo) -> Self {
        self.aux_spans.push(CodeInModule {
            module_id: self.main_span.module_id.clone_boxed(),
            code: span.copy_with_extra(info),
        });
        self
    }

    /// Returns the source of the error.
    pub fn kind(&self) -> &ErrorKind {
        &self.kind
    }

    /// Returns the main span for this error.
    pub fn main_span(&self) -> &CodeInModule<'a> {
        &self.main_span
    }

    /// Returns auxiliary spans for this error.
    pub fn aux_spans(&self) -> &[CodeInModule<'a, AuxErrorInfo>] {
        &self.aux_spans
    }
}

impl fmt::Display for Error<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.main_span.fmt_location(formatter)?;
        write!(formatter, ": {}", self.kind)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error<'_> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.kind)
    }
}

impl StripCode for Error<'_> {
    type Stripped = Error<'static>;

    fn strip_code(self) -> Self::Stripped {
        Error {
            kind: self.kind,
            main_span: self.main_span.strip_code(),
            aux_spans: self
                .aux_spans
                .into_iter()
                .map(StripCode::strip_code)
                .collect(),
        }
    }
}

/// Result of an expression evaluation.
pub type EvalResult<'a, T> = Result<Value<'a, T>, Error<'a>>;

/// Code fragment together with information about the module containing the fragment.
#[derive(Debug)]
pub struct CodeInModule<'a, T = ()> {
    module_id: Box<dyn ModuleId>,
    code: MaybeSpanned<'a, T>,
}

impl<T: Clone> Clone for CodeInModule<'_, T> {
    fn clone(&self) -> Self {
        Self {
            module_id: self.module_id.clone_boxed(),
            code: self.code.clone(),
        }
    }
}

impl<'a> CodeInModule<'a> {
    pub(crate) fn new(module_id: &dyn ModuleId, span: MaybeSpanned<'a>) -> Self {
        Self {
            module_id: module_id.clone_boxed(),
            code: span,
        }
    }
}

impl<'a, T> CodeInModule<'a, T> {
    /// Returns the ID of the module containing this fragment.
    pub fn module_id(&self) -> &dyn ModuleId {
        self.module_id.as_ref()
    }

    /// Returns the code fragment within the module. The fragment may be stripped
    /// (i.e., contain only location info, not the code string itself).
    pub fn code(&self) -> &MaybeSpanned<'a, T> {
        &self.code
    }

    fn fmt_location(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}:{}:{}",
            self.module_id,
            self.code.location_line(),
            self.code.get_column()
        )
    }
}

impl<T: Clone + 'static> StripCode for CodeInModule<'_, T> {
    type Stripped = CodeInModule<'static, T>;

    fn strip_code(self) -> Self::Stripped {
        CodeInModule {
            module_id: self.module_id,
            code: self.code.strip_code(),
        }
    }
}

/// Element of a backtrace, i.e., a function / method call.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BacktraceElement<'a> {
    /// Function name.
    pub fn_name: String,
    /// Code span of the function definition. `None` for native functions.
    pub def_span: Option<CodeInModule<'a>>,
    /// Code span of the function call.
    pub call_span: CodeInModule<'a>,
}

impl StripCode for BacktraceElement<'_> {
    type Stripped = BacktraceElement<'static>;

    fn strip_code(self) -> Self::Stripped {
        BacktraceElement {
            fn_name: self.fn_name,
            def_span: self.def_span.map(StripCode::strip_code),
            call_span: self.call_span.strip_code(),
        }
    }
}

/// Error backtrace.
#[derive(Debug, Default)]
pub(crate) struct Backtrace<'a> {
    calls: Vec<BacktraceElement<'a>>,
}

impl<'a> Backtrace<'a> {
    /// Appends a function call into the backtrace.
    pub fn push_call(
        &mut self,
        fn_name: &str,
        def_span: Option<CodeInModule<'a>>,
        call_span: CodeInModule<'a>,
    ) {
        self.calls.push(BacktraceElement {
            fn_name: fn_name.to_owned(),
            def_span,
            call_span,
        });
    }

    /// Pops a function call.
    pub fn pop_call(&mut self) {
        self.calls.pop();
    }
}

impl StripCode for Backtrace<'_> {
    type Stripped = Backtrace<'static>;

    fn strip_code(self) -> Self::Stripped {
        Backtrace {
            calls: self.calls.into_iter().map(StripCode::strip_code).collect(),
        }
    }
}

/// Error with the associated backtrace.
///
/// Use the [`StripCode`] implementation to convert this to the `'static` lifetime, e.g.,
/// before boxing it into `Box<dyn std::error::Error>` or `anyhow::Error`.
#[derive(Debug)]
pub struct ErrorWithBacktrace<'a> {
    inner: Error<'a>,
    backtrace: Backtrace<'a>,
}

impl<'a> ErrorWithBacktrace<'a> {
    pub(crate) fn new(inner: Error<'a>, backtrace: Backtrace<'a>) -> Self {
        Self { inner, backtrace }
    }

    /// Returns the source of the error.
    pub fn source(&self) -> &Error<'a> {
        &self.inner
    }

    /// Iterates over the error backtrace, starting from the most recent call.
    pub fn backtrace(&self) -> impl Iterator<Item = &BacktraceElement<'a>> + '_ {
        self.backtrace.calls.iter().rev()
    }
}

impl fmt::Display for ErrorWithBacktrace<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.inner, formatter)?;

        if formatter.alternate() && !self.backtrace.calls.is_empty() {
            writeln!(formatter, "\nBacktrace (most recent call last):")?;
            for (index, call) in self.backtrace.calls.iter().enumerate() {
                write!(formatter, "{:>4}: {} ", index + 1, call.fn_name)?;

                if let Some(ref def_span) = call.def_span {
                    write!(formatter, "(module `{}`)", def_span.module_id)?;
                } else {
                    formatter.write_str("(native)")?;
                }

                write!(formatter, " called at ")?;
                call.call_span.fmt_location(formatter)?;
                writeln!(formatter)?;
            }
        }
        Ok(())
    }
}

impl StripCode for ErrorWithBacktrace<'_> {
    type Stripped = ErrorWithBacktrace<'static>;

    fn strip_code(self) -> Self::Stripped {
        ErrorWithBacktrace {
            inner: self.inner.strip_code(),
            backtrace: self.backtrace.strip_code(),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ErrorWithBacktrace<'_> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        std::error::Error::source(&self.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::ToString;

    #[test]
    fn display_for_eval_error() {
        let err = ErrorKind::Undefined("test".to_owned());
        assert_eq!(err.to_string(), "Variable `test` is not defined");

        let err = ErrorKind::ArgsLenMismatch {
            def: LvalueLen::AtLeast(2),
            call: 1,
        };
        assert!(err
            .to_string()
            .ends_with("definition requires at least 2 arg(s), call has 1"));
    }

    #[test]
    fn display_for_spanned_eval_error() {
        let input = "(_, test) = (1, 2);";
        let main_span = MaybeSpanned::from_str(input, 4..8);
        let err = Error::new(
            &"test_module",
            &main_span,
            ErrorKind::Undefined("test".to_owned()),
        );
        let err_string = err.to_string();
        assert_eq!(
            err_string,
            "test_module:1:5: Variable `test` is not defined"
        );
    }

    #[test]
    fn display_for_error_with_backtrace() {
        let input = "(_, test) = (1, 2);";
        let main_span = MaybeSpanned::from_str(input, 4..8);
        let err = Error::new(&"test", &main_span, ErrorKind::Undefined("test".to_owned()));

        let mut err = ErrorWithBacktrace::new(err, Backtrace::default());
        let call_span = CodeInModule::new(&"test", MaybeSpanned::from_str(input, ..));
        err.backtrace.push_call("test_fn", None, call_span);

        let err_string = err.to_string();
        assert_eq!(err_string, "test:1:5: Variable `test` is not defined");

        let expanded_err_string = format!("{:#}", err);
        assert!(expanded_err_string.starts_with("test:1:5: Variable `test` is not defined"));
        assert!(expanded_err_string.contains("\nBacktrace"));
        assert!(expanded_err_string.contains("\n   1: test_fn (native) called at test:1:1"));
    }
}
