//! Evaluation errors.

use core::fmt;

pub use arithmetic_parser::UnsupportedType;
use arithmetic_parser::{BinaryOp, LocatedSpan, Location, LvalueLen, Op, UnaryOp};

use crate::{
    alloc::{format, vec, Arc, Box, HashSet, String, ToOwned, ToString, Vec},
    exec::ModuleId,
    fns::FromValueError,
    Value,
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
            Self::IntegerOverflow => formatter.write_str("integer overflow or underflow"),
            Self::DivisionByZero => formatter.write_str("integer division by zero"),
            Self::InvalidExponent => formatter.write_str("exponent is too large or negative"),
            Self::NoInverse => formatter.write_str("integer has no multiplicative inverse"),
            Self::InvalidOp(err) => write!(formatter, "invalid operation: {err}"),
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
            Self::BinaryOp(op) => write!(formatter, "{op}"),
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
#[derive(Debug)]
#[non_exhaustive]
pub enum ErrorKind {
    /// Mismatch between length of tuples in a binary operation or assignment.
    TupleLenMismatch {
        /// Length of a tuple on the left-hand side.
        lhs: LvalueLen,
        /// Length of a tuple on the right-hand side.
        rhs: usize,
        /// Context in which the error has occurred.
        context: TupleLenMismatchContext,
    },

    /// Field set differs between LHS and RHS, which are both objects.
    FieldsMismatch {
        /// Fields in LHS.
        lhs_fields: HashSet<String>,
        /// Fields in RHS.
        rhs_fields: HashSet<String>,
        /// Operation being performed.
        op: BinaryOp,
    },

    /// Mismatch between the number of arguments in the function definition and its call.
    ArgsLenMismatch {
        /// Number of args at the function definition.
        def: LvalueLen,
        /// Number of args at the function call.
        call: usize,
    },

    /// Cannot destructure a non-tuple variable.
    CannotDestructure,

    /// Repeated assignment to the same variable in function args or tuple destructuring.
    RepeatedAssignment {
        /// Context in which the error has occurred.
        context: RepeatedAssignmentContext,
    },

    /// Repeated field in object initialization (e.g., `#{ x: 1, x: 2 }`) or destructure
    /// (e.g., `{ x, x }`).
    RepeatedField,

    /// Variable with the enclosed name is not defined.
    Undefined(String),
    /// Variable is not initialized.
    Uninitialized(String),

    /// Field name is invalid.
    InvalidFieldName(String),

    /// Value is not callable (i.e., it is not a function).
    CannotCall,
    /// Value cannot be indexed (i.e., it is not a tuple).
    CannotIndex,
    /// A field cannot be accessed for the value (i.e., it is not an object).
    CannotAccessFields,

    /// Index is out of bounds for the indexed tuple.
    IndexOutOfBounds {
        /// Index.
        index: usize,
        /// Actual tuple length.
        len: usize,
    },
    /// Object does not have a required field.
    NoField {
        /// Missing field.
        field: String,
        /// Available fields in the object in no particular order.
        available_fields: Vec<String>,
    },

    /// Generic error during execution of a native function.
    NativeCall(String),

    /// Error while converting arguments for [`FnWrapper`](crate::fns::FnWrapper).
    Wrapper(FromValueError),

    /// Unexpected operand type for the specified operation.
    UnexpectedOperand {
        /// Operation which failed.
        op: Op,
    },

    /// Value cannot be compared to other values. Only primitive values can be compared; other value types
    /// cannot.
    CannotCompare,

    /// Construct not supported by the interpreter.
    Unsupported(UnsupportedType),

    /// [`Arithmetic`](crate::arith::Arithmetic) error, such as division by zero.
    Arithmetic(ArithmeticError),
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TupleLenMismatch { context, lhs, rhs } => {
                write!(
                    formatter,
                    "Mismatch between length of tuples in {context}: LHS has {lhs} element(s), whereas RHS has {rhs}"
                )
            }
            Self::FieldsMismatch {
                op,
                lhs_fields,
                rhs_fields,
            } => {
                write!(
                    formatter,
                    "Cannot perform {op} on objects: LHS has fields {lhs_fields:?}, whereas RHS has fields {rhs_fields:?}"
                )
            }
            Self::ArgsLenMismatch { def, call } => {
                write!(
                    formatter,
                    "Mismatch between the number of arguments in the function definition and its call: \
                     definition requires {def} arg(s), call has {call}"
                )
            }
            Self::CannotDestructure => {
                formatter.write_str("Cannot destructure a non-tuple variable")
            }
            Self::RepeatedAssignment { context } => write!(
                formatter,
                "Repeated assignment to the same variable in {context}"
            ),
            Self::RepeatedField => formatter.write_str("Repeated object field"),
            Self::Undefined(var) => write!(formatter, "Variable `{var}` is not defined"),
            Self::Uninitialized(var) => write!(formatter, "Variable `{var}` is not initialized"),
            Self::InvalidFieldName(name) => write!(formatter, "`{name}` is not a valid field name"),
            Self::CannotCall => formatter.write_str("Value is not callable"),
            Self::CannotIndex => formatter.write_str("Value cannot be indexed"),
            Self::CannotAccessFields => {
                formatter.write_str("Fields cannot be accessed for the object")
            }
            Self::IndexOutOfBounds { index, len } => write!(
                formatter,
                "Attempting to get element {index} from tuple with length {len}"
            ),
            Self::NoField { field, .. } => write!(formatter, "Object does not have field {field}"),
            Self::NativeCall(err) => write!(formatter, "Failed executing native function: {err}"),
            Self::Wrapper(err) => write!(
                formatter,
                "Failed converting arguments for native function wrapper: {err}"
            ),
            Self::UnexpectedOperand { op } => write!(formatter, "Unexpected operand type for {op}"),
            Self::CannotCompare => formatter.write_str("Value cannot be compared to other values"),
            Self::Unsupported(ty) => write!(formatter, "Unsupported {ty}"),
            Self::Arithmetic(err) => write!(formatter, "Arithmetic error: {err}"),
        }
    }
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
                format!("Mismatch between length of tuples in {context}")
            }
            Self::FieldsMismatch { op, .. } => {
                format!("Mismatch between object shapes during {op}")
            }
            Self::ArgsLenMismatch { .. } => {
                "Mismatch between the number of arguments in the function definition and its call"
                    .to_owned()
            }
            Self::CannotDestructure => "Cannot destructure a non-tuple variable".to_owned(),
            Self::RepeatedAssignment { context } => {
                format!("Repeated assignment to the same variable in {context}")
            }
            Self::RepeatedField => "Repeated object field".to_owned(),
            Self::Undefined(name) => format!("Variable `{name}` is not defined"),
            Self::Uninitialized(name) => format!("Variable `{name}` is not initialized"),
            Self::InvalidFieldName(name) => format!("`{name}` is not a valid field name"),
            Self::CannotCall => "Value is not callable".to_owned(),
            Self::CannotIndex => "Value cannot be indexed".to_owned(),
            Self::CannotAccessFields => "Value has no fields".to_owned(),
            Self::IndexOutOfBounds { len, .. } => {
                format!("Index out of bounds for tuple with length {len}")
            }
            Self::NoField { field, .. } => format!("Object does not have field {field}"),
            Self::NativeCall(message) => message.clone(),
            Self::Wrapper(err) => err.to_string(),
            Self::UnexpectedOperand { op } => format!("Unexpected operand type for {op}"),
            Self::CannotCompare => "Value is not comparable".to_owned(),
            Self::Unsupported(_) => "Grammar construct not supported".to_owned(),
            Self::Arithmetic(_) => "Arithmetic error".to_owned(),
        }
    }

    /// Returns a short description of the spanned information.
    pub fn main_span_info(&self) -> String {
        match self {
            Self::TupleLenMismatch { context, lhs, .. } => {
                format!("LHS of {context} with {lhs} element(s)")
            }
            Self::FieldsMismatch { lhs_fields, .. } => {
                format!("LHS with fields {lhs_fields:?}")
            }
            Self::ArgsLenMismatch { call, .. } => format!("Called with {call} arg(s) here"),
            Self::CannotDestructure => "Failed destructuring".to_owned(),
            Self::RepeatedAssignment { .. } => "Re-assigned variable".to_owned(),
            Self::RepeatedField => "Repeated object field".to_owned(),
            Self::Undefined(_) => "Undefined variable occurrence".to_owned(),
            Self::Uninitialized(_) => "Uninitialized value".to_owned(),
            Self::InvalidFieldName(_) => "Invalid field".to_owned(),
            Self::CannotIndex | Self::IndexOutOfBounds { .. } => "Indexing operation".to_owned(),
            Self::CannotAccessFields | Self::NoField { .. } => "Field access".to_owned(),
            Self::CannotCall | Self::NativeCall(_) | Self::Wrapper(_) => "Failed call".to_owned(),
            Self::UnexpectedOperand { .. } => "Operand of wrong type".to_owned(),
            Self::CannotCompare => "Cannot be compared".to_owned(),
            Self::Unsupported(ty) => format!("Unsupported {ty}"),
            Self::Arithmetic(e) => e.to_string(),
        }
    }

    /// Returns information helping fix the error.
    pub fn help(&self) -> Option<String> {
        Some(match self {
            Self::TupleLenMismatch { context, .. } => format!(
                "If both args of {context} are tuples, the number of elements in them must agree"
            ),
            Self::FieldsMismatch { op, .. } => {
                format!("If both args of {op} are objects, their field names must be the same")
            }
            Self::CannotDestructure => "Only tuples can be destructured".to_owned(),
            Self::RepeatedAssignment { context } => {
                format!("In {context}, all assigned variables must have different names")
            }
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
                op: Op::Binary(BinaryOp::And | BinaryOp::Or),
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
    pub(crate) fn arg_value<T: fmt::Display>(value: &Value<T>) -> Self {
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
                write!(formatter, "RHS with the {size}-element tuple")
            }
            Self::UnbalancedRhsObject(fields) => {
                write!(formatter, "RHS object with fields {fields:?}")
            }
            Self::InvalidArg => formatter.write_str("Invalid argument"),
            Self::ArgValue(val) => write!(formatter, "Has value: {val}"),
        }
    }
}

/// Evaluation error together with one or more relevant code spans.
#[derive(Debug)]
pub struct Error {
    kind: Box<ErrorKind>,
    main_location: LocationInModule,
    aux_locations: Vec<LocationInModule<AuxErrorInfo>>,
}

impl Error {
    pub(crate) fn new<Span, T>(
        module_id: Arc<dyn ModuleId>,
        main_span: &LocatedSpan<Span, T>,
        kind: ErrorKind,
    ) -> Self
    where
        Span: Copy,
        Location: From<LocatedSpan<Span>>,
    {
        Self {
            kind: Box::new(kind),
            main_location: LocationInModule::new(module_id, main_span.with_no_extra().into()),
            aux_locations: vec![],
        }
    }

    pub(crate) fn from_parts(main_span: LocationInModule, kind: ErrorKind) -> Self {
        Self {
            kind: Box::new(kind),
            main_location: main_span,
            aux_locations: vec![],
        }
    }

    /// Adds an auxiliary span to this error. The `span` must be in the same module
    /// as the main span.
    #[must_use]
    pub fn with_location<T>(mut self, location: &Location<T>, info: AuxErrorInfo) -> Self {
        self.aux_locations.push(LocationInModule {
            module_id: self.main_location.module_id.clone(),
            location: location.copy_with_extra(info),
        });
        self
    }

    /// Returns the source of the error.
    pub fn kind(&self) -> &ErrorKind {
        &self.kind
    }

    /// Returns the main span for this error.
    pub fn location(&self) -> &LocationInModule {
        &self.main_location
    }

    /// Returns auxiliary spans for this error.
    pub fn aux_spans(&self) -> &[LocationInModule<AuxErrorInfo>] {
        &self.aux_locations
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.main_location.fmt_location(formatter)?;
        write!(formatter, ": {}", self.kind)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.kind)
    }
}

/// Result of an expression evaluation.
pub type EvalResult<T> = Result<Value<T>, Error>;

/// Code fragment together with information about the module containing the fragment.
#[derive(Debug, Clone)]
pub struct LocationInModule<T = ()> {
    module_id: Arc<dyn ModuleId>,
    location: Location<T>,
}

impl LocationInModule {
    pub(crate) fn new(module_id: Arc<dyn ModuleId>, location: Location) -> Self {
        Self {
            module_id,
            location,
        }
    }
}

impl<T> LocationInModule<T> {
    /// Returns the ID of the module containing this fragment.
    pub fn module_id(&self) -> &dyn ModuleId {
        self.module_id.as_ref()
    }

    /// Returns the code fragment within the module. The fragment may be stripped
    /// (i.e., contain only location info, not the code string itself).
    pub fn in_module(&self) -> &Location<T> {
        &self.location
    }

    pub(crate) fn fmt_location(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}:{}:{}",
            self.module_id,
            self.location.location_line(),
            self.location.get_column()
        )
    }
}

/// Element of a backtrace, i.e., a function / method call.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BacktraceElement {
    /// Function name.
    pub fn_name: String,
    /// Code span of the function definition. `None` for native functions.
    pub def_location: Option<LocationInModule>,
    /// Code span of the function call.
    pub call_location: LocationInModule,
}

/// Error backtrace.
#[derive(Debug, Default)]
pub(crate) struct Backtrace {
    calls: Vec<BacktraceElement>,
}

impl Backtrace {
    /// Appends a function call into the backtrace.
    pub fn push_call(
        &mut self,
        fn_name: &str,
        def_location: Option<LocationInModule>,
        call_location: LocationInModule,
    ) {
        self.calls.push(BacktraceElement {
            fn_name: fn_name.to_owned(),
            def_location,
            call_location,
        });
    }

    /// Pops a function call.
    pub fn pop_call(&mut self) {
        self.calls.pop();
    }
}

/// Error with the associated backtrace.
#[derive(Debug)]
pub struct ErrorWithBacktrace {
    inner: Error,
    backtrace: Backtrace,
}

impl ErrorWithBacktrace {
    pub(crate) fn new(inner: Error, backtrace: Backtrace) -> Self {
        Self { inner, backtrace }
    }

    /// Returns the source of the error.
    pub fn source(&self) -> &Error {
        &self.inner
    }

    /// Iterates over the error backtrace, starting from the most recent call.
    pub fn backtrace(&self) -> impl Iterator<Item = &BacktraceElement> + '_ {
        self.backtrace.calls.iter().rev()
    }
}

impl fmt::Display for ErrorWithBacktrace {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.inner, formatter)?;

        if formatter.alternate() && !self.backtrace.calls.is_empty() {
            writeln!(formatter, "\nBacktrace (most recent call last):")?;
            for (index, call) in self.backtrace.calls.iter().enumerate() {
                write!(formatter, "{:>4}: {} ", index + 1, call.fn_name)?;

                if let Some(ref def_span) = call.def_location {
                    write!(formatter, "(module `{}`)", def_span.module_id)?;
                } else {
                    formatter.write_str("(native)")?;
                }

                write!(formatter, " called at ")?;
                call.call_location.fmt_location(formatter)?;
                writeln!(formatter)?;
            }
        }
        Ok(())
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ErrorWithBacktrace {
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
        let main_span = Location::from_str(input, 4..8);
        let err = Error::new(
            Arc::new("test_module"),
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
        let main_span = Location::from_str(input, 4..8);
        let err = Error::new(
            Arc::new("test"),
            &main_span,
            ErrorKind::Undefined("test".to_owned()),
        );

        let mut err = ErrorWithBacktrace::new(err, Backtrace::default());
        let call_span = LocationInModule::new(Arc::new("test"), Location::from_str(input, ..));
        err.backtrace.push_call("test_fn", None, call_span);

        let err_string = err.to_string();
        assert_eq!(err_string, "test:1:5: Variable `test` is not defined");

        let expanded_err_string = format!("{err:#}");
        assert!(expanded_err_string.starts_with("test:1:5: Variable `test` is not defined"));
        assert!(expanded_err_string.contains("\nBacktrace"));
        assert!(expanded_err_string.contains("\n   1: test_fn (native) called at test:1:1"));
    }
}
