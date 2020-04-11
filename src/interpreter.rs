//! Simple interpreter for ASTs produced by the parser.
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
//! - Functions are first-class (in fact, a function is just a variant of a [`Value`]).
//! - Functions can capture variables (including other functions). All captures are by value.
//! - Arithmetic operations are defined on primitive vars and tuples. With tuples, operations
//!   are performed per-element. Binary operations require tuples of the same size,
//!   or a tuple and a primitive value. As an example, `(1, 2) + 3` and `(2, 3) / (4, 5)` are valid,
//!   but `(1, 2) * (3, 4, 5)` isn't.
//! - No type checks are performed before evaluation.
//! - Type annotations are completely ignored. This means that the interpreter may execute
//!   code that is incorrect with annotations (e.g., assignment of a tuple to a variable which
//!   is annotated to have a numeric type).
//!
//! [`Interpreter`]: struct.Interpreter.html
//! [`Value`]: enum.Value.html
//!
//! # Examples
//!
//! ```
//! use arithmetic_parser::{
//!     grammars::F32Grammar,
//!     interpreter::{Assert, BinaryFn, Interpreter, Value},
//!     Grammar, GrammarExt, Span,
//! };
//!
//! const MIN: BinaryFn<fn(f32, f32) -> f32> =
//!     BinaryFn::new(|x, y| if x < y { x } else { y });
//! const MAX: BinaryFn<fn(f32, f32) -> f32> =
//!     BinaryFn::new(|x, y| if x > y { x } else { y });
//!
//! let mut context = Interpreter::new();
//! // Add some native functions to the interpreter.
//! context
//!     .innermost_scope()
//!     .insert_native_fn("min", MIN)
//!     .insert_native_fn("max", MAX)
//!     .insert_native_fn("assert", Assert);
//! // Create a new scope to make native functions non-deletable.
//! context.create_scope();
//!
//! let program = r#"
//!     ## The interpreter supports all parser features, including
//!     ## function definitions, tuples and blocks.
//!     order = |x, y| (min(x, y), max(x, y));
//!     assert(order(0.5, -1) == (-1, 0.5));
//!     (_, M) = order(3^2, { x = 3; x + 5 });
//!     M
//! "#;
//! let program = F32Grammar::parse_statements(Span::new(program)).unwrap();
//! let ret = context.evaluate(&program).unwrap();
//! assert_eq!(ret, Value::Number(9.0));
//! ```

pub use self::{
    error::{
        AuxErrorInfo, Backtrace, BacktraceElement, ErrorWithBacktrace, EvalError, EvalResult,
        RepeatedAssignmentContext, SpannedEvalError, TupleLenMismatchContext,
    },
    functions::{Assert, BinaryFn, Compare, If, Loop, UnaryFn},
};

use num_traits::{Num, Pow};

use std::{collections::HashMap, fmt, iter, ops, rc::Rc};

use crate::{
    helpers::{cover_spans, create_span_ref},
    BinaryOp, Block, Expr, FnDefinition, Grammar, Lvalue, Op, Span, Spanned, SpannedExpr,
    SpannedLvalue, SpannedStatement, Statement, UnaryOp,
};

mod error;
mod functions;

/// Opaque context for native calls.
#[derive(Debug)]
pub struct CallContext<'r, 'a> {
    fn_name: Span<'a>,
    call_span: Span<'a>,
    backtrace: &'r mut Option<Backtrace<'a>>,
}

impl<'a> CallContext<'_, 'a> {
    /// Returns the call span.
    pub fn apply_call_span<T>(&self, value: T) -> Spanned<'a, T> {
        create_span_ref(&self.call_span, value)
    }

    /// Creates the error spanning the call site.
    pub fn call_site_error(&self, error: EvalError) -> SpannedEvalError<'a> {
        SpannedEvalError::new(&self.call_span, error)
    }

    /// Checks argument count and returns an error if it doesn't match.
    pub fn check_args_count<T: Grammar>(
        &self,
        args: &[SpannedValue<'a, T>],
        expected_count: usize,
    ) -> Result<(), SpannedEvalError<'a>> {
        if args.len() == expected_count {
            Ok(())
        } else {
            Err(self.call_site_error(EvalError::ArgsLenMismatch {
                def: expected_count,
                call: args.len(),
            }))
        }
    }
}

/// Function on zero or more `Value`s.
pub trait NativeFn<T: Grammar> {
    /// Executes the function on the specified arguments.
    fn evaluate<'a>(
        &self,
        args: Vec<SpannedValue<'a, T>>,
        context: &mut CallContext<'_, 'a>,
    ) -> EvalResult<'a, T>;
}

impl<T: Grammar> fmt::Debug for dyn NativeFn<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("NativeFn").finish()
    }
}

/// Function defined within the interpreter.
#[derive(Debug)]
pub struct InterpretedFn<'a, T: Grammar> {
    definition: Spanned<'a, FnDefinition<'a, T>>,
    captures: Scope<'a, T>,
}

impl<'a, T: Grammar> InterpretedFn<'a, T> {
    /// Creates a new function.
    fn new(
        definition: Spanned<'a, FnDefinition<'a, T>>,
        context: &Interpreter<'a, T>,
    ) -> Result<Self, SpannedEvalError<'a>> {
        let mut validator = FnValidator::new();
        validator.eval_function(&definition.extra, context)?;
        Ok(Self {
            definition,
            captures: validator.captures,
        })
    }

    /// Returns the number of arguments for this function.
    pub fn arg_count(&self) -> usize {
        self.definition.extra.args.len()
    }

    /// Returns values captures by this function.
    pub fn captures(&self) -> &HashMap<String, Value<'a, T>> {
        &self.captures.variables
    }
}

impl<'a, T> InterpretedFn<'a, T>
where
    T: Grammar,
    T::Lit: Num + ops::Neg<Output = T::Lit> + Pow<T::Lit, Output = T::Lit>,
{
    /// Evaluates the function. This can be used from native functions.
    pub fn evaluate(
        &self,
        args: Vec<SpannedValue<'a, T>>,
        context: &mut CallContext<'_, 'a>,
    ) -> EvalResult<'a, T> {
        self.eval_inner(context.fn_name, context.call_span, args, context.backtrace)
    }

    fn eval_inner(
        &self,
        fn_name: Span<'a>,
        call_span: Span<'a>,
        args: Vec<SpannedValue<'a, T>>,
        backtrace: &mut Option<Backtrace<'a>>,
    ) -> EvalResult<'a, T> {
        if args.len() != self.definition.extra.args.len() {
            let err = EvalError::ArgsLenMismatch {
                def: self.definition.extra.args.len(),
                call: args.len(),
            };
            let args_span = cover_spans(
                create_span_ref(&self.definition, ()),
                &self.definition.extra.args,
            );
            let err =
                SpannedEvalError::new(&call_span, err).with_span(&args_span, AuxErrorInfo::FnArgs);
            return Err(err);
        }

        let def = &self.definition.extra;
        if let Some(backtrace) = backtrace {
            // FIXME: distinguish between interpreted and native calls.
            backtrace.push_call(
                fn_name.fragment,
                create_span_ref(&self.definition, ()),
                call_span,
            );
        }

        let mut context = Interpreter::from_scope(self.captures.clone());
        for (lvalue, val) in def.args.iter().zip(args) {
            context.innermost_scope().assign(lvalue, val)?;
        }
        let result = context.evaluate_inner(&def.body, backtrace);

        if result.is_ok() {
            if let Some(backtrace) = backtrace {
                backtrace.pop_call();
            }
        }
        result
    }
}

fn extract_vars<'it, 'a: 'it, T: 'it>(
    vars: &mut HashMap<&'a str, Span<'a>>,
    lvalues: impl Iterator<Item = &'it SpannedLvalue<'a, T>>,
    context: RepeatedAssignmentContext,
) -> Result<(), SpannedEvalError<'a>> {
    for lvalue in lvalues {
        match &lvalue.extra {
            Lvalue::Variable { .. } => {
                if lvalue.fragment != "_" {
                    let var_span = create_span_ref(lvalue, ());
                    if let Some(prev_span) = vars.insert(lvalue.fragment, var_span) {
                        let err = EvalError::RepeatedAssignment { context };
                        return Err(SpannedEvalError::new(lvalue, err)
                            .with_span(&prev_span, AuxErrorInfo::PrevAssignment));
                    }
                }
            }

            Lvalue::Tuple(fragments) => {
                extract_vars(vars, fragments.iter(), context)?;
            }
        }
    }
    Ok(())
}

/// Helper context for symbolic execution of a function body in order to determine
/// variables captured by the function.
#[derive(Debug)]
struct FnValidator<'a, T>
where
    T: Grammar,
{
    local_vars: Vec<HashMap<&'a str, Span<'a>>>,
    captures: Scope<'a, T>,
}

impl<'a, T: Grammar> FnValidator<'a, T> {
    fn new() -> Self {
        Self {
            local_vars: vec![],
            captures: Scope::new(),
        }
    }

    /// Collects variables captured by the function into a single `Scope`.
    fn eval_function(
        &mut self,
        definition: &FnDefinition<'a, T>,
        context: &Interpreter<'a, T>,
    ) -> Result<(), SpannedEvalError<'a>> {
        self.local_vars.push(HashMap::new());

        extract_vars(
            self.local_vars.last_mut().unwrap(),
            definition.args.iter(),
            RepeatedAssignmentContext::FnArgs,
        )?;
        for statement in &definition.body.statements {
            self.eval_statement(statement, context)?;
        }
        if let Some(ref return_expr) = definition.body.return_value {
            self.eval(return_expr, context)?;
        }

        // Remove local vars defined *within* the function.
        self.local_vars.pop();
        Ok(())
    }

    fn has_var(&self, var_name: &str) -> bool {
        self.captures.contains_var(var_name)
            || self.local_vars.iter().any(|set| set.contains_key(var_name))
    }

    /// Processes a local variable in the rvalue position.
    fn eval_local_var(
        &mut self,
        var_name: &str,
        context: &Interpreter<'a, T>,
    ) -> Result<(), EvalError> {
        if self.has_var(var_name) {
            // No action needs to be performed.
        } else if let Some(val) = context.get_var(var_name) {
            self.captures.insert_var(var_name, val.clone());
        } else {
            return Err(EvalError::Undefined(var_name.to_owned()));
        }

        Ok(())
    }

    /// Evaluates an expression with the function validation semantics, i.e., to determine
    /// function captures.
    fn eval(
        &mut self,
        expr: &SpannedExpr<'a, T>,
        context: &Interpreter<'a, T>,
    ) -> Result<(), SpannedEvalError<'a>> {
        match &expr.extra {
            Expr::Variable => {
                let var_name = expr.fragment;
                self.eval_local_var(var_name, context)
                    .map_err(|e| SpannedEvalError::new(expr, e))?;
            }

            Expr::Literal(_) => { /* no action */ }

            Expr::Tuple(fragments) => {
                for fragment in fragments {
                    self.eval(fragment, context)?;
                }
            }
            Expr::Unary { inner, .. } => {
                self.eval(inner, context)?;
            }
            Expr::Binary { lhs, rhs, .. } => {
                self.eval(lhs, context)?;
                self.eval(rhs, context)?;
            }

            Expr::Function { args, name } => {
                for arg in args {
                    self.eval(arg, context)?;
                }

                let fn_name = name.fragment;
                self.eval_local_var(fn_name, context)
                    .map_err(|e| SpannedEvalError::new(name, e))?;
            }

            Expr::Block(block) => {
                for statement in &block.statements {
                    self.eval_statement(statement, context)?;
                }
                if let Some(ref return_expr) = block.return_value {
                    self.eval(return_expr, context)?;
                }
            }

            Expr::FnDefinition(def) => {
                self.eval_function(def, context)?;
            }
        }
        Ok(())
    }

    /// Evaluates a statement using the provided context.
    fn eval_statement(
        &mut self,
        statement: &SpannedStatement<'a, T>,
        context: &Interpreter<'a, T>,
    ) -> Result<(), SpannedEvalError<'a>> {
        match &statement.extra {
            Statement::Expr(expr) => self.eval(expr, context),
            Statement::Assignment { lhs, rhs } => {
                self.eval(rhs, context)?;
                let mut new_vars = HashMap::new();
                extract_vars(
                    &mut new_vars,
                    iter::once(lhs),
                    RepeatedAssignmentContext::Assignment,
                )?;
                self.local_vars.last_mut().unwrap().extend(&new_vars);
                Ok(())
            }
        }
    }
}

/// Function definition. Functions can be either native (defined in the Rust code) or defined
/// in the interpreter.
#[derive(Debug)]
pub enum Function<'a, T>
where
    T: Grammar,
{
    /// Native function.
    Native(Rc<dyn NativeFn<T>>),
    /// Interpreted function.
    Interpreted(Rc<InterpretedFn<'a, T>>),
}

impl<T: Grammar> Clone for Function<'_, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Native(function) => Self::Native(Rc::clone(&function)),
            Self::Interpreted(function) => Self::Interpreted(Rc::clone(&function)),
        }
    }
}

impl<'a, T> Function<'a, T>
where
    T: Grammar,
    T::Lit: Num + ops::Neg<Output = T::Lit> + Pow<T::Lit, Output = T::Lit>,
{
    /// Evaluates the function on the specified arguments.
    pub fn evaluate(
        &self,
        args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a>,
    ) -> EvalResult<'a, T> {
        match self {
            Self::Native(function) => function.evaluate(args, ctx),
            Self::Interpreted(function) => function.evaluate(args, ctx),
        }
    }
}

/// Values produced by expressions during their interpretation.
#[derive(Debug)]
pub enum Value<'a, T>
where
    T: Grammar,
{
    /// Primitive value: a single literal.
    Number(T::Lit),
    /// Boolean value.
    Bool(bool),
    /// Function.
    Function(Function<'a, T>),
    /// Tuple of zero or more values.
    Tuple(Vec<Value<'a, T>>),
}

/// Value together with a span that has produced it.
pub type SpannedValue<'a, T> = Spanned<'a, Value<'a, T>>;

impl<'a, T: Grammar> Value<'a, T> {
    /// Creates a value for a native function.
    pub fn native_fn(function: impl NativeFn<T> + 'static) -> Self {
        Self::Function(Function::Native(Rc::new(function)))
    }

    /// Creates a value for an interpreted function.
    fn interpreted_fn(function: InterpretedFn<'a, T>) -> Self {
        Self::Function(Function::Interpreted(Rc::new(function)))
    }

    /// Creates a void value (an empty tuple).
    pub fn void() -> Self {
        Self::Tuple(vec![])
    }

    /// Checks if the value is void.
    pub fn is_void(&self) -> bool {
        match self {
            Self::Tuple(tuple) if tuple.is_empty() => true,
            _ => false,
        }
    }

    /// Checks if this value is a function.
    pub fn is_function(&self) -> bool {
        match self {
            Self::Function(_) => true,
            _ => false,
        }
    }
}

impl<T: Grammar> Clone for Value<'_, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Number(lit) => Self::Number(lit.clone()),
            Self::Bool(bool) => Self::Bool(*bool),
            Self::Function(function) => Self::Function(function.clone()),
            Self::Tuple(tuple) => Self::Tuple(tuple.clone()),
        }
    }
}

impl<T: Grammar> PartialEq for Value<'_, T>
where
    T::Lit: PartialEq,
{
    fn eq(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Self::Number(this), Self::Number(other)) => this == other,
            (Self::Bool(this), Self::Bool(other)) => this == other,
            (Self::Tuple(this), Self::Tuple(other)) => this == other,
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
    inner: EvalError,
    side: Option<OpSide>,
}

impl BinaryOpError {
    fn new(op: BinaryOp) -> Self {
        Self {
            inner: EvalError::UnexpectedOperand { op: Op::Binary(op) },
            side: None,
        }
    }

    fn tuple(op: BinaryOp, lhs: usize, rhs: usize) -> Self {
        Self {
            inner: EvalError::TupleLenMismatch {
                lhs,
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

    fn span<'a>(
        self,
        total_span: Span<'a>,
        lhs_span: Span<'a>,
        rhs_span: Span<'a>,
    ) -> SpannedEvalError<'a> {
        let main_span = match self.side {
            Some(OpSide::Lhs) => lhs_span,
            Some(OpSide::Rhs) => rhs_span,
            None => total_span,
        };

        let aux_info = if let EvalError::TupleLenMismatch { rhs, .. } = self.inner {
            Some(AuxErrorInfo::UnbalancedRhs(rhs))
        } else {
            None
        };

        let mut err = SpannedEvalError::new(&main_span, self.inner);
        if let Some(aux_info) = aux_info {
            err = err.with_span(&rhs_span, aux_info);
        }
        err
    }
}

impl<'a, T> Value<'a, T>
where
    T: Grammar,
    T::Lit: Num + ops::Neg<Output = T::Lit> + Pow<T::Lit, Output = T::Lit>,
{
    fn try_binary_op_inner(
        self,
        rhs: Self,
        op: BinaryOp,
        primitive_op: fn(T::Lit, T::Lit) -> T::Lit,
    ) -> Result<Self, BinaryOpError> {
        match (self, rhs) {
            (Self::Number(this), Self::Number(other)) => {
                Ok(Self::Number(primitive_op(this, other)))
            }

            (Self::Number(this), Self::Tuple(other)) => {
                let res: Result<Vec<_>, _> = other
                    .into_iter()
                    .map(|y| match y {
                        Self::Number(y) => Ok(Self::Number(primitive_op(this.clone(), y))),
                        _ => Err(BinaryOpError::new(op).with_side(OpSide::Rhs)),
                    })
                    .collect();
                res.map(Self::Tuple)
            }

            (Self::Tuple(this), Self::Number(other)) => {
                let res: Result<Vec<_>, _> = this
                    .into_iter()
                    .map(|x| match x {
                        Self::Number(x) => Ok(Self::Number(primitive_op(x, other.clone()))),
                        _ => Err(BinaryOpError::new(op).with_side(OpSide::Lhs)),
                    })
                    .collect();
                res.map(Self::Tuple)
            }

            (Self::Tuple(this), Self::Tuple(other)) => {
                if this.len() == other.len() {
                    let res: Result<Vec<_>, _> = this
                        .into_iter()
                        .zip(other)
                        .map(|(x, y)| x.try_binary_op_inner(y, op, primitive_op))
                        .collect();
                    res.map(Self::Tuple)
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

    fn try_binary_op(
        total_span: Span<'a>,
        lhs: Spanned<'a, Self>,
        rhs: Spanned<'a, Self>,
        op: BinaryOp,
        primitive_op: fn(T::Lit, T::Lit) -> T::Lit,
    ) -> Result<Self, SpannedEvalError<'a>> {
        let lhs_span = create_span_ref(&lhs, ());
        let rhs_span = create_span_ref(&rhs, ());
        lhs.extra
            .try_binary_op_inner(rhs.extra, op, primitive_op)
            .map_err(|e| e.span(total_span, lhs_span, rhs_span))
    }

    fn try_add(
        total_span: Span<'a>,
        lhs: Spanned<'a, Self>,
        rhs: Spanned<'a, Self>,
    ) -> Result<Self, SpannedEvalError<'a>> {
        Self::try_binary_op(total_span, lhs, rhs, BinaryOp::Add, |x, y| x + y)
    }

    fn try_sub(
        total_span: Span<'a>,
        lhs: Spanned<'a, Self>,
        rhs: Spanned<'a, Self>,
    ) -> Result<Self, SpannedEvalError<'a>> {
        Self::try_binary_op(total_span, lhs, rhs, BinaryOp::Sub, |x, y| x - y)
    }

    fn try_mul(
        total_span: Span<'a>,
        lhs: Spanned<'a, Self>,
        rhs: Spanned<'a, Self>,
    ) -> Result<Self, SpannedEvalError<'a>> {
        Self::try_binary_op(total_span, lhs, rhs, BinaryOp::Mul, |x, y| x * y)
    }

    fn try_div(
        total_span: Span<'a>,
        lhs: Spanned<'a, Self>,
        rhs: Spanned<'a, Self>,
    ) -> Result<Self, SpannedEvalError<'a>> {
        Self::try_binary_op(total_span, lhs, rhs, BinaryOp::Div, |x, y| x / y)
    }

    fn try_pow(
        total_span: Span<'a>,
        lhs: Spanned<'a, Self>,
        rhs: Spanned<'a, Self>,
    ) -> Result<Self, SpannedEvalError<'a>> {
        Self::try_binary_op(total_span, lhs, rhs, BinaryOp::Power, |x, y| x.pow(y))
    }

    fn try_neg(self) -> Result<Self, EvalError> {
        match self {
            Self::Number(val) => Ok(Self::Number(-val)),
            Self::Tuple(tuple) => {
                let res: Result<Vec<_>, _> = tuple.into_iter().map(|elem| elem.try_neg()).collect();
                res.map(Self::Tuple)
            }

            _ => Err(EvalError::UnexpectedOperand {
                op: UnaryOp::Neg.into(),
            }),
        }
    }

    fn try_not(self) -> Result<Self, EvalError> {
        match self {
            Self::Bool(val) => Ok(Self::Bool(!val)),
            Self::Tuple(tuple) => {
                let res: Result<Vec<_>, _> = tuple.into_iter().map(|elem| elem.try_not()).collect();
                res.map(Self::Tuple)
            }

            _ => Err(EvalError::UnexpectedOperand {
                op: UnaryOp::Not.into(),
            }),
        }
    }

    fn try_compare(self, rhs: Self) -> Result<bool, EvalError> {
        match (self, rhs) {
            (Self::Number(this), Self::Number(other)) => Ok(this == other),
            (Self::Tuple(this), Self::Tuple(other)) => {
                if this.len() == other.len() {
                    this.into_iter()
                        .zip(other)
                        .try_fold(true, |acc, (x, y)| Ok(acc && x.try_compare(y)?))
                } else {
                    Err(EvalError::TupleLenMismatch {
                        lhs: this.len(),
                        rhs: other.len(),
                        context: TupleLenMismatchContext::BinaryOp(BinaryOp::Eq),
                    })
                }
            }

            _ => Err(EvalError::UnexpectedOperand {
                op: BinaryOp::Eq.into(),
            }),
        }
    }
}

/// Variable scope containing functions and variables.
#[derive(Debug)]
pub struct Scope<'a, T: Grammar> {
    variables: HashMap<String, Value<'a, T>>,
}

impl<T: Grammar> Clone for Scope<'_, T> {
    fn clone(&self) -> Self {
        Self {
            variables: self.variables.clone(),
        }
    }
}

impl<T: Grammar> Default for Scope<'_, T> {
    fn default() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }
}

impl<'a, T: Grammar> Scope<'a, T> {
    /// Creates a new scope with no associated variables.
    pub fn new() -> Self {
        Self::default()
    }

    /// Checks if the scope contains a variable with the specified name.
    pub fn contains_var(&self, name: &str) -> bool {
        self.variables.contains_key(name)
    }

    /// Gets the variable with the specified name.
    pub fn get_var(&self, name: &str) -> Option<&Value<'a, T>> {
        self.variables.get(name)
    }

    /// Returns an iterator over all variables in this scope.
    pub fn variables(&self) -> impl Iterator<Item = (&str, &Value<'a, T>)> + '_ {
        self.variables
            .iter()
            .map(|(name, value)| (name.as_str(), value))
    }

    /// Defines a variable with the specified name and value.
    pub fn insert_var(&mut self, name: &str, value: Value<'a, T>) -> &mut Self {
        self.variables.insert(name.to_owned(), value);
        self
    }

    /// Removes all variables from the scope.
    pub fn clear(&mut self) {
        self.variables.clear();
    }

    /// Inserts a native function into the context.
    pub fn insert_native_fn<F>(&mut self, name: &str, fun: F) -> &mut Self
    where
        F: NativeFn<T> + 'static,
    {
        self.variables
            .insert(name.to_owned(), Value::native_fn(fun));
        self
    }

    fn assign(
        &mut self,
        lvalue: &SpannedLvalue<'a, T::Type>,
        rvalue: SpannedValue<'a, T>,
    ) -> Result<(), SpannedEvalError<'a>> {
        let total_rvalue_span = create_span_ref(&rvalue, ());
        self.do_assign(lvalue, rvalue.extra, total_rvalue_span)
    }

    fn do_assign(
        &mut self,
        lvalue: &SpannedLvalue<'a, T::Type>,
        rvalue: Value<'a, T>,
        total_rvalue_span: Span<'a>,
    ) -> Result<(), SpannedEvalError<'a>> {
        // TODO: This check is repeated for function bodies.
        extract_vars(
            &mut HashMap::new(),
            iter::once(lvalue),
            RepeatedAssignmentContext::Assignment,
        )?;

        match &lvalue.extra {
            Lvalue::Variable { .. } => {
                let var_name = lvalue.fragment;
                if var_name != "_" {
                    self.variables.insert(var_name.to_owned(), rvalue);
                }
            }

            Lvalue::Tuple(assignments) => {
                if let Value::Tuple(fragments) = rvalue {
                    if assignments.len() != fragments.len() {
                        let err = EvalError::TupleLenMismatch {
                            lhs: assignments.len(),
                            rhs: fragments.len(),
                            context: TupleLenMismatchContext::Assignment,
                        };
                        return Err(SpannedEvalError::new(&lvalue, err)
                            .with_span(&total_rvalue_span, AuxErrorInfo::Rvalue));
                    }

                    for (assignment, fragment) in assignments.iter().zip(fragments) {
                        self.do_assign(assignment, fragment, total_rvalue_span)?;
                    }
                } else {
                    return Err(SpannedEvalError::new(&lvalue, EvalError::CannotDestructure)
                        .with_span(&total_rvalue_span, AuxErrorInfo::Rvalue));
                }
            }
        }
        Ok(())
    }
}

/// Interpreter for statements and expressions.
///
/// See the [module docs](index.html) for the examples of usage.
#[derive(Debug)]
pub struct Interpreter<'a, T: Grammar> {
    scopes: Vec<Scope<'a, T>>,
}

impl<T: Grammar> Default for Interpreter<'_, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T: Grammar> Interpreter<'a, T> {
    /// Creates a new empty context.
    pub fn new() -> Self {
        Self {
            scopes: vec![Scope::new()],
        }
    }

    fn from_scope(scope: Scope<'a, T>) -> Self {
        Self {
            scopes: vec![scope],
        }
    }

    /// Returns an exclusive reference to the innermost scope.
    pub fn innermost_scope(&mut self) -> &mut Scope<'a, T> {
        self.scopes.last_mut().unwrap()
    }

    /// Creates a new scope and pushes it onto the stack.
    pub fn create_scope(&mut self) {
        self.scopes.push(Scope::new());
    }

    /// Pops the innermost scope.
    pub fn pop_scope(&mut self) -> Option<Scope<'a, T>> {
        if self.scopes.len() > 1 {
            self.scopes.pop() // should always be `Some(_)`
        } else {
            None
        }
    }

    /// Gets the variable with the specified name. The variable is looked up starting from
    /// the innermost scope.
    pub fn get_var(&self, name: &str) -> Option<&Value<'a, T>> {
        self.scopes
            .iter()
            .rev()
            .filter_map(|scope| scope.get_var(name))
            .next()
    }
}

impl<'a, T: Grammar> Interpreter<'a, T>
where
    T::Lit: Num + ops::Neg<Output = T::Lit> + Pow<T::Lit, Output = T::Lit>,
{
    fn evaluate_binary_expr(
        &mut self,
        expr_span: Span<'a>,
        spanned_lhs: &SpannedExpr<'a, T>,
        spanned_rhs: &SpannedExpr<'a, T>,
        op: Spanned<'a, BinaryOp>,
        backtrace: &mut Option<Backtrace<'a>>,
    ) -> EvalResult<'a, T> {
        let lhs = self.evaluate_expr_inner(spanned_lhs, backtrace)?;

        // Short-circuit logic for bool operations.
        match op.extra {
            BinaryOp::And | BinaryOp::Or => match lhs {
                Value::Bool(b) => {
                    if !b && op.extra == BinaryOp::And {
                        return Ok(Value::Bool(false));
                    } else if b && op.extra == BinaryOp::Or {
                        return Ok(Value::Bool(true));
                    }
                }

                _ => {
                    let err = EvalError::UnexpectedOperand {
                        op: op.extra.into(),
                    };
                    return Err(SpannedEvalError::new(&spanned_lhs, err));
                }
            },

            _ => { /* do nothing yet */ }
        }

        let rhs = self.evaluate_expr_inner(spanned_rhs, backtrace)?;
        let lhs = create_span_ref(&spanned_lhs, lhs);
        let rhs = create_span_ref(&spanned_rhs, rhs);

        match op.extra {
            BinaryOp::Add => Value::try_add(expr_span, lhs, rhs),
            BinaryOp::Sub => Value::try_sub(expr_span, lhs, rhs),
            BinaryOp::Mul => Value::try_mul(expr_span, lhs, rhs),
            BinaryOp::Div => Value::try_div(expr_span, lhs, rhs),
            BinaryOp::Power => Value::try_pow(expr_span, lhs, rhs),

            BinaryOp::Eq | BinaryOp::NotEq => {
                let eq = lhs
                    .extra
                    .try_compare(rhs.extra)
                    .map_err(|e| SpannedEvalError::new(&expr_span, e))?;
                Ok(Value::Bool(if op.extra == BinaryOp::Eq { eq } else { !eq }))
            }

            BinaryOp::And | BinaryOp::Or => {
                match rhs.extra {
                    // This works since we know that AND / OR hasn't short-circuited.
                    Value::Bool(b) => Ok(Value::Bool(b)),

                    _ => {
                        let err = EvalError::UnexpectedOperand {
                            op: op.extra.into(),
                        };
                        Err(SpannedEvalError::new(&rhs, err))
                    }
                }
            }
        }
    }

    fn evaluate_expr_inner(
        &mut self,
        expr: &SpannedExpr<'a, T>,
        backtrace: &mut Option<Backtrace<'a>>,
    ) -> EvalResult<'a, T> {
        match &expr.extra {
            Expr::Variable => self.get_var(expr.fragment).cloned().ok_or_else(|| {
                let err = EvalError::Undefined(expr.fragment.to_owned());
                SpannedEvalError::new(expr, err)
            }),
            Expr::Literal(value) => Ok(Value::Number(value.to_owned())),

            Expr::Tuple(fragments) => {
                let fragments: Result<Vec<_>, _> = fragments
                    .iter()
                    .map(|frag| self.evaluate_expr_inner(frag, backtrace))
                    .collect();
                fragments.map(Value::Tuple)
            }

            Expr::Function { name, args } => {
                let func = self.evaluate_expr_inner(name, backtrace)?;
                let func = match func {
                    Value::Function(func) => func,
                    _ => {
                        return Err(SpannedEvalError::new(expr, EvalError::CannotCall));
                    }
                };

                let args: Result<Vec<_>, _> = args
                    .iter()
                    .map(|arg| {
                        let span = create_span_ref(&arg, ());
                        self.evaluate_expr_inner(arg, backtrace)
                            .map(|value| create_span_ref(&span, value))
                    })
                    .collect();
                let args = args?;
                let mut context = CallContext {
                    fn_name: create_span_ref(name, ()),
                    call_span: create_span_ref(expr, ()),
                    backtrace,
                };
                func.evaluate(args, &mut context)
            }

            // Arithmetic operations
            Expr::Unary { inner, op } => {
                let val = self.evaluate_expr_inner(inner, backtrace)?;

                match op.extra {
                    UnaryOp::Not => val.try_not().map_err(|e| SpannedEvalError::new(expr, e)),
                    UnaryOp::Neg => val.try_neg().map_err(|e| SpannedEvalError::new(expr, e)),
                }
            }

            Expr::Binary { lhs, rhs, op } => {
                self.evaluate_binary_expr(create_span_ref(expr, ()), lhs, rhs, *op, backtrace)
            }

            Expr::Block(statements) => {
                self.create_scope();
                let result = self.evaluate_inner(statements, backtrace);
                self.scopes.pop(); // Clear the scope in any case
                result
            }

            Expr::FnDefinition(def) => {
                let fun = InterpretedFn::new(create_span_ref(expr, def.clone()), self)?;
                Ok(Value::interpreted_fn(fun))
            }
        }
    }

    /// Evaluates expression.
    pub fn evaluate_expr(
        &mut self,
        expr: &SpannedExpr<'a, T>,
    ) -> Result<Value<'a, T>, ErrorWithBacktrace<'a>> {
        let mut backtrace = Some(Backtrace::default());
        self.evaluate_expr_inner(expr, &mut backtrace)
            .map_err(|e| ErrorWithBacktrace::new(e, backtrace.unwrap()))
    }

    /// Evaluates a list of statements.
    fn evaluate_inner(
        &mut self,
        block: &Block<'a, T>,
        backtrace: &mut Option<Backtrace<'a>>,
    ) -> EvalResult<'a, T> {
        use crate::Statement::*;

        for statement in &block.statements {
            match &statement.extra {
                Expr(expr) => {
                    self.evaluate_expr_inner(expr, backtrace)?;
                }

                Assignment { lhs, rhs } => {
                    let evaluated = self.evaluate_expr_inner(rhs, backtrace)?;
                    let evaluated = create_span_ref(&rhs, evaluated);
                    self.scopes.last_mut().unwrap().assign(lhs, evaluated)?;
                }
            }
        }

        let return_value = if let Some(ref return_expr) = block.return_value {
            self.evaluate_expr_inner(return_expr, backtrace)?
        } else {
            Value::void()
        };
        Ok(return_value)
    }

    /// Evaluates a list of statements.
    pub fn evaluate(
        &mut self,
        block: &Block<'a, T>,
    ) -> Result<Value<'a, T>, ErrorWithBacktrace<'a>> {
        let mut backtrace = Some(Backtrace::default());
        self.evaluate_inner(block, &mut backtrace)
            .map_err(|e| ErrorWithBacktrace::new(e, backtrace.unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{grammars::F32Grammar, GrammarExt};

    use assert_matches::assert_matches;

    const SIN: UnaryFn<fn(f32) -> f32> = UnaryFn::new(f32::sin);

    #[test]
    fn basic_program() {
        let program = Span::new("x = 1; y = 2; x + y");
        let block = F32Grammar::parse_statements(program).unwrap();
        let mut interpreter = Interpreter::new();
        let return_value = interpreter.evaluate(&block).unwrap();
        assert_eq!(return_value, Value::Number(3.0));
        assert_eq!(*interpreter.get_var("x").unwrap(), Value::Number(1.0));
        assert_eq!(*interpreter.get_var("y").unwrap(), Value::Number(2.0));
    }

    #[test]
    fn basic_program_with_tuples() {
        let program = Span::new("tuple = (1 - 3, 2); (x, _) = tuple;");
        let block = F32Grammar::parse_statements(program).unwrap();
        let mut interpreter = Interpreter::new();
        let return_value = interpreter.evaluate(&block).unwrap();
        assert_eq!(return_value, Value::void());
        assert_eq!(
            *interpreter.get_var("tuple").unwrap(),
            Value::Tuple(vec![Value::Number(-2.0), Value::Number(2.0)])
        );
        assert_eq!(*interpreter.get_var("x").unwrap(), Value::Number(-2.0));
    }

    #[test]
    fn arithmetic_ops_on_tuples() {
        let program = Span::new(
            r#"x = (1, 2) + (3, 4);
            (y, z) = (0, 3) * (2, 0.5) - x;
            u = (1, 2) + 3 * (0.5, z);"#,
        );
        let block = F32Grammar::parse_statements(program).unwrap();
        let mut interpreter = Interpreter::new();
        interpreter.evaluate(&block).unwrap();
        assert_eq!(
            *interpreter.get_var("x").unwrap(),
            Value::Tuple(vec![Value::Number(4.0), Value::Number(6.0)])
        );
        assert_eq!(*interpreter.get_var("y").unwrap(), Value::Number(-4.0));
        assert_eq!(*interpreter.get_var("z").unwrap(), Value::Number(-4.5));
        assert_eq!(
            *interpreter.get_var("u").unwrap(),
            Value::Tuple(vec![Value::Number(2.5), Value::Number(-11.5)])
        );

        let program = "1 / (2, 4)";
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let return_value = interpreter.evaluate(&block).unwrap();
        assert_eq!(
            return_value,
            Value::Tuple(vec![Value::Number(0.5), Value::Number(0.25)])
        );

        let program = "1 / (2, (4, 0.2))";
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let err = interpreter.evaluate(&block).unwrap_err();
        assert_matches!(
            err.source(),
            EvalError::UnexpectedOperand { ref op } if *op == BinaryOp::Div.into()
        );
    }

    #[test]
    fn program_with_blocks() {
        let program = Span::new("z = { x = 1; x + 3 };");
        let block = F32Grammar::parse_statements(program).unwrap();
        let mut interpreter = Interpreter::new();
        let return_value = interpreter.evaluate(&block).unwrap();
        assert_eq!(return_value, Value::void());
        assert_eq!(*interpreter.get_var("z").unwrap(), Value::Number(4.0));
        assert!(interpreter.get_var("x").is_none());
    }

    #[test]
    fn program_with_interpreted_function() {
        let program = Span::new("foo = |x| x + 5; foo(3.0)");
        let block = F32Grammar::parse_statements(program).unwrap();
        let mut interpreter = Interpreter::new();
        let return_value = interpreter.evaluate(&block).unwrap();
        assert_eq!(return_value, Value::Number(8.0));
        assert!(interpreter.get_var("foo").unwrap().is_function());
    }

    #[test]
    fn destructuring_in_fn_args() {
        let program = r#"
            swap = |x, (y, z)| ((x, y), z);
            swap(1, (2, 3))
        "#;
        let program = Span::new(program);
        let block = F32Grammar::parse_statements(program).unwrap();
        let return_value = Interpreter::new().evaluate(&block).unwrap();
        let inner_tuple = Value::Tuple(vec![Value::Number(1.0), Value::Number(2.0)]);
        assert_eq!(
            return_value,
            Value::Tuple(vec![inner_tuple, Value::Number(3.0)])
        );

        let program = r#"
            add = |x, (_, z)| x + z;
            add(1, (2, 3))
        "#;
        let program = Span::new(program);
        let block = F32Grammar::parse_statements(program).unwrap();
        let return_value = Interpreter::new().evaluate(&block).unwrap();
        assert_eq!(return_value, Value::Number(4.0));
    }

    #[test]
    fn captures_in_function() {
        let program = r#"
            x = 5;
            foo = |a| a + x;
            foo(-3)"#;
        let program = Span::new(program);
        let block = F32Grammar::parse_statements(program).unwrap();
        let return_value = Interpreter::new().evaluate(&block).unwrap();
        assert_eq!(return_value, Value::Number(2.0));

        // All captures are by value, so that redefining the captured var does not influence
        // the result.
        let program = r#"
            x = 5;
            foo = |a| a + x;
            x = 10;
            foo(-3)"#;
        let program = Span::new(program);
        let block = F32Grammar::parse_statements(program).unwrap();
        let return_value = Interpreter::new().evaluate(&block).unwrap();
        assert_eq!(return_value, Value::Number(2.0));

        // Functions may be captured as well.
        let program = r#"
            add = |x, y| x + y;
            foo = |a| add(a, 5);
            add = 0;
            foo(-3)"#;
        let program = Span::new(program);
        let block = F32Grammar::parse_statements(program).unwrap();
        let return_value = Interpreter::new().evaluate(&block).unwrap();
        assert_eq!(return_value, Value::Number(2.0));
    }

    #[test]
    fn captured_function() {
        let program = r#"
            gen = |op| { |u, v| op(u, v) - op(v, u) };
            add = gen(|x, y| x + y);
            add((1, 2), (3, 4))
        "#;
        let program = Span::new(program);
        let block = F32Grammar::parse_statements(program).unwrap();
        let mut interpreter = Interpreter::new();
        let return_value = interpreter.evaluate(&block).unwrap();
        assert_eq!(
            return_value,
            Value::Tuple(vec![Value::Number(0.0), Value::Number(0.0)])
        );

        let add = interpreter.get_var("add").unwrap();
        let add = match add {
            Value::Function(Function::Interpreted(function)) => function,
            other => panic!("Unexpected `add` value: {:?}", other),
        };
        assert_eq!(add.captures.variables.len(), 1);
        assert_matches!(add.captures.variables["op"], Value::Function(_));

        let program = r#"
            div = gen(|x, y| x / y);
            div(1, 2) == -1.5 # 1/2 - 2/1
        "#;
        let program = Span::new(program);
        let block = F32Grammar::parse_statements(program).unwrap();
        let return_value = interpreter.evaluate(&block).unwrap();
        assert_eq!(return_value, Value::Bool(true));
    }

    #[test]
    fn indirectly_captured_function() {
        let program = r#"
            gen = {
                div = |x, y| x / y;
                |u| { |v| div(u, v) - div(v, u) }
            };
            fn = gen(4);
            fn(1) == 3.75
        "#;
        let program = Span::new(program);
        let block = F32Grammar::parse_statements(program).unwrap();
        let mut interpreter = Interpreter::new();
        let return_value = interpreter.evaluate(&block).unwrap();
        assert_eq!(return_value, Value::Bool(true));

        // Check that `div` is captured both by the external and internal functions.
        let functions = [
            interpreter.get_var("fn").unwrap(),
            interpreter.get_var("gen").unwrap(),
        ];
        for function in &functions {
            let function = match function {
                Value::Function(Function::Interpreted(function)) => function,
                other => panic!("Unexpected `fn` value: {:?}", other),
            };
            assert!(function.captures.get_var("div").unwrap().is_function());
        }
    }

    #[test]
    fn captured_var_in_returned_fn() {
        let program = r#"
            gen = |x| {
                y = (x, x^2);
                # Check that `x` below is not taken from the arg above, but rather
                # from the function argument. `y` though should be captured
                # from the surrounding function.
                |x| y - (x, x^2)
            };
            foo = gen(2);
            foo(1) == (1, 3) && foo(2) == (0, 0) && foo(3) == (-1, -5)
        "#;
        let program = Span::new(program);
        let block = F32Grammar::parse_statements(program).unwrap();
        let return_value = Interpreter::new().evaluate(&block).unwrap();
        assert_eq!(return_value, Value::Bool(true));
    }

    #[test]
    fn embedded_function() {
        let program = r#"
            gen_add = |x| |y| x + y;
            add = gen_add(5.0);
            add(-3) + add(-5)
        "#;
        let program = Span::new(program);
        let block = F32Grammar::parse_statements(program).unwrap();
        let mut interpreter = Interpreter::new();
        let return_value = interpreter.evaluate(&block).unwrap();
        assert_eq!(return_value, Value::Number(2.0));

        let program = Span::new("add = gen_add(-3); add(-1)");
        let block = F32Grammar::parse_statements(program).unwrap();
        let return_value = interpreter.evaluate(&block).unwrap();
        assert_eq!(return_value, Value::Number(-4.0));

        let function = match interpreter.get_var("add").unwrap() {
            Value::Function(Function::Interpreted(function)) => function,
            other => panic!("Unexpected `add` value: {:?}", other),
        };
        let captures: Vec<_> = function.captures.variables().collect();
        assert_eq!(captures, [("x", &Value::Number(-3.0))]);
    }

    #[test]
    fn first_class_functions() {
        let program = r#"
            apply = |fn, x, y| (fn(x), fn(y));
            apply(|x| x + 3, 1, -2)"#;
        let program = Span::new(program);
        let block = F32Grammar::parse_statements(program).unwrap();
        let mut interpreter = Interpreter::new();
        let return_value = interpreter.evaluate(&block).unwrap();
        assert_eq!(
            return_value,
            Value::Tuple(vec![Value::Number(4.0), Value::Number(1.0)])
        );

        let program = r#"
            repeat = |fn, x| fn(fn(fn(x)));
            a = repeat(|x| x * 2, 1);
            b = {
                lambda = |x| x / 2 - 1;
                repeat(lambda, 2)
            };
            (a, b)"#;
        let program = Span::new(program);
        let block = F32Grammar::parse_statements(program).unwrap();
        let return_value = interpreter.evaluate(&block).unwrap();
        assert_eq!(
            return_value,
            Value::Tuple(vec![Value::Number(8.0), Value::Number(-1.5)])
        );
        assert!(interpreter.get_var("lambda").is_none());
    }

    #[test]
    fn immediately_executed_function() {
        let mut interpreter = Interpreter::new();
        let program = "-|x| { x + 5 }(-3)";
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let return_value = interpreter.evaluate(&block).unwrap();
        assert_eq!(return_value, Value::Number(-2.0));

        let program = "2 + |x| { x + 5 }(-3)";
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let return_value = interpreter.evaluate(&block).unwrap();
        assert_eq!(return_value, Value::Number(4.0));

        let program = "add = |x, y| x + y; add(10, |x| { x + 5 }(-3))";
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let return_value = interpreter.evaluate(&block).unwrap();
        assert_eq!(return_value, Value::Number(12.0));
    }

    #[test]
    fn program_with_native_function() {
        let mut interpreter = Interpreter::new();
        interpreter.innermost_scope().insert_native_fn("sin", SIN);

        let program = Span::new("sin(1.0) - 3");
        let block = F32Grammar::parse_statements(program).unwrap();
        let return_value = interpreter.evaluate(&block).unwrap();
        assert_eq!(return_value, Value::Number(1.0_f32.sin() - 3.0));
    }

    #[test]
    fn function_aliasing() {
        let program = "alias = sin; alias(1.0)";
        let program = Span::new(program);
        let block = F32Grammar::parse_statements(program).unwrap();
        let mut interpreter = Interpreter::new();
        interpreter.innermost_scope().insert_native_fn("sin", SIN);
        let return_value = interpreter.evaluate(&block).unwrap();
        assert_eq!(return_value, Value::Number(1.0_f32.sin()));

        let sin = interpreter.get_var("sin").unwrap();
        let sin = match sin {
            Value::Function(Function::Native(function)) => function,
            _ => panic!("Unexpected `sin` value: {:?}", sin),
        };
        let alias = interpreter.get_var("alias").unwrap();
        let alias = match alias {
            Value::Function(Function::Native(function)) => function,
            _ => panic!("Unexpected `alias` value: {:?}", alias),
        };
        assert!(Rc::ptr_eq(sin, alias));
    }

    #[test]
    fn undefined_var() {
        let program = "x + 3";
        let program = Span::new(program);
        let block = F32Grammar::parse_statements(program).unwrap();
        let err = Interpreter::new().evaluate(&block).unwrap_err();
        assert_eq!(err.main_span().fragment, "x");
        assert_matches!(err.source(), EvalError::Undefined(ref var) if var == "x");
    }

    #[test]
    fn undefined_function() {
        let program = "1 + sin(-5.0)";
        let program = Span::new(program);
        let block = F32Grammar::parse_statements(program).unwrap();
        let err = Interpreter::new().evaluate(&block).unwrap_err();
        assert_eq!(err.main_span().fragment, "sin");
        assert_matches!(err.source(), EvalError::Undefined(ref var) if var == "sin");
    }

    #[test]
    fn arg_len_mismatch() {
        let mut interpreter = Interpreter::new();
        let program = Span::new("foo = |x| x + 5; foo()");
        let block = F32Grammar::parse_statements(program).unwrap();
        let err = interpreter.evaluate(&block).unwrap_err();
        assert_eq!(err.main_span().fragment, "foo()");
        assert_matches!(err.source(), EvalError::ArgsLenMismatch { def: 1, call: 0 });

        let program = Span::new("foo(1, 2) * 3.0");
        let block = F32Grammar::parse_statements(program).unwrap();
        let err = interpreter.evaluate(&block).unwrap_err();
        assert_eq!(err.main_span().fragment, "foo(1, 2)");
        assert_matches!(err.source(), EvalError::ArgsLenMismatch { def: 1, call: 2 });
    }

    #[test]
    fn repeated_args_in_fn_definition() {
        let mut interpreter = Interpreter::new();

        let program = Span::new("add = |x, x| x + 2;");
        let block = F32Grammar::parse_statements(program).unwrap();
        let err = interpreter.evaluate(&block).unwrap_err();
        assert_eq!(err.main_span().fragment, "x");
        assert_eq!(err.main_span().offset, 10);
        assert_matches!(
            err.source(),
            EvalError::RepeatedAssignment {
                context: RepeatedAssignmentContext::FnArgs
            }
        );

        let program = Span::new("add = |x, (y, x)| x + y;");
        let block = F32Grammar::parse_statements(program).unwrap();
        let err = interpreter.evaluate(&block).unwrap_err();
        assert_eq!(err.main_span().fragment, "x");
        assert_eq!(err.main_span().offset, 14);
        assert_matches!(err.source(), EvalError::RepeatedAssignment { .. });
    }

    #[test]
    fn repeated_var_in_lvalue() {
        let program = Span::new("(x, x) = (1, 2);");
        let block = F32Grammar::parse_statements(program).unwrap();
        let err = Interpreter::new().evaluate(&block).unwrap_err();
        assert_eq!(err.main_span().fragment, "x");
        assert_eq!(err.main_span().offset, 4);
        assert_matches!(
            err.source(),
            EvalError::RepeatedAssignment {
                context: RepeatedAssignmentContext::Assignment,
            }
        );
    }

    #[test]
    fn error_in_function_args() {
        let program = r#"
            add = |x, (_, z)| x + z;
            add(1, 2)
        "#;
        let program = Span::new(program);
        let block = F32Grammar::parse_statements(program).unwrap();

        let err = Interpreter::new().evaluate(&block).unwrap_err();
        assert_eq!(err.main_span().fragment, "(_, z)");
        assert_matches!(err.source(), EvalError::CannotDestructure);
    }

    #[test]
    fn cannot_call_error() {
        let program = Span::new("x = 5; x(1.0)");
        let block = F32Grammar::parse_statements(program).unwrap();
        let err = Interpreter::new().evaluate(&block).unwrap_err();
        assert_eq!(err.main_span().offset, 7);
        assert_matches!(err.source(), EvalError::CannotCall);

        let program = Span::new("2 + 1.0(5)");
        let block = F32Grammar::parse_statements(program).unwrap();
        let err = Interpreter::new().evaluate(&block).unwrap_err();
        assert_eq!(err.main_span().fragment, "1.0(5)");
        assert_matches!(err.source(), EvalError::CannotCall);
    }

    #[test]
    fn tuple_len_mismatch_error() {
        let program = "x = (1, 2) + (3, 4, 5);";
        let program = Span::new(program);
        let block = F32Grammar::parse_statements(program).unwrap();
        let err = Interpreter::new().evaluate(&block).unwrap_err();
        assert_eq!(err.main_span().fragment, "(1, 2) + (3, 4, 5)");
        assert_matches!(err.source(), EvalError::TupleLenMismatch { lhs: 2, rhs: 3, .. });
    }

    #[test]
    fn cannot_destructure_error() {
        let program = "(x, y) = 1.0;";
        let program = Span::new(program);
        let block = F32Grammar::parse_statements(program).unwrap();
        let err = Interpreter::new().evaluate(&block).unwrap_err();
        assert_eq!(err.main_span().fragment, "(x, y)");
        assert_matches!(err.source(), EvalError::CannotDestructure);
    }

    #[test]
    fn unexpected_operand() {
        let mut interpreter = Interpreter::new();

        let program = Span::new("1 / || 2");
        let block = F32Grammar::parse_statements(program).unwrap();
        let err = interpreter.evaluate(&block).unwrap_err();
        assert_eq!(err.main_span().fragment, "|| 2");
        assert_matches!(
            err.source(),
            EvalError::UnexpectedOperand { ref op } if *op == BinaryOp::Div.into()
        );

        let program = Span::new("1 == 1 && !(2, 3)");
        let block = F32Grammar::parse_statements(program).unwrap();
        let err = interpreter.evaluate(&block).unwrap_err();
        assert_eq!(err.main_span().fragment, "!(2, 3)");
        assert_matches!(
            err.source(),
            EvalError::UnexpectedOperand { ref op } if *op == UnaryOp::Not.into()
        );

        let program = Span::new("|x| { x + 5 } + 10");
        let block = F32Grammar::parse_statements(program).unwrap();
        let err = interpreter.evaluate(&block).unwrap_err();
        assert_matches!(
            err.source(),
            EvalError::UnexpectedOperand { ref op } if *op == BinaryOp::Add.into()
        );
    }

    #[test]
    fn native_fn_error() {
        let mut interpreter = Interpreter::new();
        interpreter.innermost_scope().insert_native_fn("sin", SIN);

        let program = Span::new("1 + sin(-5.0, 2.0)");
        let block = F32Grammar::parse_statements(program).unwrap();
        let err = interpreter.evaluate(&block).unwrap_err();
        assert_eq!(err.main_span().fragment, "sin(-5.0, 2.0)");
        assert_matches!(err.source(), EvalError::ArgsLenMismatch { def: 1, call: 2 });

        let program = Span::new("1 + sin((-5, 2))");
        let block = F32Grammar::parse_statements(program).unwrap();
        let err = interpreter.evaluate(&block).unwrap_err();
        assert_eq!(err.main_span().fragment, "sin((-5, 2))");
        assert_matches!(
            err.source(),
            EvalError::NativeCall(ref msg) if msg.contains("requires one primitive argument")
        );
    }
}
