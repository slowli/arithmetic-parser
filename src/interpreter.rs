//! Simple interpreter for ASTs produced by the parser.
//!
//! # Examples
//!
//! ```
//! use arithmetic_parser::{
//!     grammars::F32Grammar,
//!     interpreter::{Assert, BinaryFn, Context, Value},
//!     Grammar, GrammarExt, Span,
//! };
//!
//! const MIN: BinaryFn<fn(f32, f32) -> f32> =
//!     BinaryFn::new(|x, y| if x < y { x } else { y });
//! const MAX: BinaryFn<fn(f32, f32) -> f32> =
//!     BinaryFn::new(|x, y| if x > y { x } else { y });
//!
//! let mut context = Context::new();
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
//! assert_eq!(ret, Value::Simple(9.0));
//! ```

use num_traits::{Num, Pow};

use std::{
    collections::{HashMap, HashSet},
    fmt, ops,
    rc::Rc,
};

use crate::{
    helpers::{create_span, create_span_ref},
    BinaryOp, Block, Expr, FnDefinition, Grammar, Lvalue, Op, Span, Spanned, SpannedExpr,
    SpannedLvalue, SpannedStatement, Statement, UnaryOp,
};

mod functions;
pub use self::functions::{Assert, BinaryFn, UnaryFn};

/// Errors that can occur during interpreting expressions and statements.
#[derive(Debug)]
pub enum EvalError {
    /// Mismatch between length of tuples in a binary operation or assignment.
    TupleLenMismatch {
        /// Length of a tuple on the left-hand side.
        lhs: usize,
        /// Length of a tuple on the right-hand side.
        rhs: usize,
    },
    /// Cannot destructure a no-tuple variable.
    CannotDestructure,
    /// Variable with the enclosed name is not defined.
    Undefined(String),
    /// Variable with the enclosed name is not callable (i.e., is not a function).
    CannotCall(String),
    /// Error during execution of a native function.
    NativeCall(anyhow::Error),
    /// Embedded function definitions are not yet supported by the interpreter.
    EmbeddedFunction,
    /// Unexpected operand type(s) for the specified operation.
    UnexpectedOperand {
        /// Operation which failed.
        op: Op,
    },
}

/// Function on zero or more `Value`s.
pub trait NativeFn<T: Grammar> {
    /// Executes the function on the specified arguments.
    fn execute<'a>(&self, args: &[Value<'a, T>]) -> anyhow::Result<Value<'a, T>>;
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
        context: &Context<'a, T>,
    ) -> Result<Self, Spanned<'a, EvalError>> {
        let captures = FnContext::captures(&definition, context)?;
        Ok(Self {
            definition,
            captures,
        })
    }
}

/// Helper context for symbolic execution of a function body in order to determine
/// variables captured by the function.
#[derive(Debug)]
struct FnContext<'a, T>
where
    T: Grammar,
{
    local_vars: HashSet<&'a str>,
    captures: Scope<'a, T>,
}

impl<'a, T: Grammar> FnContext<'a, T> {
    /// Collects variables captured by the function into a single `Scope`.
    fn captures(
        definition: &Spanned<'a, FnDefinition<'a, T>>,
        context: &Context<'a, T>,
    ) -> Result<Scope<'a, T>, Spanned<'a, EvalError>> {
        let mut fn_context = Self {
            local_vars: HashSet::new(),
            captures: Scope::new(),
        };

        for arg in &definition.extra.args {
            fn_context.set_local_vars(arg);
        }
        for statement in &definition.extra.body.statements {
            fn_context.eval_statement(statement, context)?;
        }
        if let Some(ref return_expr) = definition.extra.body.return_value {
            fn_context.eval(return_expr, context)?;
        }

        Ok(fn_context.captures)
    }

    /// Extracts local variables from the provided lvalue.
    fn set_local_vars(&mut self, lvalue: &SpannedLvalue<'a, T::Type>) {
        match lvalue.extra {
            Lvalue::Variable { .. } => {
                if lvalue.fragment != "_" {
                    self.local_vars.insert(lvalue.fragment);
                }
            }
            Lvalue::Tuple(ref fragments) => {
                for fragment in fragments {
                    self.set_local_vars(fragment);
                }
            }
        }
    }

    /// Processes a local variable in the rvalue position.
    fn eval_local_var(
        &mut self,
        var_name: &str,
        context: &Context<'a, T>,
    ) -> Result<(), EvalError> {
        if self.local_vars.contains(var_name) || self.captures.contains_var(var_name) {
            // No action needs to be performed.
        } else if let Some(val) = context.get_var(var_name) {
            self.captures.insert_var(var_name, val.clone());
        } else {
            return Err(EvalError::Undefined(var_name.to_owned()));
        }

        Ok(())
    }

    /// Evaluates an expression using the provided context.
    fn eval(
        &mut self,
        expr: &SpannedExpr<'a, T>,
        context: &Context<'a, T>,
    ) -> Result<(), Spanned<'a, EvalError>> {
        match &expr.extra {
            Expr::Variable => {
                let var_name = expr.fragment;
                self.eval_local_var(var_name, context)
                    .map_err(|e| create_span_ref(expr, e))?;
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
                    .map_err(|e| create_span_ref(name, e))?;
            }

            Expr::Block(block) => {
                for statement in &block.statements {
                    self.eval_statement(statement, context)?;
                }
                if let Some(ref return_expr) = block.return_value {
                    self.eval(return_expr, context)?;
                }
            }

            Expr::FnDefinition(_) => {
                return Err(create_span_ref(expr, EvalError::EmbeddedFunction));
            }
        }
        Ok(())
    }

    /// Evaluates a statement using the provided context.
    fn eval_statement(
        &mut self,
        statement: &SpannedStatement<'a, T>,
        context: &Context<'a, T>,
    ) -> Result<(), Spanned<'a, EvalError>> {
        match &statement.extra {
            Statement::Expr(expr) => self.eval(expr, context),
            Statement::Assignment { lhs, rhs } => {
                self.eval(rhs, context)?;
                self.set_local_vars(lhs);
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

/// Values produced by expressions during their interpretation.
#[derive(Debug)]
pub enum Value<'a, T>
where
    T: Grammar,
{
    /// Primitive value: a single literal.
    Simple(T::Lit),
    /// Boolean value.
    Bool(bool),
    /// Function.
    Function(Function<'a, T>),
    /// Tuple of zero or more values.
    Tuple(Vec<Value<'a, T>>),
}

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
            Self::Simple(lit) => Self::Simple(lit.clone()),
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
            (Self::Simple(this), Self::Simple(other)) => this == other,
            (Self::Tuple(this), Self::Tuple(other)) => this == other,
            _ => false,
        }
    }
}

impl<T> Value<'_, T>
where
    T: Grammar,
    T::Lit: Num + ops::Neg<Output = T::Lit> + Pow<T::Lit, Output = T::Lit>,
{
    fn try_binary_op(
        self,
        rhs: Self,
        op: BinaryOp,
        primitive_op: fn(T::Lit, T::Lit) -> T::Lit,
    ) -> Result<Self, EvalError> {
        match (self, rhs) {
            (Self::Simple(this), Self::Simple(other)) => {
                Ok(Self::Simple(primitive_op(this, other)))
            }
            (Self::Tuple(this), Self::Tuple(other)) => {
                if this.len() == other.len() {
                    let res: Result<Vec<_>, _> = this
                        .into_iter()
                        .zip(other)
                        .map(|(x, y)| x.try_binary_op(y, op, primitive_op))
                        .collect();
                    res.map(Self::Tuple)
                } else {
                    Err(EvalError::TupleLenMismatch {
                        lhs: this.len(),
                        rhs: other.len(),
                    })
                }
            }

            _ => Err(EvalError::UnexpectedOperand { op: Op::Binary(op) }),
        }
    }

    fn try_add(self, rhs: Self) -> Result<Self, EvalError> {
        self.try_binary_op(rhs, BinaryOp::Add, |x, y| x + y)
    }

    fn try_sub(self, rhs: Self) -> Result<Self, EvalError> {
        self.try_binary_op(rhs, BinaryOp::Sub, |x, y| x - y)
    }

    fn try_mul(self, rhs: Self) -> Result<Self, EvalError> {
        self.try_binary_op(rhs, BinaryOp::Mul, |x, y| x * y)
    }

    fn try_div(self, rhs: Self) -> Result<Self, EvalError> {
        self.try_binary_op(rhs, BinaryOp::Div, |x, y| x / y)
    }

    fn try_pow(self, rhs: Self) -> Result<Self, EvalError> {
        self.try_binary_op(rhs, BinaryOp::Power, |x, y| x.pow(y))
    }

    fn try_neg(self) -> Result<Self, EvalError> {
        match self {
            Self::Simple(val) => Ok(Self::Simple(-val)),
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
            (Self::Simple(this), Self::Simple(other)) => Ok(this == other),
            (Self::Tuple(this), Self::Tuple(other)) => {
                if this.len() == other.len() {
                    this.into_iter()
                        .zip(other)
                        .try_fold(true, |acc, (x, y)| Ok(acc && x.try_compare(y)?))
                } else {
                    Err(EvalError::TupleLenMismatch {
                        lhs: this.len(),
                        rhs: other.len(),
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

    pub(crate) fn assign<'lv>(
        &mut self,
        lvalue: &SpannedLvalue<'lv, T::Type>,
        rvalue: Value<'a, T>,
    ) -> Result<(), Spanned<'lv, EvalError>> {
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
                        return Err(create_span_ref(
                            lvalue,
                            EvalError::TupleLenMismatch {
                                lhs: assignments.len(),
                                rhs: fragments.len(),
                            },
                        ));
                    }

                    for (assignment, fragment) in assignments.iter().zip(fragments) {
                        self.assign(assignment, fragment)?;
                    }
                } else {
                    return Err(create_span_ref(lvalue, EvalError::CannotDestructure));
                }
            }
        }
        Ok(())
    }
}

/// Call backtrace.
#[derive(Debug, Default)]
pub struct Backtrace<'a> {
    calls: Vec<BacktraceElement<'a>>,
}

/// Function call.
#[derive(Debug, Clone, Copy)]
pub struct BacktraceElement<'a> {
    /// Function name.
    pub fn_name: &'a str,
    /// Code span of the function definition.
    pub def_span: Span<'a>,
    /// Code span of the function call.
    pub call_span: Span<'a>,
}

impl<'a> Backtrace<'a> {
    /// Iterates over the backtrace, starting from the most recent call.
    pub fn calls(&self) -> impl Iterator<Item = BacktraceElement<'a>> + '_ {
        self.calls.iter().rev().cloned()
    }

    /// Appends a function call into the backtrace.
    fn push_call(&mut self, fn_name: &'a str, def_span: Span<'a>, call_span: Span<'a>) {
        self.calls.push(BacktraceElement {
            fn_name,
            def_span,
            call_span,
        });
    }

    /// Pops a function call.
    fn pop_call(&mut self) {
        self.calls.pop();
    }
}

/// Error with the associated backtrace.
#[derive(Debug)]
pub struct ErrorWithBacktrace<'a> {
    /// Error.
    pub inner: Spanned<'a, EvalError>,
    /// Backtrace information.
    pub backtrace: Backtrace<'a>,
}

/// Stack of variable scopes that can be used to evaluate `Statement`s.
#[derive(Debug)]
pub struct Context<'a, T: Grammar> {
    scopes: Vec<Scope<'a, T>>,
}

impl<T: Grammar> Default for Context<'_, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T: Grammar> Context<'a, T> {
    /// Creates a new context.
    pub fn new() -> Self {
        Self {
            scopes: vec![Scope::new()],
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

impl<'a, T: Grammar> Context<'a, T>
where
    T::Lit: Num + ops::Neg<Output = T::Lit> + Pow<T::Lit, Output = T::Lit>,
{
    fn evaluate_fn(
        &mut self,
        expr: &SpannedExpr<'a, T>,
        func: &InterpretedFn<'a, T>,
        args: &[Value<'a, T>],
        backtrace: &mut Option<Backtrace<'a>>,
    ) -> Result<Value<'a, T>, Spanned<'a, EvalError>> {
        let name = match expr.extra {
            Expr::Function { name, .. } => name,
            _ => unreachable!(),
        };
        let def = &func.definition.extra;

        if let Some(backtrace) = backtrace {
            backtrace.push_call(
                &name.fragment[1..],
                create_span_ref(&func.definition, ()),
                create_span_ref(expr, ()),
            );
        }
        self.scopes.push(func.captures.clone());
        for (lvalue, val) in def.args.iter().zip(args) {
            self.innermost_scope().assign(lvalue, val.clone())?;
        }
        let result = self.evaluate_inner(&def.body, backtrace);

        if result.is_ok() {
            if let Some(backtrace) = backtrace {
                backtrace.pop_call();
            }
        }
        self.pop_scope();
        result
    }

    fn evaluate_binary_expr(
        &mut self,
        expr_span: Span<'a>,
        spanned_lhs: &SpannedExpr<'a, T>,
        spanned_rhs: &SpannedExpr<'a, T>,
        op: Spanned<'a, BinaryOp>,
        backtrace: &mut Option<Backtrace<'a>>,
    ) -> Result<Value<'a, T>, Spanned<'a, EvalError>> {
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
                    return Err(create_span_ref(
                        spanned_lhs,
                        EvalError::UnexpectedOperand {
                            op: op.extra.into(),
                        },
                    ));
                }
            },

            _ => { /* do nothing yet */ }
        }

        let rhs = self.evaluate_expr_inner(spanned_rhs, backtrace)?;

        match op.extra {
            BinaryOp::Add => lhs.try_add(rhs).map_err(|e| create_span(expr_span, e)),
            BinaryOp::Sub => lhs.try_sub(rhs).map_err(|e| create_span(expr_span, e)),
            BinaryOp::Mul => lhs.try_mul(rhs).map_err(|e| create_span(expr_span, e)),
            BinaryOp::Div => lhs.try_div(rhs).map_err(|e| create_span(expr_span, e)),
            BinaryOp::Power => lhs.try_pow(rhs).map_err(|e| create_span(expr_span, e)),

            BinaryOp::Eq | BinaryOp::NotEq => {
                let eq = lhs
                    .try_compare(rhs)
                    .map_err(|e| create_span(expr_span, e))?;
                Ok(Value::Bool(if op.extra == BinaryOp::Eq { eq } else { !eq }))
            }

            BinaryOp::And | BinaryOp::Or => {
                match rhs {
                    // This works since we know that AND / OR hasn't short-circuited.
                    Value::Bool(b) => Ok(Value::Bool(b)),

                    _ => {
                        let err = EvalError::UnexpectedOperand {
                            op: op.extra.into(),
                        };
                        Err(create_span_ref(spanned_rhs, err))
                    }
                }
            }
        }
    }

    fn evaluate_expr_inner(
        &mut self,
        expr: &SpannedExpr<'a, T>,
        backtrace: &mut Option<Backtrace<'a>>,
    ) -> Result<Value<'a, T>, Spanned<'a, EvalError>> {
        match &expr.extra {
            Expr::Variable => self.get_var(expr.fragment).cloned().ok_or_else(|| {
                create_span_ref(expr, EvalError::Undefined(expr.fragment.to_owned()))
            }),
            Expr::Literal(value) => Ok(Value::Simple(value.to_owned())),

            Expr::Tuple(fragments) => {
                let fragments: Result<Vec<_>, _> = fragments
                    .iter()
                    .map(|frag| self.evaluate_expr_inner(frag, backtrace))
                    .collect();
                fragments.map(Value::Tuple)
            }

            Expr::Function { name, ref args } => {
                let args: Result<Vec<_>, _> = args
                    .iter()
                    .map(|arg| self.evaluate_expr_inner(arg, backtrace))
                    .collect();
                let args = args?;

                let resolved_name = name.fragment;
                if let Some(func) = self.get_var(resolved_name).cloned() {
                    let func = match func {
                        Value::Function(func) => func,
                        _ => {
                            let err = EvalError::CannotCall(resolved_name.to_owned());
                            return Err(create_span_ref(expr, err));
                        }
                    };

                    match func {
                        Function::Interpreted(ref func) => {
                            self.evaluate_fn(expr, func, &args, backtrace)
                        }
                        Function::Native(ref func) => func
                            .execute(&args)
                            .map_err(|e| create_span_ref(expr, EvalError::NativeCall(e))),
                    }
                } else {
                    Err(create_span(
                        *name,
                        EvalError::Undefined(resolved_name.to_owned()),
                    ))
                }
            }

            // Arithmetic operations
            Expr::Unary { inner, op } => {
                let val = self.evaluate_expr_inner(inner, backtrace)?;

                match op.extra {
                    UnaryOp::Not => val.try_not().map_err(|e| create_span_ref(expr, e)),
                    UnaryOp::Neg => val.try_neg().map_err(|e| create_span_ref(expr, e)),
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
            .map_err(|e| ErrorWithBacktrace {
                inner: e,
                backtrace: backtrace.unwrap(),
            })
    }

    /// Evaluates a list of statements.
    fn evaluate_inner(
        &mut self,
        block: &Block<'a, T>,
        backtrace: &mut Option<Backtrace<'a>>,
    ) -> Result<Value<'a, T>, Spanned<'a, EvalError>> {
        use crate::Statement::*;

        for statement in &block.statements {
            match &statement.extra {
                Expr(expr) => {
                    self.evaluate_expr_inner(expr, backtrace)?;
                }

                Assignment { ref lhs, ref rhs } => {
                    let evaluated = self.evaluate_expr_inner(rhs, backtrace)?;
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
    pub fn evaluate(&mut self, block: &Block<'a, T>) -> Result<Value<T>, ErrorWithBacktrace<'a>> {
        let mut backtrace = Some(Backtrace::default());
        self.evaluate_inner(block, &mut backtrace)
            .map_err(|e| ErrorWithBacktrace {
                inner: e,
                backtrace: backtrace.unwrap(),
            })
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
        let mut context = Context::new();
        let return_value = context.evaluate(&block).unwrap();
        assert_eq!(return_value, Value::Simple(3.0));
        assert_eq!(*context.get_var("x").unwrap(), Value::Simple(1.0));
        assert_eq!(*context.get_var("y").unwrap(), Value::Simple(2.0));
    }

    #[test]
    fn basic_program_with_tuples() {
        let program = Span::new("tuple = (1 - 3, 2); (x, _) = tuple;");
        let block = F32Grammar::parse_statements(program).unwrap();
        let mut context = Context::new();
        let return_value = context.evaluate(&block).unwrap();
        assert_eq!(return_value, Value::void());
        assert_eq!(
            *context.get_var("tuple").unwrap(),
            Value::Tuple(vec![Value::Simple(-2.0), Value::Simple(2.0)])
        );
        assert_eq!(*context.get_var("x").unwrap(), Value::Simple(-2.0));
    }

    #[test]
    fn arithmetic_ops_on_tuples() {
        let program = Span::new(
            r#"x = (1, 2) + (3, 4);
            (y, z) = (0, 3) * (2, 0.5) - x;
            # u = (1, 2) + 3 * (0.5, z);"#,
        );
        let block = F32Grammar::parse_statements(program).unwrap();
        let mut context = Context::new();
        context.evaluate(&block).unwrap();
        assert_eq!(
            *context.get_var("x").unwrap(),
            Value::Tuple(vec![Value::Simple(4.0), Value::Simple(6.0)])
        );
        assert_eq!(*context.get_var("y").unwrap(), Value::Simple(-4.0));
        assert_eq!(*context.get_var("z").unwrap(), Value::Simple(-4.5));
    }

    #[test]
    fn program_with_blocks() {
        let program = Span::new("z = { x = 1; x + 3 };");
        let block = F32Grammar::parse_statements(program).unwrap();
        let mut context = Context::new();
        let return_value = context.evaluate(&block).unwrap();
        assert_eq!(return_value, Value::void());
        assert_eq!(*context.get_var("z").unwrap(), Value::Simple(4.0));
        assert!(context.get_var("x").is_none());
    }

    #[test]
    fn program_with_interpreted_function() {
        let program = Span::new("foo = |x| x + 5; foo(3.0)");
        let block = F32Grammar::parse_statements(program).unwrap();
        let mut context = Context::new();
        let return_value = context.evaluate(&block).unwrap();
        assert_eq!(return_value, Value::Simple(8.0));
        assert!(context.get_var("foo").unwrap().is_function());
    }

    #[test]
    fn captures_in_function() {
        let program = r#"
            x = 5;
            foo = |a| a + x;
            foo(-3)"#;
        let program = Span::new(program);
        let block = F32Grammar::parse_statements(program).unwrap();
        let mut context = Context::new();
        let return_value = context.evaluate(&block).unwrap();
        assert_eq!(return_value, Value::Simple(2.0));

        // All captures are by value, so that redefining the captured var does not influence
        // the result.
        let program = r#"
            x = 5;
            foo = |a| a + x;
            x = 10;
            foo(-3)"#;
        let program = Span::new(program);
        let block = F32Grammar::parse_statements(program).unwrap();
        let mut context = Context::new();
        let return_value = context.evaluate(&block).unwrap();
        assert_eq!(return_value, Value::Simple(2.0));

        // Functions may be captured as well.
        let program = r#"
            add = |x, y| x + y;
            foo = |a| add(a, 5);
            add = 0;
            foo(-3)"#;
        let program = Span::new(program);
        let block = F32Grammar::parse_statements(program).unwrap();
        let mut context = Context::new();
        let return_value = context.evaluate(&block).unwrap();
        assert_eq!(return_value, Value::Simple(2.0));
    }

    #[test]
    fn first_class_functions() {
        let program = r#"
            apply = |fn, x, y| (fn(x), fn(y));
            apply(|x| x + 3, 1, -2)"#;
        let program = Span::new(program);
        let block = F32Grammar::parse_statements(program).unwrap();
        let mut context = Context::new();
        let return_value = context.evaluate(&block).unwrap();
        assert_eq!(
            return_value,
            Value::Tuple(vec![Value::Simple(4.0), Value::Simple(1.0)])
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
        let return_value = context.evaluate(&block).unwrap();
        assert_eq!(
            return_value,
            Value::Tuple(vec![Value::Simple(8.0), Value::Simple(-1.5)])
        );
        assert!(context.get_var("lambda").is_none());
    }

    #[test]
    fn program_with_native_function() {
        let mut context = Context::new();
        context.innermost_scope().insert_native_fn("sin", SIN);

        let program = Span::new("sin(1.0) - 3");
        let block = F32Grammar::parse_statements(program).unwrap();
        let return_value = context.evaluate(&block).unwrap();
        assert_eq!(return_value, Value::Simple(1.0_f32.sin() - 3.0));
    }

    #[test]
    fn function_aliasing() {
        let program = "alias = sin; alias(1.0)";
        let program = Span::new(program);
        let block = F32Grammar::parse_statements(program).unwrap();
        let mut context = Context::new();
        context.innermost_scope().insert_native_fn("sin", SIN);
        let return_value = context.evaluate(&block).unwrap();
        assert_eq!(return_value, Value::Simple(1.0_f32.sin()));

        let sin = context.get_var("sin").unwrap();
        let sin = match sin {
            Value::Function(Function::Native(function)) => function,
            _ => panic!("Unexpected `sin` value: {:?}", sin),
        };
        let alias = context.get_var("alias").unwrap();
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
        let mut context = Context::new();
        let err = context.evaluate(&block).unwrap_err();
        assert_eq!(err.inner.fragment, "x");
        assert_matches!(err.inner.extra, EvalError::Undefined(ref var) if var == "x");
    }

    #[test]
    fn undefined_function() {
        let program = "1 + sin(-5.0)";
        let program = Span::new(program);
        let block = F32Grammar::parse_statements(program).unwrap();
        let mut context = Context::new();
        let err = context.evaluate(&block).unwrap_err();
        assert_eq!(err.inner.fragment, "sin");
        assert_matches!(err.inner.extra, EvalError::Undefined(ref var) if var == "sin");
    }

    #[test]
    fn cannot_call_error() {
        let program = "x = 5; x(1.0)";
        let program = Span::new(program);
        let block = F32Grammar::parse_statements(program).unwrap();
        let mut context = Context::new();
        let err = context.evaluate(&block).unwrap_err();
        assert_eq!(err.inner.offset, 7);
        assert_matches!(err.inner.extra, EvalError::CannotCall(ref var) if var == "x");
    }

    #[test]
    fn tuple_len_mismatch_error() {
        let program = "x = (1, 2) + (3, 4, 5);";
        let program = Span::new(program);
        let block = F32Grammar::parse_statements(program).unwrap();
        let mut context = Context::new();
        let err = context.evaluate(&block).unwrap_err();
        assert_eq!(err.inner.fragment, "(1, 2) + (3, 4, 5)");
        assert_matches!(
            err.inner.extra,
            EvalError::TupleLenMismatch { lhs: 2, rhs: 3 }
        );
    }

    #[test]
    fn cannot_destructure_error() {
        let program = "(x, y) = 1.0;";
        let program = Span::new(program);
        let block = F32Grammar::parse_statements(program).unwrap();
        let mut context = Context::new();
        let err = context.evaluate(&block).unwrap_err();
        assert_eq!(err.inner.fragment, "(x, y)");
        assert_matches!(err.inner.extra, EvalError::CannotDestructure);
    }

    #[test]
    fn unexpected_operand() {
        let mut context = Context::new();

        let program = Span::new("1 / (2, 3)");
        let block = F32Grammar::parse_statements(program).unwrap();
        let err = context.evaluate(&block).unwrap_err();
        assert_eq!(err.inner.fragment, "1 / (2, 3)");
        assert_matches!(
            err.inner.extra,
            EvalError::UnexpectedOperand { ref op } if *op == BinaryOp::Div.into()
        );

        let program = Span::new("1 == 1 && !(2, 3)");
        let block = F32Grammar::parse_statements(program).unwrap();
        let err = context.evaluate(&block).unwrap_err();
        assert_eq!(err.inner.fragment, "!(2, 3)");
        assert_matches!(
            err.inner.extra,
            EvalError::UnexpectedOperand { ref op } if *op == UnaryOp::Not.into()
        );

        let program = Span::new("|x| { x + 5 } + 10");
        let block = F32Grammar::parse_statements(program).unwrap();
        let err = context.evaluate(&block).unwrap_err();
        assert_matches!(
            err.inner.extra,
            EvalError::UnexpectedOperand { ref op } if *op == BinaryOp::Add.into()
        );
    }

    #[test]
    fn native_fn_error() {
        let mut context = Context::new();
        context.innermost_scope().insert_native_fn("sin", SIN);

        let program = "1 + sin(-5.0, 2.0)";
        let program = Span::new(program);
        let block = F32Grammar::parse_statements(program).unwrap();
        let err = context.evaluate(&block).unwrap_err();
        assert_eq!(err.inner.fragment, "sin(-5.0, 2.0)");
        assert_matches!(
            err.inner.extra,
            EvalError::NativeCall(ref e)
                if e.to_string().contains("requires one primitive argument")
        );
    }
}
