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
//!     interpreter::{fns, Interpreter, Value},
//!     Grammar, GrammarExt, Span,
//! };
//!
//! const MIN: fns::Binary<fn(f32, f32) -> f32> =
//!     fns::Binary::new(|x, y| if x < y { x } else { y });
//! const MAX: fns::Binary<fn(f32, f32) -> f32> =
//!     fns::Binary::new(|x, y| if x > y { x } else { y });
//!
//! let mut context = Interpreter::new();
//! // Add some native functions to the interpreter.
//! context
//!     .insert_native_fn("min", MIN)
//!     .insert_native_fn("max", MAX)
//!     .insert_native_fn("assert", fns::Assert);
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
    executable::ExecutableModule,
    values::{CallContext, Function, InterpretedFn, NativeFn, SpannedValue, Value},
};

use num_traits::{Num, Pow};

use std::ops;

use self::{compiler::Compiler, executable::Env};
use crate::{Block, Grammar};

mod compiler;
mod error;
mod executable;
pub mod fns;
mod values;

/// Simple interpreter for arithmetic expressions.
#[derive(Debug)]
pub struct Interpreter<'a, T: Grammar> {
    env: Env<'a, T>,
}

impl<T: Grammar> Default for Interpreter<'_, T>
where
    T::Lit: Num + ops::Neg<Output = T::Lit> + Pow<T::Lit, Output = T::Lit>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T> Interpreter<'a, T>
where
    T: Grammar,
{
    /// Creates a new interpreter.
    pub fn new() -> Self {
        Self { env: Env::new() }
    }

    /// Gets a variable by name.
    pub fn get_var(&self, name: &'a str) -> Option<&Value<'a, T>> {
        self.env.get_var(name)
    }

    /// Iterates over variables.
    pub fn variables(&self) -> impl Iterator<Item = (&'a str, &Value<'a, T>)> + '_ {
        self.env.variables()
    }

    /// Inserts a variable with the specified name.
    pub fn insert_var(&mut self, name: &'a str, value: Value<'a, T>) -> &mut Self {
        self.env.push_var(name, value);
        self
    }

    /// Inserts a native function with the specified name.
    pub fn insert_native_fn(
        &mut self,
        name: &'a str,
        native_fn: impl NativeFn<T> + 'static,
    ) -> &mut Self {
        self.insert_var(name, Value::native_fn(native_fn))
    }
}

impl<'a, T> Interpreter<'a, T>
where
    T: Grammar,
    T::Lit: Num + ops::Neg<Output = T::Lit> + Pow<T::Lit, Output = T::Lit>,
{
    /// Evaluates a list of statements.
    pub fn evaluate(
        &mut self,
        block: &Block<'a, T>,
    ) -> Result<Value<'a, T>, ErrorWithBacktrace<'a>> {
        let executable = Compiler::compile_module(&self.env, block, true)
            .map_err(ErrorWithBacktrace::with_empty_trace)?;
        let mut backtrace = Backtrace::default();
        let result = self
            .env
            .execute(executable.inner(), Some(&mut backtrace))
            .map_err(|err| ErrorWithBacktrace::new(err, backtrace));
        self.env.compress();
        result
    }

    /// Compiles the provided block, returning the compiled module and its imports.
    /// The imports can the be changed to run the module with different params.
    pub fn compile(
        &self,
        program: &Block<'a, T>,
    ) -> Result<ExecutableModule<'a, T>, SpannedEvalError<'a>> {
        Compiler::compile_module(&self.env, program, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{grammars::F32Grammar, BinaryOp, GrammarExt, LvalueLen, Span, UnaryOp};

    use assert_matches::assert_matches;
    use std::{collections::HashMap, iter::FromIterator, rc::Rc};

    const SIN: fns::Unary<fn(f32) -> f32> = fns::Unary::new(f32::sin);

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
        let return_value = interpreter.evaluate(&block).unwrap();
        assert_eq!(
            return_value,
            Value::Tuple(vec![
                Value::Number(0.5),
                Value::Tuple(vec![Value::Number(0.25), Value::Number(5.0)])
            ])
        );

        let program = "(1, 2) / |x| { x + 1 }";
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let err = interpreter.evaluate(&block).unwrap_err();
        assert_matches!(
            err.source(),
            EvalError::UnexpectedOperand { ref op } if *op == BinaryOp::Div.into()
        );
    }

    #[test]
    fn comparisons() {
        let mut interpreter = Interpreter::new();
        interpreter
            .insert_var("true", Value::Bool(true))
            .insert_var("false", Value::Bool(false))
            .insert_native_fn("sin", SIN);

        let program = r#"
            foo = |x| { x + 1 };
            alias = foo;

            1 == foo(0) && 1 != (1,) && (2, 3) != (2,) && (2, 3) != (2, 3, 4) && () == ()
                && true == true && true != false && (true, false) == (true, false)
                && foo == foo && foo == alias && foo != 1 && sin == sin && foo != sin
                && (foo, (-1, 3)) == (alias, (-1, 3))
        "#;
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let return_value = interpreter.evaluate(&block).unwrap();
        assert_eq!(return_value, Value::Bool(true));
    }

    #[test]
    fn tuple_destructuring_with_middle() {
        let program = r#"
            (x, ...) = (1, 2, 3);
            (..., u, v) = (-5, -4, -3, -2, -1);
            (_, _, ...middle, y) = (1, 2, 3, 4, 5);
            ((_, a), ..., (b, ...)) = ((1, 2), (3, 4, 5));
            x == 1 && u == -2 && v == -1 && middle == (3, 4) && y == 5 && a == 2 && b == 3
        "#;
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let mut interpreter = Interpreter::new();
        let return_value = interpreter.evaluate(&block).unwrap();
        assert_eq!(return_value, Value::Bool(true));
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
        let captures = add.captures();
        assert_eq!(captures.len(), 1);
        assert_matches!(captures["op"], Value::Function(_));

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
    fn variadic_function() {
        let program = r#"
            call = |fn, ...xs| fn(xs);
            (call(|x| -x, 1, 2, -3.5), call(|x| 1/x, 4, -0.125))
        "#;
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let mut interpreter = Interpreter::new();
        let return_value = interpreter.evaluate(&block).unwrap();

        let first = Value::Tuple(vec![
            Value::Number(-1.0),
            Value::Number(-2.0),
            Value::Number(3.5),
        ]);
        let second = Value::Tuple(vec![Value::Number(0.25), Value::Number(-8.0)]);
        assert_eq!(return_value, Value::Tuple(vec![first, second]));

        let program = r#"
            call = |fn, ...xs, y| fn(xs, y);
            call(|x, y| x + y, 1, 2, -3.5) == (-2.5, -1.5)
        "#;
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let mut interpreter = Interpreter::new();
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
            assert!(function.captures()["div"].is_function());
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
        let captures = function.captures();
        assert_eq!(
            captures,
            HashMap::from_iter(vec![("x", &Value::Number(-3.0))])
        );
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
        interpreter.insert_native_fn("sin", SIN);

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
        interpreter.insert_native_fn("sin", SIN);
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
    fn method_call() {
        let program = "add = |x, y| x + y; 1.0.add(2)";
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let return_value = Interpreter::new().evaluate(&block).unwrap();
        assert_eq!(return_value, Value::Number(3.0));

        let program = r#"
            gen = |x| { |y| x + y };
            (1, -2).gen()(3) == (4, 1)
        "#;
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let return_value = Interpreter::new().evaluate(&block).unwrap();
        assert_eq!(return_value, Value::Bool(true));
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
        assert_matches!(
            err.source(),
            EvalError::ArgsLenMismatch {
                def: LvalueLen::Exact(1),
                call: 0,
            }
        );

        let program = Span::new("foo(1, 2) * 3.0");
        let block = F32Grammar::parse_statements(program).unwrap();
        let err = interpreter.evaluate(&block).unwrap_err();
        assert_eq!(err.main_span().fragment, "foo(1, 2)");
        assert_matches!(
            err.source(),
            EvalError::ArgsLenMismatch {
                def: LvalueLen::Exact(1),
                call: 2,
            }
        );
    }

    #[test]
    fn arg_len_mismatch_with_variadic_function() {
        let mut interpreter = Interpreter::new();
        let program = Span::new("foo = |fn, ...xs| fn(xs); foo()");
        let block = F32Grammar::parse_statements(program).unwrap();
        let err = interpreter.evaluate(&block).unwrap_err();
        assert_eq!(err.main_span().fragment, "foo()");
        assert_matches!(
            err.source(),
            EvalError::ArgsLenMismatch {
                def: LvalueLen::AtLeast(1),
                call: 0,
            }
        );
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

        let program = Span::new("(x, ...x) = (1, 2);");
        let block = F32Grammar::parse_statements(program).unwrap();
        let err = Interpreter::new().evaluate(&block).unwrap_err();
        assert_eq!(err.main_span().fragment, "x");
        assert_eq!(err.main_span().offset, 7);
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
        assert_eq!(err.main_span().fragment, "(1, 2)");
        assert_matches!(
            err.source(),
            EvalError::TupleLenMismatch { lhs: LvalueLen::Exact(2), rhs: 3, .. }
        );
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
        interpreter.insert_native_fn("sin", SIN);

        let program = Span::new("1 + sin(-5.0, 2.0)");
        let block = F32Grammar::parse_statements(program).unwrap();
        let err = interpreter.evaluate(&block).unwrap_err();
        assert_eq!(err.main_span().fragment, "sin(-5.0, 2.0)");
        assert_matches!(
            err.source(),
            EvalError::ArgsLenMismatch {
                def: LvalueLen::Exact(1),
                call: 2
            }
        );

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
