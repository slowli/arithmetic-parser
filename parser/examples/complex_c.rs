//! Demonstrates how arithmetic expressions can be rewritten to a C-like language.
//! (More precisely, the generated expressions are compatible with OpenCL C.)
//!
//! The example uses complex-valued literals and features a primitive form of constant folding.

use std::{
    collections::{HashMap, HashSet},
    fmt::{self, Write as _},
    iter::FromIterator,
    ops,
};

use arithmetic_parser::{
    grammars::{NumGrammar, Parse, Untyped},
    BinaryOp, Block, Expr, FnDefinition, InputSpan, Lvalue, OpPriority, SpannedExpr, SpannedLvalue,
    Statement, UnaryOp,
};
use num_complex::Complex32;

type ComplexGrammar = Untyped<NumGrammar<Complex32>>;

/// Evaluated expression.
#[derive(Debug, Clone)]
enum Evaluated {
    /// Expression is symbolic, i.e., cannot be evaluated eagerly (at least given our
    /// primitive constant folding algorithm).
    Symbolic {
        /// Expression contents serialized as a string.
        content: String,
        /// Priority of the latest executed operation.
        priority: OpPriority,
    },
    /// Expression has a value known in compile time.
    Value(Complex32),
}

impl Evaluated {
    fn var(name: impl Into<String>) -> Self {
        Self::Symbolic {
            content: name.into(),
            priority: OpPriority::max_priority(),
        }
    }

    /// Converts this expression into a string.
    fn into_string(self, caller_priority: OpPriority) -> String {
        match self {
            Evaluated::Symbolic { content, priority } => {
                if priority < caller_priority {
                    format!("({content})")
                } else {
                    content
                }
            }
            Evaluated::Value(val) => format!("(float2)({}, {})", val.re, val.im),
        }
    }
}

impl fmt::Display for Evaluated {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Value(val) => write!(formatter, "(float2)({}, {})", val.re, val.im),
            Self::Symbolic { content, .. } => formatter.write_str(content),
        }
    }
}

impl ops::Add for Evaluated {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        use self::Evaluated::*;

        match (self, rhs) {
            (Value(lhs), Value(rhs)) => Value(lhs + rhs),
            (lhs, rhs) => {
                let priority = BinaryOp::Add.priority();
                let content = format!(
                    "{} + {}",
                    lhs.into_string(priority),
                    rhs.into_string(priority)
                );
                Symbolic { content, priority }
            }
        }
    }
}

impl ops::Sub for Evaluated {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        use self::Evaluated::*;

        match (self, rhs) {
            (Value(lhs), Value(rhs)) => Value(lhs - rhs),
            (lhs, rhs) => {
                let priority = BinaryOp::Sub.priority();
                let content = format!(
                    "{} - {}",
                    lhs.into_string(priority),
                    rhs.into_string(priority)
                );
                Symbolic { content, priority }
            }
        }
    }
}

impl ops::Mul for Evaluated {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        use self::Evaluated::*;

        match (self, rhs) {
            (Value(lhs), Value(rhs)) => Value(lhs * rhs),
            (lhs, rhs) => Symbolic {
                content: format!("complex_mul({lhs}, {rhs})"),
                priority: OpPriority::max_priority(),
            },
        }
    }
}

impl ops::Div for Evaluated {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        use self::Evaluated::*;

        match (self, rhs) {
            (Value(lhs), Value(rhs)) => Value(lhs / rhs),
            (lhs, rhs) => Symbolic {
                content: format!("complex_div({lhs}, {rhs})"),
                priority: OpPriority::max_priority(),
            },
        }
    }
}

impl ops::BitXor for Evaluated {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        use self::Evaluated::*;

        match (self, rhs) {
            (Value(lhs), Value(rhs)) => Value(lhs.powc(rhs)),
            (lhs, rhs) => Symbolic {
                content: format!("complex_pow({lhs}, {rhs})"),
                priority: OpPriority::max_priority(),
            },
        }
    }
}

impl ops::Neg for Evaluated {
    type Output = Self;

    fn neg(self) -> Self::Output {
        use self::Evaluated::*;

        match self {
            Value(value) => Value(-value),
            value => {
                let priority = OpPriority::max_priority();
                let content = format!("-{}", value.into_string(priority));
                Symbolic { content, priority }
            }
        }
    }
}

#[derive(Debug)]
struct Context<'a> {
    variables: HashMap<&'a str, Evaluated>,
    functions: HashSet<&'a str>,
}

impl<'a> Context<'a> {
    fn new() -> Self {
        Self {
            variables: HashMap::new(),
            functions: HashSet::from_iter(vec!["sinh", "cosh", "tanh"]),
        }
    }

    fn generate_code(block: &Block<'a, ComplexGrammar>) -> String {
        let mut code = String::new();

        for statement in &block.statements {
            match &statement.extra {
                Statement::Assignment { lhs, rhs } => match &rhs.extra {
                    Expr::FnDefinition(fn_def) => {
                        if !code.is_empty() {
                            code += "\n\n";
                        }
                        code += &Self::eval_function(fn_def, lhs.fragment());
                    }
                    _ => panic!("Top-level statements should be function definitions"),
                },
                _ => panic!("Top-level statements should be function definitions"),
            }
        }

        code
    }

    fn eval_function(fn_def: &FnDefinition<'a, ComplexGrammar>, name: &str) -> String {
        let mut context = Self::new();
        let mut evaluated = format!("float2 {name}(");
        let args = &fn_def.args.extra.start;

        for (i, arg) in args.iter().enumerate() {
            let was_present = context
                .variables
                .insert(arg.fragment(), Evaluated::var(*arg.fragment()));
            if was_present.is_some() {
                panic!("Cannot redefine function argument `{}`", arg.fragment());
            }

            evaluated += "float2 ";
            evaluated += arg.fragment();
            if i + 1 < args.len() {
                evaluated += ", ";
            }
        }
        evaluated += ") {\n";

        for statement in &fn_def.body.statements {
            match &statement.extra {
                Statement::Expr(_) => panic!("Useless expression: {}", statement.fragment()),
                Statement::Assignment { lhs, rhs } => {
                    if let Some(line) = context.eval_assignment(lhs, rhs) {
                        evaluated += "    ";
                        evaluated += &line;
                        evaluated += "\n";
                    }
                }
                _ => panic!("Unsupported statement type: {statement:?}"),
            }
        }

        let return_value = fn_def
            .body
            .return_value
            .as_ref()
            .expect("Function does not have return value");
        writeln!(evaluated, "    return {};", context.eval_expr(return_value)).unwrap();
        // ^ `unwrap()` is safe: writing to a `String` cannot fail
        evaluated += "}";
        evaluated
    }

    fn eval_assignment(
        &mut self,
        lhs: &SpannedLvalue<'a, ()>,
        rhs: &SpannedExpr<'a, ComplexGrammar>,
    ) -> Option<String> {
        let variable_name = match lhs.extra {
            Lvalue::Variable { .. } => *lhs.fragment(),
            Lvalue::Tuple(_) => unreachable!("Tuples are disabled in parser"),
            _ => panic!("Unsupported lvalue type: {lhs:?}"),
        };

        if self.variables.contains_key(variable_name) {
            panic!("Cannot redefine variable `{variable_name}`");
        }

        // Evaluate the RHS.
        let value = self.eval_expr(rhs);
        let return_value = if let Evaluated::Symbolic { .. } = value {
            Some(format!("float2 {variable_name} = {value};"))
        } else {
            None
        };
        self.variables.insert(variable_name, value);
        return_value
    }

    fn eval_expr(&self, expr: &SpannedExpr<'a, ComplexGrammar>) -> Evaluated {
        match &expr.extra {
            Expr::Variable => {
                let var = self
                    .variables
                    .get(expr.fragment())
                    .unwrap_or_else(|| panic!("Variable {} is not defined", expr.fragment()));

                match var {
                    Evaluated::Symbolic { .. } => Evaluated::var(*expr.fragment()),
                    value => value.to_owned(),
                }
            }
            Expr::Literal(lit) => Evaluated::Value(*lit),

            Expr::Unary { op, inner } => match op.extra {
                UnaryOp::Neg => -self.eval_expr(inner),
                UnaryOp::Not => panic!("Boolean operations are not supported: {}", expr.fragment()),
                _ => panic!("Unsupported unary op: {:?}", op.extra),
            },

            Expr::Binary { lhs, op, rhs } => {
                let lhs_value = self.eval_expr(lhs);
                let rhs_value = self.eval_expr(rhs);

                match op.extra {
                    BinaryOp::Add => lhs_value + rhs_value,
                    BinaryOp::Sub => lhs_value - rhs_value,
                    BinaryOp::Mul => lhs_value * rhs_value,
                    BinaryOp::Div => lhs_value / rhs_value,
                    BinaryOp::Power => lhs_value ^ rhs_value,
                    _ => panic!("Boolean operations are not supported: {}", expr.fragment()),
                }
            }

            Expr::Function { name, args } => {
                let fn_name = name.fragment();
                if !self.functions.contains(fn_name) {
                    panic!("Undefined function `{fn_name}`");
                }

                let mut fn_call = format!("complex_{fn_name}(");
                let arg_values = args.iter().map(|arg| self.eval_expr(arg));
                for (i, arg) in arg_values.enumerate() {
                    fn_call += &arg.to_string();
                    if i + 1 < args.len() {
                        fn_call += ", ";
                    }
                }
                fn_call += ")";

                Evaluated::Symbolic {
                    content: fn_call,
                    priority: OpPriority::max_priority(),
                }
            }

            Expr::Method { .. } => panic!("Methods are not supported"),
            Expr::FnDefinition(_) => panic!("Embedded function declarations are not supported"),
            Expr::Block(_) | Expr::Tuple(_) => unreachable!("Disabled in parser"),
            _ => panic!("Unsupported expression: {:?}", expr.extra),
        }
    }
}

// TODO: allow to read input from command line.
fn main() {
    // The input should be one or more function definitions.
    const EXPR: &str = "computation = |z| {
        // Define a constant.
        c = (-0.5 + 0.2i) * 3; // Note that this constant will be folded.
        d = (-0.3i * 2 + 2) * z^2 + c^2;

        // Returned expression.
        sinh(z^2 + c) * d
    };

    other_computation = |a, b| sinh(a^2 + b^2) * tanh(a / b);";

    let span = InputSpan::new(EXPR);
    let statements = ComplexGrammar::parse_statements(span).unwrap();

    let code = Context::generate_code(&statements);
    println!("Generated code:\n\n{code}");
}
