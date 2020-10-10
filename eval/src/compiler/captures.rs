//! Captures extractor.

use hashbrown::HashMap;

use core::iter;

use crate::{
    alloc::{vec, Vec},
    compiler::CMP_FUNCTION_NAME,
    AuxErrorInfo, EvalError, RepeatedAssignmentContext, SpannedEvalError,
};
use arithmetic_parser::{
    BinaryOp, Block, Destructure, Expr, FnDefinition, Grammar, Lvalue, Spanned, SpannedExpr,
    SpannedLvalue, SpannedStatement, Statement,
};

/// Helper context for symbolic execution of a function body or a block in order to determine
/// variables captured by it.
#[derive(Debug)]
pub(super) struct CapturesExtractor<'a, F> {
    local_vars: Vec<HashMap<&'a str, Spanned<'a>>>,
    action: F,
}

impl<'a, F> CapturesExtractor<'a, F>
where
    F: FnMut(&'a str, Spanned<'a>) -> Result<(), EvalError>,
{
    pub fn new(action: F) -> Self {
        Self {
            local_vars: vec![],
            action,
        }
    }

    /// Collects variables captured by the function into a single `Scope`.
    pub fn eval_function<T: Grammar>(
        &mut self,
        definition: &FnDefinition<'a, T>,
    ) -> Result<(), SpannedEvalError<'a>> {
        self.local_vars.push(HashMap::new());
        extract_vars(
            self.local_vars.last_mut().unwrap(),
            &definition.args.extra,
            RepeatedAssignmentContext::FnArgs,
        )?;
        self.eval_block_inner(&definition.body)
    }

    fn has_var(&self, var_name: &str) -> bool {
        self.local_vars.iter().any(|set| set.contains_key(var_name))
    }

    /// Processes a local variable in the rvalue position.
    fn eval_local_var<T>(&mut self, var_span: &Spanned<'a, T>) -> Result<(), EvalError> {
        if self.has_var(var_span.fragment()) {
            // No action needs to be performed.
            Ok(())
        } else {
            (self.action)(var_span.fragment(), var_span.with_no_extra())
        }
    }

    fn eval_cmp(&mut self, op_span: &Spanned<'a, BinaryOp>) -> Result<(), EvalError> {
        if self.has_var(CMP_FUNCTION_NAME) {
            Ok(())
        } else {
            (self.action)(CMP_FUNCTION_NAME, op_span.with_no_extra())
        }
    }

    /// Evaluates an expression with the function validation semantics, i.e., to determine
    /// function captures.
    fn eval<T: Grammar>(&mut self, expr: &SpannedExpr<'a, T>) -> Result<(), SpannedEvalError<'a>> {
        match &expr.extra {
            Expr::Variable => {
                self.eval_local_var(expr)
                    .map_err(|e| SpannedEvalError::new(expr, e))?;
            }

            Expr::Literal(_) => { /* no action */ }

            Expr::Tuple(fragments) => {
                for fragment in fragments {
                    self.eval(fragment)?;
                }
            }
            Expr::Unary { inner, .. } => {
                self.eval(inner)?;
            }
            Expr::Binary { lhs, rhs, op } => {
                self.eval(lhs)?;
                self.eval(rhs)?;

                if op.extra.is_order_comparison() {
                    self.eval_cmp(op)
                        .map_err(|e| SpannedEvalError::new(op, e))?;
                }
            }

            Expr::Function { args, name } => {
                for arg in args {
                    self.eval(arg)?;
                }
                self.eval(name)?;
            }

            Expr::Method {
                args,
                receiver,
                name,
            } => {
                self.eval(receiver)?;
                for arg in args {
                    self.eval(arg)?;
                }

                self.eval_local_var(name)
                    .map_err(|e| SpannedEvalError::new(name, e))?;
            }

            Expr::Block(block) => {
                self.local_vars.push(HashMap::new());
                self.eval_block_inner(block)?;
            }

            Expr::FnDefinition(def) => {
                self.eval_function(def)?;
            }
        }
        Ok(())
    }

    /// Evaluates a statement using the provided context.
    fn eval_statement<T: Grammar>(
        &mut self,
        statement: &SpannedStatement<'a, T>,
    ) -> Result<(), SpannedEvalError<'a>> {
        match &statement.extra {
            Statement::Expr(expr) => self.eval(expr),
            Statement::Assignment { lhs, rhs } => {
                self.eval(rhs)?;
                let mut new_vars = HashMap::new();
                extract_vars_iter(
                    &mut new_vars,
                    iter::once(lhs),
                    RepeatedAssignmentContext::Assignment,
                )?;
                self.local_vars.last_mut().unwrap().extend(&new_vars);
                Ok(())
            }
        }
    }

    fn eval_block_inner<T: Grammar>(
        &mut self,
        block: &Block<'a, T>,
    ) -> Result<(), SpannedEvalError<'a>> {
        self.local_vars.push(HashMap::new());
        for statement in &block.statements {
            self.eval_statement(statement)?;
        }
        if let Some(ref return_expr) = block.return_value {
            self.eval(return_expr)?;
        }
        self.local_vars.pop();
        Ok(())
    }

    pub fn eval_block<T: Grammar>(
        &mut self,
        block: &Block<'a, T>,
    ) -> Result<(), SpannedEvalError<'a>> {
        self.local_vars.push(HashMap::new());
        self.eval_block_inner(block)
    }
}

fn extract_vars<'a, T>(
    vars: &mut HashMap<&'a str, Spanned<'a>>,
    lvalues: &Destructure<'a, T>,
    context: RepeatedAssignmentContext,
) -> Result<(), SpannedEvalError<'a>> {
    let middle = lvalues
        .middle
        .as_ref()
        .and_then(|rest| rest.extra.to_lvalue());
    let all_lvalues = lvalues
        .start
        .iter()
        .chain(middle.as_ref())
        .chain(&lvalues.end);
    extract_vars_iter(vars, all_lvalues, context)
}

pub(super) fn extract_vars_iter<'it, 'a: 'it, T: 'it>(
    vars: &mut HashMap<&'a str, Spanned<'a>>,
    lvalues: impl Iterator<Item = &'it SpannedLvalue<'a, T>>,
    context: RepeatedAssignmentContext,
) -> Result<(), SpannedEvalError<'a>> {
    for lvalue in lvalues {
        match &lvalue.extra {
            Lvalue::Variable { .. } => {
                let var_name = *lvalue.fragment();
                if var_name != "_" {
                    let var_span = lvalue.with_no_extra();
                    if let Some(prev_span) = vars.insert(var_name, var_span) {
                        let err = EvalError::RepeatedAssignment { context };
                        return Err(SpannedEvalError::new(lvalue, err)
                            .with_span(&prev_span, AuxErrorInfo::PrevAssignment));
                    }
                }
            }

            Lvalue::Tuple(fragments) => {
                extract_vars(vars, fragments, context)?;
            }
        }
    }
    Ok(())
}

/// Helper enum for `CompilerExt` implementations that allows to reduce code duplication.
#[derive(Debug)]
pub(super) enum CompilerExtTarget<'r, 'a, T: Grammar> {
    Block(&'r Block<'a, T>),
    FnDefinition(&'r FnDefinition<'a, T>),
}

impl<'a, T: Grammar> CompilerExtTarget<'_, 'a, T> {
    pub fn get_undefined_variables(
        self,
    ) -> Result<HashMap<&'a str, Spanned<'a>>, SpannedEvalError<'a>> {
        let mut undefined_vars = HashMap::new();
        let mut extractor = CapturesExtractor::new(|var_name, var_span| {
            if !undefined_vars.contains_key(var_name) {
                undefined_vars.insert(var_name, var_span);
            }
            Ok(())
        });

        match self {
            Self::Block(block) => extractor.eval_block_inner(block)?,
            Self::FnDefinition(definition) => extractor.eval_function(definition)?,
        }

        Ok(undefined_vars)
    }
}
