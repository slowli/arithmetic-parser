//! Captures extractor.

use core::iter;

use arithmetic_parser::{
    grammars::Grammar, Block, Destructure, Expr, FnDefinition, Location, Lvalue, Spanned,
    SpannedExpr, SpannedLvalue, SpannedStatement, Statement,
};

use crate::{
    alloc::{vec, Arc, HashMap, String, ToOwned, Vec},
    error::{AuxErrorInfo, Error, ErrorKind, RepeatedAssignmentContext},
    exec::{ModuleId, WildcardId},
};

#[derive(Debug, Clone)]
pub(crate) struct Captures {
    map: HashMap<String, usize>,
    locations: Vec<Location>,
}

impl Captures {
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Location)> + '_ {
        let iter = self.map.iter();
        iter.map(move |(name, &idx)| (name.as_str(), &self.locations[idx]))
    }

    pub fn len(&self) -> usize {
        self.locations.len()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    #[cfg(test)]
    pub fn location(&self, name: &str) -> Option<&Location> {
        Some(&self.locations[*self.map.get(name)?])
    }

    pub fn variables_map(&self) -> &HashMap<String, usize> {
        &self.map
    }
}

/// Helper context for symbolic execution of a function body or a block in order to determine
/// variables captured by it.
#[derive(Debug)]
pub(super) struct CapturesExtractor<'a> {
    module_id: Arc<dyn ModuleId>,
    local_vars: Vec<HashMap<&'a str, Spanned<'a>>>,
    pub captures: HashMap<&'a str, Spanned<'a>>,
}

impl<'a> CapturesExtractor<'a> {
    pub fn new(module_id: Arc<dyn ModuleId>) -> Self {
        Self {
            module_id,
            local_vars: vec![],
            captures: HashMap::new(),
        }
    }

    /// Collects variables captured by the function into a single `Scope`.
    pub fn eval_function<T: Grammar>(
        &mut self,
        definition: &FnDefinition<'a, T>,
    ) -> Result<(), Error> {
        let mut fn_local_vars = HashMap::new();
        extract_vars(
            &self.module_id,
            &mut fn_local_vars,
            &definition.args.extra,
            RepeatedAssignmentContext::FnArgs,
        )?;
        self.eval_block_inner(&definition.body, fn_local_vars)
    }

    fn has_var(&self, var_name: &str) -> bool {
        self.local_vars.iter().any(|set| set.contains_key(var_name))
    }

    /// Processes a local variable in the rvalue position.
    fn eval_local_var<T>(&mut self, var_span: &Spanned<'a, T>) {
        let var_name = *var_span.fragment();
        if !self.has_var(var_name) && !self.captures.contains_key(var_name) {
            self.captures.insert(var_name, var_span.with_no_extra());
        }
    }

    fn create_error<T>(&self, span: &Spanned<'a, T>, err: ErrorKind) -> Error {
        Error::new(self.module_id.clone(), span, err)
    }

    /// Evaluates an expression with the function validation semantics, i.e., to determine
    /// function captures.
    fn eval<T: Grammar>(&mut self, expr: &SpannedExpr<'a, T>) -> Result<(), Error> {
        match &expr.extra {
            Expr::Variable => {
                self.eval_local_var(expr);
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
            Expr::Binary { lhs, rhs, .. } => {
                self.eval(lhs)?;
                self.eval(rhs)?;
            }

            Expr::Function { args, name } => {
                for arg in args {
                    self.eval(arg)?;
                }
                self.eval(name)?;
            }

            Expr::FieldAccess { receiver, .. } => {
                self.eval(receiver)?;
            }

            Expr::Method {
                args,
                receiver,
                name,
                ..
            } => {
                self.eval(name)?;
                self.eval(receiver)?;
                for arg in args {
                    self.eval(arg)?;
                }
            }

            Expr::Block(block) => {
                self.eval_block_inner(block, HashMap::new())?;
            }
            Expr::Object(object) => {
                // Check that all field names are unique.
                let mut object_fields = HashMap::new();
                for (name, _) in &object.fields {
                    let field_str = *name.fragment();
                    if let Some(prev_span) = object_fields.insert(field_str, *name) {
                        let err = ErrorKind::RepeatedField;
                        return Err(Error::new(self.module_id.clone(), name, err)
                            .with_location(&prev_span.into(), AuxErrorInfo::PrevAssignment));
                    }
                }

                for (name, field_expr) in &object.fields {
                    if let Some(field_expr) = field_expr {
                        self.eval(field_expr)?;
                    } else {
                        self.eval_local_var(name);
                    }
                }
            }
            Expr::TypeCast { value, .. } => {
                self.eval(value)?;
            }

            Expr::FnDefinition(def) => {
                self.eval_function(def)?;
            }

            _ => {
                let err = ErrorKind::unsupported(expr.extra.ty());
                return Err(self.create_error(expr, err));
            }
        }

        Ok(())
    }

    /// Evaluates a statement using the provided context.
    fn eval_statement<T: Grammar>(
        &mut self,
        statement: &SpannedStatement<'a, T>,
    ) -> Result<(), Error> {
        match &statement.extra {
            Statement::Expr(expr) => self.eval(expr),

            Statement::Assignment { lhs, rhs } => {
                self.eval(rhs)?;
                let mut new_vars = HashMap::new();
                extract_vars_iter(
                    &self.module_id,
                    &mut new_vars,
                    iter::once(lhs),
                    RepeatedAssignmentContext::Assignment,
                )?;
                self.local_vars.last_mut().unwrap().extend(&new_vars);
                Ok(())
            }

            _ => {
                let err = ErrorKind::unsupported(statement.extra.ty());
                Err(self.create_error(statement, err))
            }
        }
    }

    fn eval_block_inner<T: Grammar>(
        &mut self,
        block: &Block<'a, T>,
        local_vars: HashMap<&'a str, Spanned<'a>>,
    ) -> Result<(), Error> {
        self.local_vars.push(local_vars);
        for statement in &block.statements {
            self.eval_statement(statement)?;
        }
        if let Some(ref return_expr) = block.return_value {
            self.eval(return_expr)?;
        }
        self.local_vars.pop();
        Ok(())
    }

    pub fn eval_block<T: Grammar>(&mut self, block: &Block<'a, T>) -> Result<(), Error> {
        self.eval_block_inner(block, HashMap::new())
    }

    pub fn into_captures(self) -> Captures {
        let (map, locations) = self
            .captures
            .into_iter()
            .enumerate()
            .map(|(i, (name, span))| ((name.to_owned(), i), Location::from(span)))
            .unzip();
        Captures { map, locations }
    }
}

fn extract_vars<'a, T>(
    module_id: &Arc<dyn ModuleId>,
    vars: &mut HashMap<&'a str, Spanned<'a>>,
    lvalues: &Destructure<'a, T>,
    context: RepeatedAssignmentContext,
) -> Result<(), Error> {
    let middle = lvalues
        .middle
        .as_ref()
        .and_then(|rest| rest.extra.to_lvalue());
    let all_lvalues = lvalues
        .start
        .iter()
        .chain(middle.as_ref())
        .chain(&lvalues.end);
    extract_vars_iter(module_id, vars, all_lvalues, context)
}

fn add_var<'a>(
    module_id: &Arc<dyn ModuleId>,
    vars: &mut HashMap<&'a str, Spanned<'a>>,
    var_span: Spanned<'a>,
    context: RepeatedAssignmentContext,
) -> Result<(), Error> {
    let var_name = *var_span.fragment();
    if var_name != "_" {
        if let Some(prev_span) = vars.insert(var_name, var_span) {
            let err = ErrorKind::RepeatedAssignment { context };
            return Err(Error::new(module_id.clone(), &var_span, err)
                .with_location(&prev_span.into(), AuxErrorInfo::PrevAssignment));
        }
    }
    Ok(())
}

pub(super) fn extract_vars_iter<'it, 'a: 'it, T: 'it>(
    module_id: &Arc<dyn ModuleId>,
    vars: &mut HashMap<&'a str, Spanned<'a>>,
    lvalues: impl Iterator<Item = &'it SpannedLvalue<'a, T>>,
    context: RepeatedAssignmentContext,
) -> Result<(), Error> {
    for lvalue in lvalues {
        match &lvalue.extra {
            Lvalue::Variable { .. } => {
                add_var(module_id, vars, lvalue.with_no_extra(), context)?;
            }

            Lvalue::Tuple(tuple) => {
                extract_vars(module_id, vars, tuple, context)?;
            }

            Lvalue::Object(object) => {
                let mut object_fields = HashMap::new();
                for field in &object.fields {
                    let field_str = *field.field_name.fragment();
                    if let Some(prev_span) = object_fields.insert(field_str, field.field_name) {
                        let err = ErrorKind::RepeatedField;
                        return Err(Error::new(module_id.clone(), &field.field_name, err)
                            .with_location(&prev_span.into(), AuxErrorInfo::PrevAssignment));
                    }

                    if let Some(binding) = &field.binding {
                        extract_vars_iter(module_id, vars, iter::once(binding), context)?;
                    } else {
                        add_var(module_id, vars, field.field_name, context)?;
                    }
                }
            }

            _ => {
                let err = ErrorKind::unsupported(lvalue.extra.ty());
                return Err(Error::new(module_id.clone(), lvalue, err));
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
    pub fn get_undefined_variables(self) -> Result<HashMap<&'a str, Spanned<'a>>, Error> {
        let mut extractor = CapturesExtractor::new(Arc::new(WildcardId));
        match self {
            Self::Block(block) => extractor.eval_block_inner(block, HashMap::new())?,
            Self::FnDefinition(definition) => extractor.eval_function(definition)?,
        }
        Ok(extractor.captures)
    }
}
