//! Compilation logic for `Expr`essions.

use core::iter;

use super::{captures::extract_vars_iter, CapturesExtractor, Compiler};
use crate::{
    alloc::{HashMap, String, ToOwned, Vec},
    error::RepeatedAssignmentContext,
    exec::{Atom, Command, CompiledExpr, Executable, ExecutableFn, FieldName, SpannedAtom},
    Error, ErrorKind,
};
use arithmetic_parser::{
    grammars::Grammar, is_valid_variable_name, BinaryOp, Block, Expr, FnDefinition, ObjectExpr,
    Spanned, SpannedExpr, SpannedStatement, Statement,
};

impl Compiler {
    fn compile_expr<'a, T: Grammar<'a>>(
        &mut self,
        executable: &mut Executable<'a, T::Lit>,
        expr: &SpannedExpr<'a, T>,
    ) -> Result<SpannedAtom<'a, T::Lit>, Error<'a>> {
        let atom = match &expr.extra {
            Expr::Literal(lit) => Atom::Constant(lit.clone()),

            Expr::Variable => self.compile_var_access(expr)?,

            Expr::TypeCast { value, .. } => {
                // Just ignore the type annotation.
                self.compile_expr(executable, value)?.extra
            }

            Expr::Tuple(tuple) => {
                let registers = tuple
                    .iter()
                    .map(|elem| {
                        self.compile_expr(executable, elem)
                            .map(|spanned| spanned.extra)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let register =
                    self.push_assignment(executable, CompiledExpr::Tuple(registers), expr);
                Atom::Register(register)
            }

            Expr::Unary { op, inner } => {
                let inner = self.compile_expr(executable, inner)?;
                let register = self.push_assignment(
                    executable,
                    CompiledExpr::Unary {
                        op: self.check_unary_op(op)?,
                        inner,
                    },
                    expr,
                );
                Atom::Register(register)
            }

            Expr::Binary { op, lhs, rhs } => {
                self.compile_binary_expr(executable, expr, op, lhs, rhs)?
            }
            Expr::Function { name, args } => self.compile_fn_call(executable, expr, name, args)?,

            Expr::FieldAccess { name, receiver } => {
                let name = if let Expr::Variable = name.extra {
                    name.with_no_extra()
                } else {
                    let err = ErrorKind::unsupported(expr.extra.ty());
                    return Err(self.create_error(expr, err));
                };
                self.compile_field_access(executable, expr, &name, receiver)?
            }

            Expr::Method {
                name,
                receiver,
                args,
                ..
            } => self.compile_method_call(executable, expr, name, receiver, args)?,

            Expr::Block(block) => self.compile_block(executable, expr, block)?,
            Expr::FnDefinition(def) => self.compile_fn_definition(executable, expr, def)?,
            Expr::Object(object) => self.compile_object(executable, expr, object)?,

            _ => {
                let err = ErrorKind::unsupported(expr.extra.ty());
                return Err(self.create_error(expr, err));
            }
        };

        Ok(expr.copy_with_extra(atom).into())
    }

    fn compile_var_access<'a, T, A>(
        &self,
        var_span: &Spanned<'a, T>,
    ) -> Result<Atom<A>, Error<'a>> {
        let var_name = *var_span.fragment();
        let register = self.vars_to_registers.get(var_name).ok_or_else(|| {
            let err = ErrorKind::Undefined(var_name.to_owned());
            self.create_error(var_span, err)
        })?;
        Ok(Atom::Register(*register))
    }

    fn compile_binary_expr<'a, T: Grammar<'a>>(
        &mut self,
        executable: &mut Executable<'a, T::Lit>,
        binary_expr: &SpannedExpr<'a, T>,
        op: &Spanned<'a, BinaryOp>,
        lhs: &SpannedExpr<'a, T>,
        rhs: &SpannedExpr<'a, T>,
    ) -> Result<Atom<T::Lit>, Error<'a>> {
        let lhs = self.compile_expr(executable, lhs)?;
        let rhs = self.compile_expr(executable, rhs)?;

        let compiled = CompiledExpr::Binary {
            op: self.check_binary_op(op)?,
            lhs,
            rhs,
        };

        let register = self.push_assignment(executable, compiled, binary_expr);
        Ok(Atom::Register(register))
    }

    fn compile_fn_call<'a, T: Grammar<'a>>(
        &mut self,
        executable: &mut Executable<'a, T::Lit>,
        call_expr: &SpannedExpr<'a, T>,
        name: &SpannedExpr<'a, T>,
        args: &[SpannedExpr<'a, T>],
    ) -> Result<Atom<T::Lit>, Error<'a>> {
        let original_name = *name.fragment();
        let original_name = if is_valid_variable_name(original_name) {
            Some(original_name.to_owned())
        } else {
            None
        };

        let name = self.compile_expr(executable, name)?;
        self.compile_fn_call_with_precompiled_name(executable, call_expr, name, original_name, args)
    }

    fn compile_fn_call_with_precompiled_name<'a, T: Grammar<'a>>(
        &mut self,
        executable: &mut Executable<'a, T::Lit>,
        call_expr: &SpannedExpr<'a, T>,
        name: SpannedAtom<'a, T::Lit>,
        original_name: Option<String>,
        args: &[SpannedExpr<'a, T>],
    ) -> Result<Atom<T::Lit>, Error<'a>> {
        let args = args
            .iter()
            .map(|arg| self.compile_expr(executable, arg))
            .collect::<Result<Vec<_>, _>>()?;
        let function = CompiledExpr::FunctionCall {
            name,
            original_name,
            args,
        };
        let register = self.push_assignment(executable, function, call_expr);
        Ok(Atom::Register(register))
    }

    fn compile_field_access<'a, T: Grammar<'a>>(
        &mut self,
        executable: &mut Executable<'a, T::Lit>,
        call_expr: &SpannedExpr<'a, T>,
        name: &Spanned<'a>,
        receiver: &SpannedExpr<'a, T>,
    ) -> Result<Atom<T::Lit>, Error<'a>> {
        let name_str = *name.fragment();
        let field = name_str
            .parse::<usize>()
            .map(FieldName::Index)
            .or_else(|_| {
                if is_valid_variable_name(name_str) {
                    Ok(FieldName::Name(name_str.to_owned()))
                } else {
                    let err = ErrorKind::InvalidFieldName(name_str.to_owned());
                    Err(self.create_error(name, err))
                }
            })?;

        let receiver = self.compile_expr(executable, receiver)?;
        let field_access = CompiledExpr::FieldAccess { receiver, field };
        let register = self.push_assignment(executable, field_access, call_expr);
        Ok(Atom::Register(register))
    }

    fn compile_method_call<'a, T: Grammar<'a>>(
        &mut self,
        executable: &mut Executable<'a, T::Lit>,
        call_expr: &SpannedExpr<'a, T>,
        name: &SpannedExpr<'a, T>,
        receiver: &SpannedExpr<'a, T>,
        args: &[SpannedExpr<'a, T>],
    ) -> Result<Atom<T::Lit>, Error<'a>> {
        let original_name = if matches!(name.extra, Expr::Variable) {
            Some((*name.fragment()).to_owned())
        } else {
            None
        };
        let name = self.compile_expr(executable, name)?;
        let args = iter::once(receiver)
            .chain(args)
            .map(|arg| self.compile_expr(executable, arg))
            .collect::<Result<Vec<_>, _>>()?;

        let function = CompiledExpr::FunctionCall {
            name,
            original_name,
            args,
        };
        let register = self.push_assignment(executable, function, call_expr);
        Ok(Atom::Register(register))
    }

    fn compile_block<'r, 'a: 'r, T: Grammar<'a>>(
        &mut self,
        executable: &mut Executable<'a, T::Lit>,
        block_expr: &SpannedExpr<'a, T>,
        block: &Block<'a, T>,
    ) -> Result<Atom<T::Lit>, Error<'a>> {
        let backup_state = self.backup();
        if self.scope_depth == 0 {
            let command = Command::StartInnerScope;
            executable.push_command(block_expr.copy_with_extra(command));
        }
        self.scope_depth += 1;

        // Move the return value to the next register.
        let return_value = self
            .compile_block_inner(executable, block)?
            .map_or(Atom::Void, |spanned| spanned.extra);

        let new_register = if let Atom::Register(ret_register) = return_value {
            let command = Command::Copy {
                source: ret_register,
                destination: backup_state.register_count,
            };
            executable.push_command(block_expr.copy_with_extra(command));
            true
        } else {
            false
        };

        // Return to the previous state. This will erase register mapping
        // for the inner scope and set the `scope_depth`.
        *self = backup_state;
        if new_register {
            self.register_count += 1;
        }
        if self.scope_depth == 0 {
            let command = Command::EndInnerScope;
            executable.push_command(block_expr.copy_with_extra(command));
        }
        executable.push_command(
            block_expr.copy_with_extra(Command::TruncateRegisters(self.register_count)),
        );

        Ok(if new_register {
            Atom::Register(self.register_count - 1)
        } else {
            Atom::Void
        })
    }

    pub(super) fn compile_block_inner<'a, T: Grammar<'a>>(
        &mut self,
        executable: &mut Executable<'a, T::Lit>,
        block: &Block<'a, T>,
    ) -> Result<Option<SpannedAtom<'a, T::Lit>>, Error<'a>> {
        for statement in &block.statements {
            self.compile_statement(executable, statement)?;
        }

        Ok(if let Some(return_value) = &block.return_value {
            Some(self.compile_expr(executable, return_value)?)
        } else {
            None
        })
    }

    #[allow(clippy::option_if_let_else)] // false positive
    fn compile_object<'a, T: Grammar<'a>>(
        &mut self,
        executable: &mut Executable<'a, T::Lit>,
        object_expr: &SpannedExpr<'a, T>,
        object: &ObjectExpr<'a, T>,
    ) -> Result<Atom<T::Lit>, Error<'a>> {
        let fields = object.fields.iter().map(|(name, field_expr)| {
            let name_str = *name.fragment();
            if let Some(field_expr) = field_expr {
                self.compile_expr(executable, field_expr)
                    .map(|register| (name_str.to_owned(), register.extra))
            } else {
                self.compile_var_access(name)
                    .map(|register| (name_str.to_owned(), register))
            }
        });
        let obj_expr = CompiledExpr::Object(fields.collect::<Result<_, _>>()?);
        let register = self.push_assignment(executable, obj_expr, object_expr);
        Ok(Atom::Register(register))
    }

    fn compile_fn_definition<'a, T: Grammar<'a>>(
        &mut self,
        executable: &mut Executable<'a, T::Lit>,
        def_expr: &SpannedExpr<'a, T>,
        def: &FnDefinition<'a, T>,
    ) -> Result<Atom<T::Lit>, Error<'a>> {
        let module_id = self.module_id.clone_boxed();

        let mut extractor = CapturesExtractor::new(module_id);
        extractor.eval_function(def)?;
        let captures = self.get_captures(extractor);

        let fn_executable = self.compile_function(def, &captures)?;
        let fn_executable = ExecutableFn {
            inner: fn_executable,
            def_span: def_expr.with_no_extra().into(),
            arg_count: def.args.extra.len(),
        };

        let ptr = executable.push_child_fn(fn_executable);
        let (capture_names, captures) = captures
            .into_iter()
            .map(|(name, value)| (name.to_owned(), value))
            .unzip();
        let register = self.push_assignment(
            executable,
            CompiledExpr::DefineFunction {
                ptr,
                captures,
                capture_names,
            },
            def_expr,
        );
        Ok(Atom::Register(register))
    }

    fn get_captures<'a, T>(
        &self,
        extractor: CapturesExtractor<'a>,
    ) -> HashMap<&'a str, SpannedAtom<'a, T>> {
        extractor
            .captures
            .into_iter()
            .map(|(var_name, var_span)| {
                let register = self.get_var(var_name);
                let capture = var_span.copy_with_extra(Atom::Register(register));
                (var_name, capture.into())
            })
            .collect()
    }

    fn compile_function<'a, T: Grammar<'a>>(
        &self,
        def: &FnDefinition<'a, T>,
        captures: &HashMap<&'a str, SpannedAtom<'a, T::Lit>>,
    ) -> Result<Executable<'a, T::Lit>, Error<'a>> {
        // Allocate registers for captures.
        let mut this = Self::new(self.module_id.clone_boxed());
        this.scope_depth = 1; // Disable generating variable annotations.

        for (i, &name) in captures.keys().enumerate() {
            this.vars_to_registers.insert(name.to_owned(), i);
        }
        this.register_count = captures.len() + 1; // one additional register for args

        let mut executable = Executable::new(self.module_id.clone_boxed());
        let args_span = def.args.with_no_extra();
        this.destructure(&mut executable, &def.args.extra, args_span, captures.len())?;

        for statement in &def.body.statements {
            this.compile_statement(&mut executable, statement)?;
        }
        if let Some(return_value) = &def.body.return_value {
            let return_atom = this.compile_expr(&mut executable, return_value)?;
            let return_span = return_atom.with_no_extra();
            let command = Command::Push(CompiledExpr::Atom(return_atom.extra));
            executable.push_command(return_span.copy_with_extra(command));
        }

        executable.finalize_function(this.register_count);
        Ok(executable)
    }

    fn compile_statement<'a, T: Grammar<'a>>(
        &mut self,
        executable: &mut Executable<'a, T::Lit>,
        statement: &SpannedStatement<'a, T>,
    ) -> Result<Option<SpannedAtom<'a, T::Lit>>, Error<'a>> {
        Ok(match &statement.extra {
            Statement::Expr(expr) => Some(self.compile_expr(executable, expr)?),

            Statement::Assignment { lhs, rhs } => {
                extract_vars_iter(
                    self.module_id.as_ref(),
                    &mut HashMap::new(),
                    iter::once(lhs),
                    RepeatedAssignmentContext::Assignment,
                )?;

                let rhs = self.compile_expr(executable, rhs)?;
                // Allocate the register for the constant if necessary.
                let rhs_register = match rhs.extra {
                    Atom::Constant(_) | Atom::Void => {
                        self.push_assignment(executable, CompiledExpr::Atom(rhs.extra), statement)
                    }
                    Atom::Register(register) => register,
                };
                self.assign(executable, lhs, rhs_register)?;
                None
            }

            _ => {
                let err = ErrorKind::unsupported(statement.extra.ty());
                return Err(self.create_error(statement, err));
            }
        })
    }
}
