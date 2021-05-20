//! Assertion functions.

use core::fmt;

use crate::{
    alloc::{format, Vec},
    error::AuxErrorInfo,
    CallContext, ErrorKind, EvalResult, NativeFn, SpannedValue, Value,
};
use arithmetic_parser::CodeFragment;

/// Assertion function.
///
/// # Type
///
/// ```text
/// fn(bool)
/// ```
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, ErrorKind, VariableMap};
/// # use assert_matches::assert_matches;
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     assert(1 + 2 != 5); // this assertion is fine
///     assert(3^2 > 10); // this one will fail
/// "#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
/// let module = Environment::new()
///     .insert_native_fn("assert", fns::Assert)
///     .compile_module("test_assert", &program)?;
///
/// let err = module.run().unwrap_err();
/// assert_eq!(*err.source().main_span().code().fragment(), "assert(3^2 > 10)");
/// assert_matches!(
///     err.source().kind(),
///     ErrorKind::NativeCall(ref msg) if msg == "Assertion failed: 3^2 > 10"
/// );
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Assert;

impl<T> NativeFn<T> for Assert {
    fn evaluate<'a>(
        &self,
        args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a, T>,
    ) -> EvalResult<'a, T> {
        ctx.check_args_count(&args, 1)?;
        match args[0].extra {
            Value::Bool(true) => Ok(Value::void()),

            Value::Bool(false) => {
                let err = if let CodeFragment::Str(code) = args[0].fragment() {
                    ErrorKind::native(format!("Assertion failed: {}", code))
                } else {
                    ErrorKind::native("Assertion failed")
                };
                Err(ctx.call_site_error(err))
            }

            _ => {
                let err = ErrorKind::native("`assert` requires a single boolean argument");
                Err(ctx
                    .call_site_error(err)
                    .with_span(&args[0], AuxErrorInfo::InvalidArg))
            }
        }
    }
}

/// Equality assertion function.
///
/// # Type
///
/// ```text
/// fn(T, T)
/// ```
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, ErrorKind, VariableMap};
/// # use assert_matches::assert_matches;
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     assert_eq(1 + 2, 3); // this assertion is fine
///     assert_eq(3^2, 10); // this one will fail
/// "#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
/// let module = Environment::new()
///     .insert_native_fn("assert_eq", fns::AssertEq)
///     .compile_module("test_assert", &program)?;
///
/// let err = module.run().unwrap_err();
/// assert_eq!(*err.source().main_span().code().fragment(), "assert_eq(3^2, 10)");
/// assert_matches!(
///     err.source().kind(),
///     ErrorKind::NativeCall(ref msg) if msg == "Assertion failed: 3^2 == 10"
/// );
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct AssertEq;

impl<T: fmt::Display> NativeFn<T> for AssertEq {
    fn evaluate<'a>(
        &self,
        args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a, T>,
    ) -> EvalResult<'a, T> {
        ctx.check_args_count(&args, 2)?;

        let is_equal = args[0]
            .extra
            .eq_by_arithmetic(&args[1].extra, ctx.arithmetic());

        if is_equal {
            Ok(Value::void())
        } else {
            let err = if let (CodeFragment::Str(lhs), CodeFragment::Str(rhs)) =
                (args[0].fragment(), args[1].fragment())
            {
                ErrorKind::native(format!("Assertion failed: {} == {}", lhs, rhs))
            } else {
                ErrorKind::native("Equality assertion failed")
            };
            Err(ctx
                .call_site_error(err)
                .with_span(&args[0], AuxErrorInfo::arg_value(&args[0].extra))
                .with_span(&args[1], AuxErrorInfo::arg_value(&args[1].extra)))
        }
    }
}
