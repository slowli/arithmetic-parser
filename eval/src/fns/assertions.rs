//! Assertion functions.

use core::{cmp::Ordering, fmt};

use super::extract_fn;
use crate::{
    alloc::Vec,
    error::{AuxErrorInfo, Error},
    CallContext, ErrorKind, EvalResult, NativeFn, SpannedValue, Value,
};

/// Assertion function.
///
/// # Type
///
/// (using [`arithmetic-typing`](https://docs.rs/arithmetic-typing/) notation)
///
/// ```text
/// (Bool) -> ()
/// ```
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, ErrorKind, ExecutableModule};
/// # use assert_matches::assert_matches;
/// # fn main() -> anyhow::Result<()> {
/// let program = "
///     assert(1 + 2 != 5); // this assertion is fine
///     assert(3^2 > 10); // this one will fail
/// ";
/// let module = Untyped::<F32Grammar>::parse_statements(program)?;
/// let module = ExecutableModule::new("test_assert", &module)?;
///
/// let mut env = Environment::new();
/// env.insert_native_fn("assert", fns::Assert);
///
/// let err = module.with_env(&env)?.run().unwrap_err();
/// assert_eq!(
///     err.source().location().in_module().span(&program),
///     "assert(3^2 > 10)"
/// );
/// assert_matches!(
///     err.source().kind(),
///     ErrorKind::NativeCall(msg) if msg == "Assertion failed"
/// );
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Assert;

impl<T> NativeFn<T> for Assert {
    fn evaluate<'a>(
        &self,
        args: Vec<SpannedValue<T>>,
        ctx: &mut CallContext<'_, T>,
    ) -> EvalResult<T> {
        ctx.check_args_count(&args, 1)?;
        match args[0].extra {
            Value::Bool(true) => Ok(Value::void()),

            Value::Bool(false) => {
                let err = ErrorKind::native("Assertion failed");
                Err(ctx.call_site_error(err))
            }

            _ => {
                let err = ErrorKind::native("`assert` requires a single boolean argument");
                Err(ctx
                    .call_site_error(err)
                    .with_location(&args[0], AuxErrorInfo::InvalidArg))
            }
        }
    }
}

fn create_error_with_values<T: fmt::Display>(
    err: ErrorKind,
    args: &[SpannedValue<T>],
    ctx: &CallContext<'_, T>,
) -> Error {
    ctx.call_site_error(err)
        .with_location(&args[0], AuxErrorInfo::arg_value(&args[0].extra))
        .with_location(&args[1], AuxErrorInfo::arg_value(&args[1].extra))
}

/// Equality assertion function.
///
/// # Type
///
/// (using [`arithmetic-typing`](https://docs.rs/arithmetic-typing/) notation)
///
/// ```text
/// ('T, 'T) -> ()
/// ```
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, ErrorKind, ExecutableModule};
/// # use assert_matches::assert_matches;
/// # fn main() -> anyhow::Result<()> {
/// let program = "
///     assert_eq(1 + 2, 3); // this assertion is fine
///     assert_eq(3^2, 10); // this one will fail
/// ";
/// let module = Untyped::<F32Grammar>::parse_statements(program)?;
/// let module = ExecutableModule::new("test_assert", &module)?;
///
/// let mut env = Environment::new();
/// env.insert_native_fn("assert_eq", fns::AssertEq);
///
/// let err = module.with_env(&env)?.run().unwrap_err();
/// assert_eq!(
///     err.source().location().in_module().span(program),
///     "assert_eq(3^2, 10)"
/// );
/// assert_matches!(
///     err.source().kind(),
///     ErrorKind::NativeCall(msg) if msg == "Equality assertion failed"
/// );
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct AssertEq;

impl<T: fmt::Display> NativeFn<T> for AssertEq {
    fn evaluate(&self, args: Vec<SpannedValue<T>>, ctx: &mut CallContext<'_, T>) -> EvalResult<T> {
        ctx.check_args_count(&args, 2)?;

        let is_equal = args[0]
            .extra
            .eq_by_arithmetic(&args[1].extra, ctx.arithmetic());

        if is_equal {
            Ok(Value::void())
        } else {
            let err = ErrorKind::native("Equality assertion failed");
            Err(create_error_with_values(err, &args, ctx))
        }
    }
}

/// Assertion that two values are close to each other.
///
/// Unlike [`AssertEq`], the arguments must be primitive. The function is parameterized by
/// the tolerance threshold.
///
/// # Type
///
/// (using [`arithmetic-typing`](https://docs.rs/arithmetic-typing/) notation)
///
/// ```text
/// (Num, Num) -> ()
/// ```
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, ExecutableModule};
/// # use assert_matches::assert_matches;
/// # fn main() -> anyhow::Result<()> {
/// let program = "
///     assert_close(sqrt(9), 3); // this assertion is fine
///     assert_close(sqrt(10), 3); // this one should fail
/// ";
/// let module = Untyped::<F32Grammar>::parse_statements(program)?;
/// let module = ExecutableModule::new("test_assert", &module)?;
///
/// let mut env = Environment::new();
/// env.insert_native_fn("assert_close", fns::AssertClose::new(1e-4))
///     .insert_wrapped_fn("sqrt", f32::sqrt);
///
/// let err = module.with_env(&env)?.run().unwrap_err();
/// assert_eq!(
///     err.source().location().in_module().span(program),
///     "assert_close(sqrt(10), 3)"
/// );
/// # Ok(())
/// # }
/// ```
// TODO: support structured values?
#[derive(Debug, Clone, Copy)]
pub struct AssertClose<T> {
    tolerance: T,
}

impl<T> AssertClose<T> {
    /// Creates a function with the specified tolerance threshold. No checks are performed
    /// on the threshold (e.g., that it is positive).
    pub const fn new(tolerance: T) -> Self {
        Self { tolerance }
    }

    fn extract_primitive_ref<'r>(
        ctx: &mut CallContext<'_, T>,
        value: &'r SpannedValue<T>,
    ) -> Result<&'r T, Error> {
        const ARG_ERROR: &str = "Function arguments must be primitive numbers";

        match &value.extra {
            Value::Prim(value) => Ok(value),
            _ => Err(ctx
                .call_site_error(ErrorKind::native(ARG_ERROR))
                .with_location(value, AuxErrorInfo::InvalidArg)),
        }
    }
}

impl<T: Clone + fmt::Display> NativeFn<T> for AssertClose<T> {
    fn evaluate(&self, args: Vec<SpannedValue<T>>, ctx: &mut CallContext<'_, T>) -> EvalResult<T> {
        ctx.check_args_count(&args, 2)?;
        let rhs = Self::extract_primitive_ref(ctx, &args[0])?;
        let lhs = Self::extract_primitive_ref(ctx, &args[1])?;

        let arith = ctx.arithmetic();
        let diff = match arith.partial_cmp(lhs, rhs) {
            Some(Ordering::Less | Ordering::Equal) => arith.sub(rhs.clone(), lhs.clone()),
            Some(Ordering::Greater) => arith.sub(lhs.clone(), rhs.clone()),
            None => {
                let err = ErrorKind::native("Values are not comparable");
                return Err(create_error_with_values(err, &args, ctx));
            }
        };
        let diff = diff.map_err(|err| ctx.call_site_error(ErrorKind::Arithmetic(err)))?;

        match arith.partial_cmp(&diff, &self.tolerance) {
            Some(Ordering::Less | Ordering::Equal) => Ok(Value::void()),
            Some(Ordering::Greater) => {
                let err = ErrorKind::native("Values are not close");
                Err(create_error_with_values(err, &args, ctx))
            }
            None => {
                let err = ErrorKind::native("Error comparing value difference to tolerance");
                Err(ctx.call_site_error(err))
            }
        }
    }
}

/// Assertion that the provided function raises an error. Errors can optionally be matched
/// against a predicate.
///
/// If an error is raised, but does not match the predicate, it is bubbled up.
///
/// # Type
///
/// (using [`arithmetic-typing`](https://docs.rs/arithmetic-typing/) notation)
///
/// ```text
/// (() -> 'T) -> ()
/// ```
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, ExecutableModule};
/// # use assert_matches::assert_matches;
/// # fn main() -> anyhow::Result<()> {
/// let program = "
///     obj = #{ x: 3 };
///     assert_fails(|| obj.x + obj.y); // pass: `obj.y` is not defined
///     assert_fails(|| obj.x); // fail: function executes successfully
/// ";
/// let module = Untyped::<F32Grammar>::parse_statements(program)?;
/// let module = ExecutableModule::new("test_assert", &module)?;
///
/// let mut env = Environment::new();
/// env.insert_native_fn("assert_fails", fns::AssertFails::default());
///
/// let err = module.with_env(&env)?.run().unwrap_err();
/// assert_eq!(
///     err.source().location().in_module().span(program),
///     "assert_fails(|| obj.x)"
/// );
/// # Ok(())
/// # }
/// ```
///
/// Custom error matching:
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{ErrorKind, fns, Environment, ExecutableModule};
/// # use assert_matches::assert_matches;
/// # fn main() -> anyhow::Result<()> {
/// let assert_fails = fns::AssertFails::new(|err| {
///     matches!(err.kind(), ErrorKind::NativeCall(_))
/// });
///
/// let program = "
///     assert_fails(|| assert_fails(1)); // pass: native error
///     assert_fails(assert_fails); // fail: arg len mismatch
/// ";
/// let module = Untyped::<F32Grammar>::parse_statements(program)?;
/// let module = ExecutableModule::new("test_assert", &module)?;
///
/// let mut env = Environment::new();
/// env.insert_native_fn("assert_fails", assert_fails);
///
/// let err = module.with_env(&env)?.run().unwrap_err();
/// assert_eq!(
///     err.source().location().in_module().span(program),
///     "assert_fails(assert_fails)"
/// );
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy)]
pub struct AssertFails {
    error_matcher: fn(&Error) -> bool,
}

impl fmt::Debug for AssertFails {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("AssertFails").finish()
    }
}

impl Default for AssertFails {
    fn default() -> Self {
        Self {
            error_matcher: |_| true,
        }
    }
}

impl AssertFails {
    /// Creates an assertion function with a custom error matcher. If the error does not match,
    /// the assertion will fail, and the error will bubble up.
    pub fn new(error_matcher: fn(&Error) -> bool) -> Self {
        Self { error_matcher }
    }
}

impl<T: 'static + Clone> NativeFn<T> for AssertFails {
    fn evaluate(
        &self,
        mut args: Vec<SpannedValue<T>>,
        ctx: &mut CallContext<'_, T>,
    ) -> EvalResult<T> {
        const ARG_ERROR: &str = "Single argument must be a function";

        ctx.check_args_count(&args, 1)?;
        let closure = extract_fn(ctx, args.pop().unwrap(), ARG_ERROR)?;
        match closure.evaluate(Vec::new(), ctx) {
            Ok(_) => {
                let err = ErrorKind::native("Function did not fail");
                Err(ctx.call_site_error(err))
            }
            Err(err) => {
                if (self.error_matcher)(&err) {
                    Ok(Value::void())
                } else {
                    // Pass the error through.
                    Err(err)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use arithmetic_parser::{Location, LvalueLen};
    use assert_matches::assert_matches;

    use super::*;
    use crate::{arith::CheckedArithmetic, exec::WildcardId, Environment, Object};

    fn span_value<T>(value: Value<T>) -> SpannedValue<T> {
        Location::from_str("", ..).copy_with_extra(value)
    }

    #[test]
    fn assert_basics() {
        let env = Environment::with_arithmetic(<CheckedArithmetic>::new());
        let mut ctx = CallContext::<u32>::mock(WildcardId, Location::from_str("", ..), &env);

        let err = Assert.evaluate(vec![], &mut ctx).unwrap_err();
        assert_matches!(err.kind(), ErrorKind::ArgsLenMismatch { .. });

        let invalid_arg = span_value(Value::Prim(1));
        let err = Assert.evaluate(vec![invalid_arg], &mut ctx).unwrap_err();
        assert_matches!(
            err.kind(),
            ErrorKind::NativeCall(s) if s.contains("requires a single boolean argument")
        );

        let false_arg = span_value(Value::Bool(false));
        let err = Assert.evaluate(vec![false_arg], &mut ctx).unwrap_err();
        assert_matches!(
            err.kind(),
            ErrorKind::NativeCall(s) if s.contains("Assertion failed")
        );

        let true_arg = span_value(Value::Bool(true));
        let return_value = Assert.evaluate(vec![true_arg.clone()], &mut ctx).unwrap();
        assert!(return_value.is_void(), "{return_value:?}");

        let err = Assert
            .evaluate(vec![true_arg.clone(), true_arg], &mut ctx)
            .unwrap_err();
        assert_matches!(err.kind(), ErrorKind::ArgsLenMismatch { .. });
    }

    #[test]
    fn assert_eq_basics() {
        let env = Environment::with_arithmetic(<CheckedArithmetic>::new());
        let mut ctx = CallContext::<u32>::mock(WildcardId, Location::from_str("", ..), &env);

        let err = AssertEq.evaluate(vec![], &mut ctx).unwrap_err();
        assert_matches!(err.kind(), ErrorKind::ArgsLenMismatch { .. });

        let x = span_value(Value::Prim(1));
        let y = span_value(Value::Prim(2));
        let err = AssertEq.evaluate(vec![x.clone(), y], &mut ctx).unwrap_err();
        assert_matches!(
            err.kind(),
            ErrorKind::NativeCall(s) if s.contains("assertion failed")
        );

        let return_value = AssertEq.evaluate(vec![x.clone(), x], &mut ctx).unwrap();
        assert!(return_value.is_void(), "{return_value:?}");
    }

    #[test]
    fn assert_close_basics() {
        let assert_close = AssertClose::new(1e-3);
        let env = Environment::new();
        let mut ctx = CallContext::<f32>::mock(WildcardId, Location::from_str("", ..), &env);

        let err = assert_close.evaluate(vec![], &mut ctx).unwrap_err();
        assert_matches!(err.kind(), ErrorKind::ArgsLenMismatch { .. });

        let one_arg = span_value(Value::Prim(1.0));
        let invalid_args = [
            Value::Bool(true),
            vec![Value::Prim(1.0)].into(),
            Object::just("test", Value::Prim(1.0)).into(),
        ];
        for invalid_arg in invalid_args {
            let err = assert_close
                .evaluate(vec![one_arg.clone(), span_value(invalid_arg)], &mut ctx)
                .unwrap_err();
            assert_matches!(
                err.kind(),
                ErrorKind::NativeCall(s) if s.contains("must be primitive numbers")
            );
        }

        let distant_values = &[(0.0, 1.0), (1.0, 1.01), (0.0, f32::INFINITY)];
        for &(x, y) in distant_values {
            let x = span_value(Value::Prim(x));
            let y = span_value(Value::Prim(y));
            let err = assert_close.evaluate(vec![x, y], &mut ctx).unwrap_err();
            assert_matches!(
                err.kind(),
                ErrorKind::NativeCall(s) if s.contains("Values are not close")
            );
        }

        let non_comparable_values = &[(0.0, f32::NAN), (f32::NAN, 1.0), (f32::NAN, f32::NAN)];
        for &(x, y) in non_comparable_values {
            let x = span_value(Value::Prim(x));
            let y = span_value(Value::Prim(y));
            let err = assert_close.evaluate(vec![x, y], &mut ctx).unwrap_err();
            assert_matches!(
                err.kind(),
                ErrorKind::NativeCall(s) if s.contains("Values are not comparable")
            );
        }

        let close_values = &[(1.0, 0.9999), (0.9999, 1.0), (1.0, 1.0)];
        for &(x, y) in close_values {
            let x = span_value(Value::Prim(x));
            let y = span_value(Value::Prim(y));
            let return_value = assert_close.evaluate(vec![x, y], &mut ctx).unwrap();
            assert!(return_value.is_void(), "{return_value:?}");
        }
    }

    #[test]
    fn assert_fails_basics() {
        let assert_fails = AssertFails::default();
        let env = Environment::new();
        let mut ctx = CallContext::<f32>::mock(WildcardId, Location::from_str("", ..), &env);

        let err = assert_fails.evaluate(vec![], &mut ctx).unwrap_err();
        assert_matches!(err.kind(), ErrorKind::ArgsLenMismatch { .. });

        let invalid_arg = span_value(Value::Prim(1.0));
        let err = assert_fails
            .evaluate(vec![invalid_arg], &mut ctx)
            .unwrap_err();
        assert_matches!(
            err.kind(),
            ErrorKind::NativeCall(s) if s.contains("must be a function")
        );

        let successful_fn = span_value(Value::wrapped_fn(|| true));
        let err = assert_fails
            .evaluate(vec![successful_fn], &mut ctx)
            .unwrap_err();
        assert_matches!(
            err.kind(),
            ErrorKind::NativeCall(s) if s.contains("Function did not fail")
        );

        let failing_fn = Value::wrapped_fn(|| Err::<f32, _>("oops".to_owned()));
        let return_value = assert_fails
            .evaluate(vec![span_value(failing_fn)], &mut ctx)
            .unwrap();
        assert!(return_value.is_void(), "{return_value:?}");
    }

    #[test]
    fn assert_fails_with_custom_matcher() {
        let assert_fails = AssertFails::new(
            |err| matches!(err.kind(), ErrorKind::NativeCall(msg) if msg == "oops"),
        );
        let env = Environment::new();
        let mut ctx = CallContext::<f32>::mock(WildcardId, Location::from_str("", ..), &env);

        let wrong_fn = Value::wrapped_fn(f32::abs);
        let err = assert_fails
            .evaluate(vec![span_value(wrong_fn)], &mut ctx)
            .unwrap_err();
        assert_matches!(
            err.kind(),
            ErrorKind::ArgsLenMismatch { def, call: 0 } if *def == LvalueLen::Exact(1)
        );

        let failing_fn = Value::wrapped_fn(|| Err::<f32, _>("oops".to_owned()));
        let return_value = assert_fails
            .evaluate(vec![span_value(failing_fn)], &mut ctx)
            .unwrap();
        assert!(return_value.is_void(), "{return_value:?}");
    }
}
