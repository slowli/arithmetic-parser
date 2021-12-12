//! Standard functions for the interpreter, and the tools to define new native functions.
//!
//! # Defining native functions
//!
//! There are several ways to define new native functions:
//!
//! - Implement [`NativeFn`] manually. This is the most versatile approach, but it can be overly
//!   verbose.
//! - Use [`FnWrapper`] or the [`wrap`] function. This allows specifying arguments / output
//!   with custom types (such as `bool` or a [`Number`]), but does not work for non-`'static`
//!   types.
//! - Use [`wrap_fn`](crate::wrap_fn) or [`wrap_fn_with_context`](crate::wrap_fn_with_context)
//!   macros. These macros support
//!   the same eloquent interface as `wrap`, and also do not have `'static` requirement for args.
//!   As a downside, debugging compile-time errors when using macros can be rather painful.
//!
//! ## Why multiple ways to do the same thing?
//!
//! In the ideal world, `FnWrapper` would be used for all cases, since it does not involve
//! macro magic. Unfortunately, stable Rust currently does not provide means to describe
//! lifetime restrictions on args / return type of wrapped functions in the general case
//! (this requires [generic associated types][GAT]). As such, the (implicit) `'static` requirement
//! is a temporary measure, and macros fill the gaps in their usual clunky manner.
//!
//! [`Number`]: crate::Number
//! [GAT]: https://github.com/rust-lang/rust/issues/44265

use once_cell::unsync::OnceCell;

use core::{cmp::Ordering, fmt};

use crate::{
    alloc::Vec, error::AuxErrorInfo, CallContext, Error, ErrorKind, EvalResult, Function, NativeFn,
    Object, OpaqueRef, Prototype, SpannedValue, Value,
};
use arithmetic_parser::StripCode;

mod array;
mod assertions;
mod flow;
#[cfg(feature = "std")]
mod std;
mod wrapper;

#[cfg(feature = "std")]
pub use self::std::Dbg;
pub use self::{
    array::{Array, Filter, Fold, Len, Map, Merge, Push},
    assertions::{Assert, AssertClose, AssertEq, AssertFails},
    flow::{If, Loop, While},
    wrapper::{
        enforce_closure_type, wrap, Binary, ErrorOutput, FnWrapper, FromValueError,
        FromValueErrorKind, FromValueErrorLocation, IntoEvalResult, Quaternary, Ternary,
        TryFromValue, Unary,
    },
};

fn extract_primitive<'a, T, A>(
    ctx: &CallContext<'_, 'a, A>,
    value: SpannedValue<'a, T>,
    error_msg: &str,
) -> Result<T, Error<'a>> {
    match value.extra {
        Value::Prim(value) => Ok(value),
        _ => Err(ctx
            .call_site_error(ErrorKind::native(error_msg))
            .with_span(&value, AuxErrorInfo::InvalidArg)),
    }
}

fn extract_array<'a, T, A>(
    ctx: &CallContext<'_, 'a, A>,
    value: SpannedValue<'a, T>,
    error_msg: &str,
) -> Result<Vec<Value<'a, T>>, Error<'a>> {
    if let Value::Tuple(array) = value.extra {
        Ok(array.into())
    } else {
        let err = ErrorKind::native(error_msg);
        Err(ctx
            .call_site_error(err)
            .with_span(&value, AuxErrorInfo::InvalidArg))
    }
}

fn extract_object<'a, T, A>(
    ctx: &CallContext<'_, 'a, A>,
    value: SpannedValue<'a, T>,
    error_msg: &str,
) -> Result<Object<'a, T>, Error<'a>> {
    if let Value::Object(object) = value.extra {
        Ok(object)
    } else {
        let err = ErrorKind::native(error_msg);
        Err(ctx
            .call_site_error(err)
            .with_span(&value, AuxErrorInfo::InvalidArg))
    }
}

fn extract_fn<'a, T, A>(
    ctx: &CallContext<'_, 'a, A>,
    value: SpannedValue<'a, T>,
    error_msg: &str,
) -> Result<Function<'a, T>, Error<'a>> {
    if let Value::Function(function) = value.extra {
        Ok(function)
    } else {
        let err = ErrorKind::native(error_msg);
        Err(ctx
            .call_site_error(err)
            .with_span(&value, AuxErrorInfo::InvalidArg))
    }
}

/// Comparator functions on two primitive arguments. All functions use [`Arithmetic`] to determine
/// ordering between the args.
///
/// # Type
///
/// ```text
/// fn(Num, Num) -> Ordering // for `Compare::Raw`
/// fn(Num, Num) -> Num // for `Compare::Min` and `Compare::Max`
/// ```
///
/// [`Arithmetic`]: crate::arith::Arithmetic
///
/// # Examples
///
/// Using `min` function:
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, Value, env::VariableMap};
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     // Finds a minimum number in an array.
///     extended_min = |...xs| fold(xs, INFINITY, min);
///     extended_min(2, -3, 7, 1, 3) == -3
/// "#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
///
/// let module = Environment::new()
///     .insert("INFINITY", Value::Prim(f32::INFINITY))
///     .insert_native_fn("fold", fns::Fold)
///     .insert_native_fn("min", fns::Compare::Min)
///     .compile_module("test_min", &program)?;
/// assert_eq!(module.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
///
/// Using `cmp` function with [`Comparisons`](crate::env::Comparisons).
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, env::{Comparisons, VariableMap}, Environment, Value};
/// # use core::iter::FromIterator;
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     map((1, -7, 0, 2), |x| cmp(x, 0)) == (GREATER, LESS, EQUAL, GREATER)
/// "#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
///
/// let module = Environment::from_iter(Comparisons.iter())
///     .insert_native_fn("map", fns::Map)
///     .compile_module("test_cmp", &program)?;
/// assert_eq!(module.run()?, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum Compare {
    /// Returns an [`Ordering`] wrapped into an [`OpaqueRef`](crate::OpaqueRef),
    /// or [`Value::void()`] if the provided values are not comparable.
    Raw,
    /// Returns the minimum of the two values. If the values are equal / not comparable, returns the first one.
    Min,
    /// Returns the maximum of the two values. If the values are equal / not comparable, returns the first one.
    Max,
}

impl Compare {
    fn extract_primitives<'a, T>(
        mut args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a, T>,
    ) -> Result<(T, T), Error<'a>> {
        ctx.check_args_count(&args, 2)?;
        let y = args.pop().unwrap();
        let x = args.pop().unwrap();
        let x = extract_primitive(ctx, x, COMPARE_ERROR_MSG)?;
        let y = extract_primitive(ctx, y, COMPARE_ERROR_MSG)?;
        Ok((x, y))
    }
}

const COMPARE_ERROR_MSG: &str = "Compare requires 2 primitive arguments";

impl<T> NativeFn<T> for Compare {
    fn evaluate<'a>(
        &self,
        args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a, T>,
    ) -> EvalResult<'a, T> {
        let (x, y) = Self::extract_primitives(args, ctx)?;
        let maybe_ordering = ctx.arithmetic().partial_cmp(&x, &y);

        if let Self::Raw = self {
            Ok(maybe_ordering.map_or_else(Value::void, Value::opaque_ref))
        } else {
            let ordering =
                maybe_ordering.ok_or_else(|| ctx.call_site_error(ErrorKind::CannotCompare))?;
            let value = match (ordering, self) {
                (Ordering::Equal, _)
                | (Ordering::Less, Self::Min)
                | (Ordering::Greater, Self::Max) => x,
                _ => y,
            };
            Ok(Value::Prim(value))
        }
    }
}

/// Creates a new [`Prototype`] from the provided [`Object`].
///
/// The functions in the provided `Object` will ber used in method resolution when applying
/// methods to [`Value`]s having this prototype. All object fields can be accessed
/// from the prototype using generic field access notation. The prototype itself is a function
/// which will wrap provided tuples or objects so that they have this prototype.
///
/// See [`Prototype`] docs for more details on prototype mechanics.
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, Value, env::VariableMap};
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     Point = impl(#{
///         len: |{x, y}| sqrt(x * x + y * y),
///     });
///     pt = Point(#{ x: 3, y: 4 });
///     assert_close(pt.len(), 5);
/// "#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
///
/// Environment::new()
///     .insert_wrapped_fn("sqrt", f32::sqrt)
///     .insert_native_fn("impl", fns::CreatePrototype)
///     .insert_native_fn("assert_close", fns::AssertClose::new(1e-4))
///     .compile_module("test_impl", &program)?
///     .run()?;
/// # Ok(())
/// # }
/// ```
///
/// It is possible to define prototype hierarchies as well:
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, Value, env::VariableMap};
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     PointStatics = impl(#{
///         new: |Self, x, y| Self(#{ x, y }),
///         zero: |Self| Self(#{ x: 0, y: 0 }),
///     });
///     Point = impl(PointStatics(#{
///         len: |{x, y}| sqrt(x * x + y * y),
///     }));
///     pt = Point.new(3, 4);
///     assert_close(pt.len(), 5);
///     assert_eq(Point.zero().len(), 0);
/// "#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
/// // Testing snipped (virtually identical to the previous case)
/// # Environment::new()
/// #   .insert_wrapped_fn("sqrt", f32::sqrt)
/// #   .insert_native_fn("impl", fns::CreatePrototype)
/// #   .insert_native_fn("assert_eq", fns::AssertEq)
/// #   .insert_native_fn("assert_close", fns::AssertClose::new(1e-4))
/// #   .compile_module("test_impl", &program)?
/// #   .run()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct CreatePrototype;

impl<T> NativeFn<T> for CreatePrototype {
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a, T>,
    ) -> EvalResult<'a, T> {
        ctx.check_args_count(&args, 1)?;
        let object = extract_object(
            ctx,
            args.pop().unwrap(),
            "Function argument must be an object",
        )?;
        Ok(Prototype::from(object).into())
    }
}

/// Allows to define a value recursively, by referencing a value being created. This is particularly
/// useful when defining [`Prototype`]s.
///
/// It works like this:
///
/// - Provide a function as the only argument. The (only) argument of this function is the value
///   being created.
/// - Do not use the uninitialized value synchronously; only use it in inner function definitions.
/// - Return the created value from a function.
///
/// # Examples
///
/// Defining a recursive prototype:
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, Value, env::VariableMap};
/// # fn main() -> anyhow::Result<()> {
/// let program = r#"
///     Stack = defer(|Self| impl(#{
///         push: |self, item| Self(push(self, item)),
///         // ^ since `Self` is used in function definition, this is OK
///     }));
///     stack = Stack((1, 2)).push(3).push(4);
///     assert_eq(stack, Stack((1, 2, 3, 4)));
/// "#;
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
///
/// Environment::new()
///     .insert_native_fn("defer", fns::Defer)
///     .insert_native_fn("impl", fns::CreatePrototype)
///     .insert_native_fn("push", fns::Push)
///     .insert_native_fn("assert_eq", fns::AssertEq)
///     .compile_module("test_defer", &program)?
///     .run()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Defer;

impl<T: Clone + 'static> NativeFn<T> for Defer {
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a, T>,
    ) -> EvalResult<'a, T> {
        const ARG_ERROR: &str = "Argument must be a function";

        ctx.check_args_count(&args, 1)?;
        let function = extract_fn(ctx, args.pop().unwrap(), ARG_ERROR)?;
        let cell = OpaqueRef::with_identity_eq(ValueCell::<T>::default());
        let spanned_cell = ctx.apply_call_span(Value::Ref(cell.clone()));
        let return_value = function.evaluate(vec![spanned_cell], ctx)?;

        let cell = cell.downcast_ref::<ValueCell<T>>().unwrap();
        // ^ `unwrap()` is safe by construction
        cell.set(return_value.clone().strip_code());
        Ok(return_value)
    }
}

#[derive(Debug)]
pub(crate) struct ValueCell<T> {
    inner: OnceCell<Value<'static, T>>,
}

impl<T> Default for ValueCell<T> {
    fn default() -> Self {
        Self {
            inner: OnceCell::new(),
        }
    }
}

impl<'a, T: 'static + fmt::Debug> From<ValueCell<T>> for Value<'a, T> {
    fn from(cell: ValueCell<T>) -> Self {
        Self::Ref(OpaqueRef::with_identity_eq(cell))
    }
}

impl<T> ValueCell<T> {
    /// Gets the internally stored value, or `None` if the cell was not initialized yet.
    pub fn get(&self) -> Option<&Value<'static, T>> {
        self.inner.get()
    }

    fn set(&self, value: Value<'static, T>) {
        self.inner
            .set(value)
            .map_err(drop)
            .expect("Repeated `ValueCell` assignment");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        env::{Environment, Filler},
        exec::{ExecutableModule, WildcardId},
    };

    use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
    use assert_matches::assert_matches;

    #[test]
    fn if_basic() {
        let block = r#"
            x = 1.0;
            if(x < 2, x + 5, 3 - x)
        "#;
        let block = Untyped::<F32Grammar>::parse_statements(block).unwrap();
        let module = ExecutableModule::builder(WildcardId, &block)
            .unwrap()
            .with_import("if", Value::native_fn(If))
            .build();
        assert_eq!(module.run().unwrap(), Value::Prim(6.0));
    }

    #[test]
    fn if_with_closures() {
        let block = r#"
            x = 4.5;
            if(x < 2, || x + 5, || 3 - x)()
        "#;
        let block = Untyped::<F32Grammar>::parse_statements(block).unwrap();
        let module = ExecutableModule::builder(WildcardId, &block)
            .unwrap()
            .with_import("if", Value::native_fn(If))
            .build();
        assert_eq!(module.run().unwrap(), Value::Prim(-1.5));
    }

    #[test]
    fn cmp_sugar() {
        let program = "x = 1.0; x > 0 && x <= 3";
        let block = Untyped::<F32Grammar>::parse_statements(program).unwrap();
        let module = ExecutableModule::builder(WildcardId, &block)
            .unwrap()
            .build();
        assert_eq!(module.run().unwrap(), Value::Bool(true));

        let bogus_program = "x = 1.0; x > (1, 2)";
        let bogus_block = Untyped::<F32Grammar>::parse_statements(bogus_program).unwrap();
        let bogus_module = ExecutableModule::builder(WildcardId, &bogus_block)
            .unwrap()
            .build();

        let err = bogus_module.run().unwrap_err();
        let err = err.source();
        assert_matches!(err.kind(), ErrorKind::CannotCompare);
        assert_eq!(*err.main_span().code().fragment(), "(1, 2)");
    }

    #[test]
    fn loop_basic() {
        let program = r#"
            // Finds the greatest power of 2 lesser or equal to the value.
            discrete_log2 = |x| {
                loop(0, |i| {
                    continue = 2^i <= x;
                    (continue, if(continue, i + 1, i - 1))
                })
            };

            (discrete_log2(1), discrete_log2(2),
                discrete_log2(4), discrete_log2(6.5), discrete_log2(1000))
        "#;
        let block = Untyped::<F32Grammar>::parse_statements(program).unwrap();

        let module = ExecutableModule::builder(WildcardId, &block)
            .unwrap()
            .with_import("loop", Value::native_fn(Loop))
            .with_import("if", Value::native_fn(If))
            .build();

        assert_eq!(
            module.run().unwrap(),
            Value::from(vec![
                Value::Prim(0.0),
                Value::Prim(1.0),
                Value::Prim(2.0),
                Value::Prim(2.0),
                Value::Prim(9.0),
            ])
        );
    }

    #[test]
    fn max_value_with_fold() {
        let program = r#"
            max_value = |...xs| {
                fold(xs, -Inf, |acc, x| if(x > acc, x, acc))
            };
            max_value(1, -2, 7, 2, 5) == 7 && max_value(3, -5, 9) == 9
        "#;
        let block = Untyped::<F32Grammar>::parse_statements(program).unwrap();

        let module = ExecutableModule::builder(WildcardId, &block)
            .unwrap()
            .with_import("Inf", Value::Prim(f32::INFINITY))
            .with_import("fold", Value::native_fn(Fold))
            .with_import("if", Value::native_fn(If))
            .build();

        assert_eq!(module.run().unwrap(), Value::Bool(true));
    }

    #[test]
    fn reverse_list_with_fold() {
        const SAMPLES: &[(&[f32], &[f32])] = &[
            (&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]),
            (&[], &[]),
            (&[1.0], &[1.0]),
        ];

        let program = r#"
            reverse = |xs| {
                fold(xs, (), |acc, x| merge((x,), acc))
            };
            xs = (-4, 3, 0, 1);
            reverse(xs) == (1, 0, 3, -4)
        "#;
        let block = Untyped::<F32Grammar>::parse_statements(program).unwrap();

        let module = ExecutableModule::builder(WildcardId, &block)
            .unwrap()
            .with_import("merge", Value::native_fn(Merge))
            .with_import("fold", Value::native_fn(Fold))
            .build();

        let mut env = module.imports().into_iter().collect::<Environment<'_, _>>();
        assert_eq!(module.run_in_env(&mut env).unwrap(), Value::Bool(true));

        let test_block = Untyped::<F32Grammar>::parse_statements("reverse(xs)").unwrap();
        let mut test_module = ExecutableModule::builder("test", &test_block)
            .unwrap()
            .with_import("reverse", env["reverse"].clone())
            .with_imports_from(&Filler::void(&["xs"]))
            .build();

        for &(input, expected) in SAMPLES {
            let input = input.iter().copied().map(Value::Prim).collect();
            let expected = expected.iter().copied().map(Value::Prim).collect();
            test_module.set_import("xs", Value::Tuple(input));
            assert_eq!(test_module.run().unwrap(), Value::Tuple(expected));
        }
    }

    #[test]
    fn error_with_min_function_args() {
        let program = "5 - min(1, (2, 3))";
        let block = Untyped::<F32Grammar>::parse_statements(program).unwrap();
        let module = ExecutableModule::builder(WildcardId, &block)
            .unwrap()
            .with_import("min", Value::native_fn(Compare::Min))
            .build();

        let err = module.run().unwrap_err();
        let err = err.source();
        assert_eq!(*err.main_span().code().fragment(), "min(1, (2, 3))");
        assert_matches!(
            err.kind(),
            ErrorKind::NativeCall(ref msg) if msg.contains("requires 2 primitive arguments")
        );
    }

    #[test]
    fn error_with_min_function_incomparable_args() {
        let program = "5 - min(1, NAN)";
        let block = Untyped::<F32Grammar>::parse_statements(program).unwrap();
        let module = ExecutableModule::builder(WildcardId, &block)
            .unwrap()
            .with_import("NAN", Value::Prim(f32::NAN))
            .with_import("min", Value::native_fn(Compare::Min))
            .build();

        let err = module.run().unwrap_err();
        let err = err.source();
        assert_eq!(*err.main_span().code().fragment(), "min(1, NAN)");
        assert_matches!(err.kind(), ErrorKind::CannotCompare);
    }
}
