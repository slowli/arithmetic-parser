//! Functions that require the Rust standard library.

use std::fmt;

use crate::{CallContext, EvalResult, ModuleId, NativeFn, SpannedValue, Value};
use arithmetic_parser::CodeFragment;

/// Acts similarly to the `dbg!` macro, outputting the argument(s) to stderr and returning
/// them. If a single argument is provided, it's returned as-is; otherwise, the arguments
/// are wrapped into a tuple.
///
/// # Examples
///
/// ```
/// # use arithmetic_parser::grammars::{F32Grammar, Parse, Untyped};
/// # use arithmetic_eval::{fns, Environment, Value, VariableMap};
/// # fn main() -> anyhow::Result<()> {
/// let program = "dbg(1 + 2) > 2.5";
/// let program = Untyped::<F32Grammar>::parse_statements(program)?;
/// let module = Environment::new()
///     .insert_native_fn("dbg", fns::Dbg)
///     .compile_module("test_assert", &program)?;
///
/// let value = module.run()?;
/// // Should output `[test_assert:1:5] 1 + 2 = 3` to stderr.
/// assert_eq!(value, Value::Bool(true));
/// # Ok(())
/// # }
/// ```
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
#[derive(Debug, Clone, Copy)]
pub struct Dbg;

impl Dbg {
    fn print_value<T: fmt::Display>(module_id: &dyn ModuleId, value: &SpannedValue<'_, T>) {
        match value.fragment() {
            CodeFragment::Str(code) => eprintln!(
                "[{module}:{line}:{col}] {code} = {val}",
                module = module_id,
                line = value.location_line(),
                col = value.get_column(),
                code = code,
                val = value.extra
            ),
            CodeFragment::Stripped(_) => eprintln!(
                "[{module}:{line}:{col}] {val}",
                module = module_id,
                line = value.location_line(),
                col = value.get_column(),
                val = value.extra
            ),
        }
    }
}

impl<T: fmt::Display> NativeFn<T> for Dbg {
    fn evaluate<'a>(
        &self,
        mut args: Vec<SpannedValue<'a, T>>,
        ctx: &mut CallContext<'_, 'a, T>,
    ) -> EvalResult<'a, T> {
        let module_id = ctx.call_span().module_id();
        for arg in &args {
            Self::print_value(module_id, arg);
        }

        Ok(if args.len() == 1 {
            args.pop().unwrap().extra
        } else {
            Value::Tuple(args.into_iter().map(|spanned| spanned.extra).collect())
        })
    }
}
