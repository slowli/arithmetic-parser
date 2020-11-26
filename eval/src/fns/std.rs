//! Functions that require the standard library.

use std::fmt;

use crate::{CallContext, EvalResult, ModuleId, NativeFn, SpannedValue, Value};
use arithmetic_parser::CodeFragment;

/// FIXME
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
