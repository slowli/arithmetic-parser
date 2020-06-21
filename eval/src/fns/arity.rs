use arithmetic_parser::Grammar;

use core::{fmt, marker::PhantomData};

use crate::{fns::extract_number, CallContext, EvalResult, NativeFn, SpannedValue, Value};

/// Arity of a function expressed as a trait.
///
/// # Implementation details
///
/// The necessity of this trait stems from the fact that stable Rust does not so far
/// support [const generics].
///
/// [const generics]: https://github.com/rust-lang/rust/issues/44580
pub trait Arity {
    /// Function arity.
    const ARITY: usize;
}

/// Wrapper of a function containing information about its arity.
///
/// The instances of this type are created by applying [`WithArity`] trait to a function or closure.
///
/// [`WithArity`]: trait.WithArity.html
#[derive(Debug, Clone, Copy)]
pub struct FnWrapper<A, F> {
    function: F,
    _arity: PhantomData<A>,
}

// Ideally, we would want to constrain `A` and `F`, but this would make it impossible to declare
// the constructor as `const fn`; see https://github.com/rust-lang/rust/issues/57563.
impl<A, F> FnWrapper<A, F> {
    /// Creates a new wrapper.
    ///
    /// Note that the created wrapper is not guaranteed to be usable as [`NativeFn`]. For this
    /// to be the case, `function` needs to be a function or an `Fn` closure,
    /// and the `A` type argument needs to be [`Arity`].
    pub const fn new(function: F) -> Self {
        Self {
            function,
            _arity: PhantomData
        }
    }
}

impl<A: Arity, F> FnWrapper<A, F> {
    /// Returns the arity of the wrapped function.
    pub fn arity(&self) -> usize {
        A::ARITY
    }
}

/// Helper trait allowing to extract arity information from a function.
///
/// # Examples
///
/// Import this trait into the scope and use it to construct functions usable for import
/// as [`NativeFn`]s:
///
/// ```
///
/// ```
pub trait WithArity<T, A: Arity> {
    /// Wraps the function so that information about its arity is retained.
    fn with_arity(self) -> FnWrapper<A, Self>
    where
        Self: Sized;
}

macro_rules! arity_fn {
    (@reverse $arg:ident) => { $arg };
    (@reverse $head:ident $($tail:ident)*) => {
        arity_fn!($($tail)*), $head
    };

    ($arity:expr, $name:ident => $($arg_name:ident : $t:ident),+) => {
        #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
        pub struct $name;

        impl Arity for $name {
            const ARITY: usize = $arity;
        }

        impl<T, F> WithArity<T, $name> for F
        where
            F: Fn($($t,)+) -> T
        {
            fn with_arity(self) -> FnWrapper<$name, Self> {
                FnWrapper {
                    function: self,
                    _arity: PhantomData,
                }
            }
        }

        impl<T, G, F> NativeFn<G> for FnWrapper<$name, F>
        where
            T: Clone + fmt::Debug,
            G: Grammar<Lit = T>,
            F: Fn($($t,)+) -> T,
        {
            fn evaluate<'a>(
                &self,
                args: Vec<SpannedValue<'a, G>>,
                context: &mut CallContext<'_, 'a>,
            ) -> EvalResult<'a, G> {
                const MSG: &str = concat!("Function requires ", $arity, " primitive argument(s)");

                context.check_args_count(&args, $arity)?;
                let mut args_iter = args.into_iter();

                $(
                    let $arg_name = args_iter.next().unwrap();
                    let $arg_name = extract_number(context, $arg_name, MSG)?;
                )+

                let output = (self.function)($($arg_name,)+);
                Ok(Value::Number(output))
            }
        }
    };
}

arity_fn!(1, Arity1 => x: T);
arity_fn!(2, Arity2 => x: T, y: T);
arity_fn!(3, Arity3 => x: T, y: T, z: T);
arity_fn!(4, Arity4 => x0: T, x1: T, x2: T, x3: T);
arity_fn!(5, Arity5 => x0: T, x1: T, x2: T, x3: T, x4: T);
arity_fn!(6, Arity6 => x0: T, x1: T, x2: T, x3: T, x4: T, x5: T);
arity_fn!(7, Arity7 => x0: T, x1: T, x2: T, x3: T, x4: T, x5: T, x6: T);
arity_fn!(8, Arity8 => x0: T, x1: T, x2: T, x3: T, x4: T, x5: T, x6: T, x7: T);
arity_fn!(9, Arity9 => x0: T, x1: T, x2: T, x3: T, x4: T, x5: T, x6: T, x7: T, x8: T);
arity_fn!(10, Arity10 => x0: T, x1: T, x2: T, x3: T, x4: T, x5: T, x6: T, x7: T, x8: T, x9: T);

/// Unary function wrapper.
pub type Unary<F> = FnWrapper<Arity1, F>;

/// Binary function wrapper.
pub type Binary<F> = FnWrapper<Arity2, F>;

/// Ternary function wrapper.
pub type Ternary<F> = FnWrapper<Arity3, F>;

/// Quaternary function wrapper.
pub type Quaternary<F> = FnWrapper<Arity4, F>;

/// Quinary function wrapper.
pub type Quinary<F> = FnWrapper<Arity5, F>;

/// Senary function wrapper.
pub type Senary<F> = FnWrapper<Arity6, F>;

/// Septenary function wrapper.
pub type Septenary<F> = FnWrapper<Arity7, F>;

/// Octonary function wrapper.
pub type Octonary<F> = FnWrapper<Arity8, F>;

/// Novenary function wrapper.
pub type Novenary<F> = FnWrapper<Arity9, F>;

/// Denary function wrapper.
pub type Denary<F> = FnWrapper<Arity10, F>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Interpreter;
    use arithmetic_parser::{grammars::F32Grammar, GrammarExt, Span};

    #[test]
    fn functions_work() {
        let unary_fn = (|x: f32| x + 3.0).with_arity();
        let binary_fn = f32::min.with_arity();
        let ternary_fn = (|x: f32, y: f32, z: f32| if x > 0.0 { y } else { z }).with_arity();

        let mut interpreter = Interpreter::new();
        interpreter
            .insert_native_fn("unary_fn", unary_fn)
            .insert_native_fn("binary_fn", binary_fn)
            .insert_native_fn("ternary_fn", ternary_fn);

        let program = r#"
            unary_fn(2) == 5 && binary_fn(1, -3) == -3 &&
                ternary_fn(1, 2, 3) == 2 && ternary_fn(-1, 2, 3) == 3
        "#;
        let block = F32Grammar::parse_statements(Span::new(program)).unwrap();
        let ret = interpreter.evaluate(&block).unwrap();
        assert_eq!(ret, Value::Bool(true));
    }
}
