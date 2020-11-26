//! Standard libraries for different arithmetics.

use num_complex::{Complex, Complex32, Complex64};
use num_traits::{CheckedRem, WrappingNeg};

use std::{fmt, iter::FromIterator, ops};

use arithmetic_eval::{fns, Comparisons, Environment, Number, Prelude, Value, VariableMap};
use arithmetic_parser::grammars::NumLiteral;

#[derive(Debug, Clone, Copy)]
#[allow(clippy::type_complexity)] // not that complex, really
pub struct StdLibrary<T: 'static> {
    constants: &'static [(&'static str, T)],
    unary: &'static [(&'static str, fn(T) -> T)],
    binary: &'static [(&'static str, fn(T, T) -> T)],
}

impl<'a, T: ReplLiteral> VariableMap<'a, T> for StdLibrary<T> {
    fn get_variable(&self, name: &str) -> Option<Value<'a, T>> {
        self.variables()
            .find_map(|(var_name, value)| if var_name == name { Some(value) } else { None })
    }
}

impl<T: ReplLiteral> StdLibrary<T> {
    fn variables(self) -> impl Iterator<Item = (&'static str, Value<'static, T>)> {
        let constants = self
            .constants
            .iter()
            .map(|(name, constant)| (*name, Value::Number(*constant)));
        let unary_fns = self
            .unary
            .iter()
            .copied()
            .map(|(name, function)| (name, Value::native_fn(fns::Unary::new(function))));
        let binary_fns = self
            .binary
            .iter()
            .copied()
            .map(|(name, function)| (name, Value::native_fn(fns::Binary::new(function))));

        constants.chain(unary_fns).chain(binary_fns)
    }
}

pub trait ReplLiteral: NumLiteral + Number + PartialEq + fmt::Display {
    const STD_LIB: StdLibrary<Self>;
}

macro_rules! declare_int_functions {
    ($type:ident) => {
        impl ReplLiteral for $type {
            const STD_LIB: StdLibrary<$type> = StdLibrary {
                constants: &[
                    ("MIN_VALUE", $type::min_value()),
                    ("MAX_VALUE", $type::max_value()),
                ],

                unary: &[
                    ("count_ones", |x| x.count_ones() as $type),
                    ("count_zeros", |x| x.count_zeros() as $type),
                    ("leading_zeros", |x| x.leading_zeros() as $type),
                    ("trailing_zeros", |x| x.trailing_zeros() as $type),
                ],

                binary: &[
                    // bit operations
                    ("bit_and", |x, y| x & y),
                    ("bit_or", |x, y| x | y),
                    ("xor", |x, y| x ^ y),
                ],
            };
        }
    };
}

declare_int_functions!(u64);
declare_int_functions!(i64);
declare_int_functions!(u128);
declare_int_functions!(i128);

pub fn create_int_env<T>(wrapping: bool) -> Environment<'static, T>
where
    T: ReplLiteral + ops::Rem + WrappingNeg + CheckedRem,
{
    const REM_ERROR_MSG: &str = "Cannot calculate remainder for a divisor of zero";

    let mut env: Environment<'static, T> = Prelude
        .iter()
        .chain(Comparisons.iter())
        .chain(T::STD_LIB.variables())
        .collect();

    if wrapping {
        env.insert_wrapped_fn("rem", |x: T, y: T| {
            if y == T::zero() {
                Err(REM_ERROR_MSG.to_owned())
            } else if y.wrapping_neg().is_one() {
                // Prevent a panic with `T::min_value() % -1`.
                Ok(T::zero())
            } else {
                Ok(x % y)
            }
        });
    } else {
        env.insert_wrapped_fn("rem", |x: T, y: T| {
            x.checked_rem(&y).ok_or_else(|| REM_ERROR_MSG.to_owned())
        });
    }
    env
}

pub fn create_modular_env(modulus: u64) -> Environment<'static, u64> {
    let mut env = Environment::from_iter(Prelude.iter());
    env.insert("MAX_VALUE", Value::Number(modulus - 1));
    env
}

macro_rules! declare_real_functions {
    ($type:ident) => {
        impl ReplLiteral for $type {
            const STD_LIB: StdLibrary<$type> = StdLibrary {
                constants: &[("E", std::$type::consts::E), ("PI", std::$type::consts::PI)],

                unary: &[
                    // Rounding functions.
                    ("floor", $type::floor),
                    ("ceil", $type::ceil),
                    ("round", $type::round),
                    ("frac", $type::fract),
                    // Exponential functions.
                    ("exp", $type::exp),
                    ("ln", $type::ln),
                    ("sinh", $type::sinh),
                    ("cosh", $type::cosh),
                    ("tanh", $type::tanh),
                    ("asinh", $type::asinh),
                    ("acosh", $type::acosh),
                    ("atanh", $type::atanh),
                    // Trigonometric functions.
                    ("sin", $type::sin),
                    ("cos", $type::cos),
                    ("tan", $type::tan),
                    ("asin", $type::asin),
                    ("acos", $type::acos),
                    ("atan", $type::atan),
                ],

                binary: &[],
            };
        }
    };
}

declare_real_functions!(f32);
declare_real_functions!(f64);

pub fn create_float_env<T: ReplLiteral>() -> Environment<'static, T> {
    Prelude
        .iter()
        .chain(Comparisons.iter())
        .chain(T::STD_LIB.variables())
        .collect()
}

macro_rules! declare_complex_functions {
    ($type:ident, $real:ident) => {
        impl ReplLiteral for $type {
            const STD_LIB: StdLibrary<$type> = StdLibrary {
                constants: &[
                    ("E", Complex::new(std::$real::consts::E, 0.0)),
                    ("PI", Complex::new(std::$real::consts::PI, 0.0)),
                ],

                unary: &[
                    ("norm", |x| Complex::new(x.norm(), 0.0)),
                    ("arg", |x| Complex::new(x.arg(), 0.0)),
                    // Exponential functions.
                    ("exp", |x| x.exp()),
                    ("ln", |x| x.ln()),
                    ("sinh", |x| x.sinh()),
                    ("cosh", |x| x.cosh()),
                    ("tanh", |x| x.tanh()),
                    ("asinh", |x| x.asinh()),
                    ("acosh", |x| x.acosh()),
                    ("atanh", |x| x.atanh()),
                    // Trigonometric functions.
                    ("sin", |x| x.sin()),
                    ("cos", |x| x.cos()),
                    ("tan", |x| x.tan()),
                    ("asin", |x| x.asin()),
                    ("acos", |x| x.acos()),
                    ("atan", |x| x.atan()),
                ],

                binary: &[],
            };
        }
    };
}

declare_complex_functions!(Complex32, f32);
declare_complex_functions!(Complex64, f64);

pub fn create_complex_env<T: ReplLiteral>() -> Environment<'static, T> {
    Prelude.iter().chain(T::STD_LIB.variables()).collect()
}
