//! Standard libraries for different arithmetics.

use std::{fmt, iter, ops};

use arithmetic_eval::{
    arith::{
        Arithmetic, ArithmeticExt, CheckedArithmetic, ModularArithmetic, OrdArithmetic,
        StdArithmetic, WrappingArithmetic,
    },
    env::{Assertions, Comparisons, Prelude},
    fns, Environment, Number, Object, Value,
};
use arithmetic_parser::grammars::NumLiteral;
use arithmetic_typing::{
    arith::Num as NumType, defs, Function, Object as ObjectType, Type, TypeEnvironment,
};
use num_complex::{Complex, Complex32, Complex64};
use num_traits::{CheckedRem, Num, WrappingNeg};

fn binary_fn() -> Type {
    Function::builder()
        .with_arg(Type::NUM)
        .with_arg(Type::NUM)
        .returning(Type::NUM)
        .into()
}

fn comparison_type_defs() -> Vec<(&'static str, Type)> {
    // TODO: imprecise typing!
    vec![
        ("LESS", Type::NUM),
        ("EQUAL", Type::NUM),
        ("GREATER", Type::NUM),
        ("cmp", binary_fn()),
        ("min", binary_fn()),
        ("max", binary_fn()),
    ]
}

#[derive(Debug, Clone, Copy)]
#[allow(clippy::type_complexity)] // not that complex, really
pub struct StdLibrary<T: 'static> {
    constants: &'static [(&'static str, T)],
    unary: &'static [(&'static str, fn(T) -> T)],
    binary: &'static [(&'static str, fn(T, T) -> T)],
}

impl<T: ReplLiteral> StdLibrary<T> {
    fn variables(&self) -> impl Iterator<Item = (&'static str, Value<T>)> {
        let constants = self
            .constants
            .iter()
            .map(|(name, constant)| (*name, Value::Prim(constant.to_owned())));
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

    fn num_object(&self) -> Object<T> {
        let num_object_entries = self.variables().filter_map(|(name, value)| {
            if value.is_function() {
                Some((name, value))
            } else {
                None
            }
        });
        num_object_entries.collect()
    }

    fn type_defs(&self) -> impl Iterator<Item = (&'static str, Type)> {
        let unary_fn = Function::builder().with_arg(Type::NUM).returning(Type::NUM);
        let unary_fn = Type::from(unary_fn);

        let constants = self.constants.iter().map(|(name, _)| (*name, Type::NUM));
        let unary_fns = self
            .unary
            .iter()
            .map(move |(name, _)| (*name, unary_fn.clone()));
        let binary_fns = self.binary.iter().map(|(name, _)| (*name, binary_fn()));

        let num_proto: ObjectType<NumType> = unary_fns.clone().chain(binary_fns.clone()).collect();
        constants
            .chain(unary_fns)
            .chain(binary_fns)
            .chain(iter::once(("Num", num_proto.into())))
    }
}

pub trait ReplLiteral: NumLiteral + Num + Number + PartialEq + fmt::Display {
    const STD_LIB: StdLibrary<Self>;
}

macro_rules! declare_int_functions {
    ($type:ident) => {
        impl ReplLiteral for $type {
            const STD_LIB: StdLibrary<$type> = StdLibrary {
                constants: &[("MIN_VALUE", $type::MIN), ("MAX_VALUE", $type::MAX)],

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

fn int_arithmetic<T>(wrapping: bool) -> Box<dyn OrdArithmetic<T>>
where
    WrappingArithmetic: OrdArithmetic<T>,
    CheckedArithmetic: OrdArithmetic<T>,
{
    if wrapping {
        Box::new(WrappingArithmetic)
    } else {
        Box::new(CheckedArithmetic::new())
    }
}

pub fn create_int_env<T>(wrapping: bool) -> (Environment<T>, TypeEnvironment)
where
    T: ReplLiteral + ops::Rem + WrappingNeg + CheckedRem,
    WrappingArithmetic: OrdArithmetic<T>,
    CheckedArithmetic: OrdArithmetic<T>,
{
    const REM_ERROR_MSG: &str = "Cannot calculate remainder for a divisor of zero";

    let mut env = Environment::<T>::with_arithmetic(int_arithmetic(wrapping));
    let vars = Prelude::iter()
        .chain(Assertions::iter())
        .chain(Comparisons::iter())
        .chain(T::STD_LIB.variables());
    env.extend(vars);

    let rem = if wrapping {
        Value::wrapped_fn(|x: T, y: T| {
            if y == T::zero() {
                Err(REM_ERROR_MSG.to_owned())
            } else if y.wrapping_neg().is_one() {
                // Prevent a panic with `T::min_value() % -1`.
                Ok(T::zero())
            } else {
                Ok(x % y)
            }
        })
    } else {
        Value::wrapped_fn(|x: T, y: T| x.checked_rem(&y).ok_or_else(|| REM_ERROR_MSG.to_owned()))
    };

    env.insert_native_fn("array", fns::Array)
        .insert("rem", rem)
        .insert("Num", T::STD_LIB.num_object().into());

    let type_env = defs::Prelude::iter()
        .chain(defs::Assertions::iter())
        .chain(comparison_type_defs())
        .chain(T::STD_LIB.type_defs())
        .chain([
            ("rem", binary_fn()),
            ("array", defs::Prelude::array(NumType::Num).into()),
        ])
        .collect();

    (env, type_env)
}

pub fn create_modular_env(modulus: u64) -> (Environment<u64>, TypeEnvironment) {
    let arith = ModularArithmetic::new(modulus).without_comparisons();
    let mut env = Environment::<u64>::with_arithmetic(arith);
    env.extend(Prelude::iter().chain(Assertions::iter()));
    env.insert("MAX_VALUE", Value::Prim(modulus - 1));

    let type_env = defs::Prelude::iter()
        .chain(defs::Assertions::iter())
        .chain([("MAX_VALUE", Type::NUM)])
        .collect();

    (env, type_env)
}

macro_rules! declare_real_functions {
    ($type:ident) => {
        impl ReplLiteral for $type {
            const STD_LIB: StdLibrary<$type> = StdLibrary {
                constants: &[
                    ("INF", $type::INFINITY),
                    ("E", std::$type::consts::E),
                    ("PI", std::$type::consts::PI),
                ],

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
                    // Misc functions.
                    ("sqrt", $type::sqrt),
                    ("cbrt", $type::cbrt),
                ],

                binary: &[],
            };
        }
    };
}

declare_real_functions!(f32);
declare_real_functions!(f64);

pub fn create_float_env<T>(tolerance: T) -> (Environment<T>, TypeEnvironment)
where
    T: ReplLiteral,
    StdArithmetic: OrdArithmetic<T>,
{
    let mut env = Environment::<T>::new();
    let vars = Prelude::iter()
        .chain(Assertions::iter())
        .chain(Comparisons::iter())
        .chain(T::STD_LIB.variables());
    env.extend(vars);

    env.insert_native_fn("array", fns::Array)
        .insert_native_fn("assert_close", fns::AssertClose::new(tolerance))
        .insert("Num", T::STD_LIB.num_object().into());

    let type_env = defs::Prelude::iter()
        .chain(defs::Assertions::iter())
        .chain(comparison_type_defs())
        .chain(T::STD_LIB.type_defs())
        .chain([
            ("array", defs::Prelude::array(NumType::Num).into()),
            (
                "assert_close",
                defs::Assertions::assert_close(NumType::Num).into(),
            ),
        ])
        .collect();
    (env, type_env)
}

macro_rules! declare_complex_functions {
    ($type:ident, $real:ident) => {
        impl ReplLiteral for $type {
            const STD_LIB: StdLibrary<$type> = StdLibrary {
                constants: &[
                    ("INF", Complex::new($real::INFINITY, 0.0)),
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

pub fn create_complex_env<T>() -> (Environment<T>, TypeEnvironment)
where
    T: ReplLiteral,
    StdArithmetic: Arithmetic<T>,
{
    let arith = StdArithmetic.without_comparisons();
    let mut env = Environment::<T>::with_arithmetic(arith);
    let vars = Prelude::iter()
        .chain(Assertions::iter())
        .chain(T::STD_LIB.variables())
        .chain([("Num", T::STD_LIB.num_object().into())]);
    env.extend(vars);

    let type_env = defs::Prelude::iter()
        .chain(defs::Assertions::iter())
        .chain(T::STD_LIB.type_defs())
        .collect();

    (env, type_env)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use arithmetic_typing::arith::Num;
    use assert_matches::assert_matches;

    use super::*;

    #[test]
    fn environments_are_consistent_for_ints() {
        let (env, type_env) = create_int_env::<i64>(true);
        assert_same_values(&env, &type_env);
    }

    fn assert_same_values<T>(env: &Environment<T>, type_env: &TypeEnvironment) {
        for (name, value) in env {
            let value_type = &type_env[name];
            match value {
                Value::Prim(_) => assert_matches!(value_type, Type::Prim(Num::Num)),
                Value::Bool(_) => assert_matches!(value_type, Type::Prim(Num::Bool)),
                Value::Tuple(_) => assert_matches!(value_type, Type::Tuple(_)),
                Value::Object(_) => assert_matches!(value_type, Type::Object(_)),
                Value::Function(_) => assert_matches!(value_type, Type::Function(_)),
                _ => { /* no check */ }
            }
        }
        let all_names: HashSet<_> = env.iter().map(|(name, _)| name).collect();
        let all_type_names: HashSet<_> = type_env.iter().map(|(name, _)| name).collect();
        assert_eq!(all_names, all_type_names);
    }

    #[test]
    fn environments_are_consistent_for_modular_arithmetic() {
        let (env, type_env) = create_modular_env(17);
        assert_same_values(&env, &type_env);
    }

    #[test]
    fn environments_are_consistent_for_floats() {
        let (env, type_env) = create_float_env::<f64>(1e-5);
        assert_same_values(&env, &type_env);
    }

    #[test]
    fn environments_are_consistent_for_complex_numbers() {
        let (env, type_env) = create_complex_env::<Complex64>();
        assert_same_values(&env, &type_env);
    }
}
