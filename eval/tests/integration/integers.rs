//! Tests for integer arithmetics.

use assert_matches::assert_matches;
use num_traits::Bounded;

use arithmetic_eval::{
    arith::{
        ArithmeticExt, Checked, CheckedArithmetic, ModularArithmetic, OrdArithmetic,
        WrappingArithmetic,
    },
    env::{Assertions, Comparisons, Environment, Prelude},
    error::{ErrorKind, ErrorWithBacktrace},
    exec::WildcardId,
    ExecutableModule, Number, Value,
};
use arithmetic_parser::grammars::{NumGrammar, NumLiteral, Parse, Untyped};

fn try_evaluate<T>(env: &mut Environment<T>, program: &str) -> Result<Value<T>, ErrorWithBacktrace>
where
    T: NumLiteral,
{
    let block = Untyped::<NumGrammar<T>>::parse_statements(program).unwrap();
    let module = ExecutableModule::new(WildcardId, &block).unwrap();
    module.with_env(env).unwrap().run()
}

fn evaluate<T>(env: &mut Environment<T>, program: &str) -> Value<T>
where
    T: NumLiteral,
{
    try_evaluate(env, program).unwrap()
}

fn test_arithmetic_base<T, A>(arithmetic: A)
where
    T: PartialEq + PartialOrd + From<u8> + NumLiteral + Number,
    A: OrdArithmetic<T> + Copy + 'static,
{
    let value = evaluate::<T>(
        &mut Environment::with_arithmetic(arithmetic),
        "2 * 2 + 1 - 2",
    );
    assert_eq!(value, Value::Prim(3_u8.into()));

    let program_with_tuples = "(1, 2, 3) * 3 - 1";
    let value = evaluate::<T>(
        &mut Environment::with_arithmetic(arithmetic),
        program_with_tuples,
    );
    let expected_numbers = [2_u8, 5, 8]
        .into_iter()
        .map(|num| Value::Prim(num.into()))
        .collect();
    assert_eq!(value, Value::Tuple(expected_numbers));

    let program_with_comparisons = "1 < 2 && 1 + 1 == 2 && 2 ^ 3 > 7";
    let mut env = Environment::with_arithmetic(arithmetic);
    env.extend(Comparisons::iter());
    let value = evaluate::<T>(&mut env, program_with_comparisons);
    assert_eq!(value, Value::Bool(true));

    let program_using_std = "(1, 2, 3).all(|x| x > 0) && (2, 3, 4).{Array.any}(|x| x == 3)";
    let mut env = Environment::with_arithmetic(arithmetic);
    env.extend(Prelude::iter().chain(Comparisons::iter()));
    let value = evaluate::<T>(&mut env, program_using_std);
    assert_eq!(value, Value::Bool(true));
}

fn test_unsigned_checked_arithmetic<T, Kind>()
where
    T: PartialEq + PartialOrd + From<u8> + NumLiteral + Number,
    CheckedArithmetic<Kind>: OrdArithmetic<T> + 'static,
{
    let arithmetic = CheckedArithmetic::<Kind>::new();
    test_arithmetic_base::<T, _>(arithmetic);

    let program = "1 - 2 + 5";
    let err =
        try_evaluate::<T>(&mut Environment::with_arithmetic(arithmetic), program).unwrap_err();
    let err_kind = err.source().kind();
    assert_matches!(
        err_kind,
        ErrorKind::Arithmetic(err) if err.to_string().contains("integer overflow")
    );
    let err_span = err.source().location().in_module();
    assert_eq!(err_span.span(program), "1 - 2");
}

#[test]
fn checked_u8_arithmetic() {
    test_unsigned_checked_arithmetic::<u8, Checked>();
}

#[test]
fn checked_u16_arithmetic() {
    test_unsigned_checked_arithmetic::<u16, Checked>();
}

#[test]
fn checked_u32_arithmetic() {
    test_unsigned_checked_arithmetic::<u32, Checked>();
}

#[test]
fn checked_u64_arithmetic() {
    test_unsigned_checked_arithmetic::<u64, Checked>();
}

#[test]
fn checked_u128_arithmetic() {
    test_unsigned_checked_arithmetic::<u128, Checked>();
}

#[cfg(feature = "bigint")]
#[test]
fn checked_unsigned_bigint_arithmetic() {
    use arithmetic_eval::arith::NegateOnlyZero;
    use num_bigint::BigUint;

    test_unsigned_checked_arithmetic::<BigUint, NegateOnlyZero>();
}

fn test_signed_checked_arithmetic<T, Kind>()
where
    T: PartialEq + PartialOrd + From<u8> + NumLiteral + Number,
    CheckedArithmetic<Kind>: OrdArithmetic<T> + 'static,
{
    let arithmetic = CheckedArithmetic::<Kind>::new();
    test_arithmetic_base::<T, _>(arithmetic);

    let value = evaluate::<T>(&mut Environment::with_arithmetic(arithmetic), "1 - 2 + 5");
    assert_eq!(value, Value::Prim(4_u8.into()));

    let program = "-2 / 0 + 1";
    let err =
        try_evaluate::<T>(&mut Environment::with_arithmetic(arithmetic), program).unwrap_err();
    let err_kind = err.source().kind();
    assert_matches!(
        err_kind,
        ErrorKind::Arithmetic(err) if err.to_string().contains("division by zero")
    );
    let err_span = err.source().location().in_module();
    assert_eq!(err_span.span(program), "-2 / 0");

    let program = "2 ^ -3 + 1";
    let err =
        try_evaluate::<T>(&mut Environment::with_arithmetic(arithmetic), program).unwrap_err();
    let err_kind = err.source().kind();
    assert_matches!(
        err_kind,
        ErrorKind::Arithmetic(err) if err.to_string().contains("exponent is too large or negative")
    );
    let err_span = err.source().location().in_module();
    assert_eq!(err_span.span(program), "2 ^ -3");
}

#[test]
fn checked_i16_arithmetic() {
    test_signed_checked_arithmetic::<i16, Checked>();
}

#[test]
fn checked_i32_arithmetic() {
    test_signed_checked_arithmetic::<i32, Checked>();
}

#[test]
fn checked_i64_arithmetic() {
    test_signed_checked_arithmetic::<i64, Checked>();
}

#[test]
fn checked_i128_arithmetic() {
    test_signed_checked_arithmetic::<i128, Checked>();
}

#[cfg(feature = "bigint")]
#[test]
fn checked_signed_bigint_arithmetic() {
    use arithmetic_eval::arith::Unchecked;
    use num_bigint::BigInt;

    test_signed_checked_arithmetic::<BigInt, Unchecked>();
}

fn test_wrapping_unsigned_arithmetic<T>()
where
    T: PartialEq + PartialOrd + From<u8> + Bounded + NumLiteral + Number,
    WrappingArithmetic: OrdArithmetic<T>,
{
    let arithmetic = WrappingArithmetic;
    test_arithmetic_base::<T, _>(arithmetic);

    let value = evaluate::<T>(&mut Environment::with_arithmetic(arithmetic), "1 - 2 + 5");
    assert_eq!(value, Value::Prim(4_u8.into()));
    let value = evaluate::<T>(&mut Environment::with_arithmetic(arithmetic), "-1");
    assert_eq!(value, Value::Prim(T::max_value()));
}

#[test]
fn wrapping_u8_arithmetic() {
    test_wrapping_unsigned_arithmetic::<u8>();
}

#[test]
fn wrapping_u16_arithmetic() {
    test_wrapping_unsigned_arithmetic::<u16>();
}

#[test]
fn wrapping_u32_arithmetic() {
    test_wrapping_unsigned_arithmetic::<u32>();
}

#[test]
fn wrapping_u64_arithmetic() {
    test_wrapping_unsigned_arithmetic::<u64>();
}

#[test]
fn wrapping_u128_arithmetic() {
    test_wrapping_unsigned_arithmetic::<u128>();
}

#[test]
fn modular_arithmetic() {
    // While comparisons with modular arithmetic don't make much sense, they can be technically
    // implemented.
    let arithmetic = ModularArithmetic::new(61).with_natural_comparison();
    test_arithmetic_base::<u32, _>(arithmetic);

    let modular_eq_program = "-1 == 60 && 1111111 == 57 && 3 / 5 == 25 && 60^60 == 1";
    let value = evaluate(
        &mut Environment::with_arithmetic(arithmetic),
        modular_eq_program,
    );
    assert_eq!(value, Value::Bool(true));

    let fermat_theorem_check = "while(1, |i| i != 0, |i| { assert_eq(i^60, 1); i + 1 })";
    let mut env = Environment::with_arithmetic(arithmetic);
    env.extend(Prelude::iter().chain(Assertions::iter()));
    evaluate(&mut env, fermat_theorem_check);
}
