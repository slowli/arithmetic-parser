//! Grammar functionality and a collection of standard grammars.
//!
//! # Defining grammars
//!
//! To define a [`Grammar`], you'll need a [`ParseLiteral`] implementation, which defines
//! how literals are parsed (numbers, strings, chars, hex- / base64-encoded byte sequences, etc.).
//! There are standard impls for floating-point number parsing and the complex numbers
//! (if the relevant feature is on).
//!
//! You may define how to parse type annotations by implementing `Grammar` explicitly.
//! Alternatively, if you don't need type annotations, a `Grammar` can be obtained from
//! a [`ParseLiteral`] impl by wrapping it into [`Untyped`].
//!
//! Once you have a `Grammar`, you can supply it as a `Base` for [`Parse`]. `Parse` methods
//! allow to parse complete or streaming [`Block`](crate::Block)s of statements.
//! Note that `Untyped` and [`Typed`] wrappers allow to avoid an explicit `Parse` impl.
//!
//! See [`ParseLiteral`], [`Grammar`] and [`Parse`] docs for the examples of various grammar
//! definitions.

use nom::{
    bytes::complete::take_while_m_n,
    character::complete::{char as tag_char, digit1},
    combinator::{map_res, not, opt, peek, recognize},
    number::complete::{double, float},
    sequence::{terminated, tuple},
    Slice,
};

use core::{f32, f64, fmt, marker::PhantomData};

mod traits;

pub use self::traits::{
    BooleanOps, Features, Grammar, IntoInputSpan, Parse, ParseLiteral, Typed, Untyped,
};

use crate::{spans::NomResult, ErrorKind, InputSpan};

/// Single-type numeric grammar parameterized by the literal type.
#[derive(Debug)]
pub struct NumGrammar<T>(PhantomData<T>);

/// Type alias for a grammar on `f32` literals.
pub type F32Grammar = NumGrammar<f32>;
/// Type alias for a grammar on `f64` literals.
pub type F64Grammar = NumGrammar<f64>;

impl<T: NumLiteral> ParseLiteral for NumGrammar<T> {
    type Lit = T;

    fn parse_literal(input: InputSpan<'_>) -> NomResult<'_, Self::Lit> {
        T::parse(input)
    }
}

/// Numeric literal used in `NumGrammar`s.
pub trait NumLiteral: 'static + Clone + fmt::Debug {
    /// Tries to parse a literal.
    fn parse(input: InputSpan<'_>) -> NomResult<'_, Self>;
}

/// Ensures that the child parser does not consume a part of a larger expression by rejecting
/// if the part following the input is an alphanumeric char or `_`.
///
/// For example, `float` parses `-Inf`, which can lead to parser failure if it's a part of
/// a larger expression (e.g., `-Infer(2, 3)`).
pub fn ensure_no_overlap<'a, T>(
    mut parser: impl FnMut(InputSpan<'a>) -> NomResult<'a, T>,
) -> impl FnMut(InputSpan<'a>) -> NomResult<'a, T> {
    let truncating_parser = move |input| {
        parser(input).map(|(rest, number)| (maybe_truncate_consumed_input(input, rest), number))
    };

    terminated(
        truncating_parser,
        peek(not(take_while_m_n(1, 1, |c: char| {
            c.is_ascii_alphabetic() || c == '_'
        }))),
    )
}

fn can_start_a_var_name(byte: u8) -> bool {
    byte == b'_' || byte.is_ascii_alphabetic()
}

fn maybe_truncate_consumed_input<'a>(input: InputSpan<'a>, rest: InputSpan<'a>) -> InputSpan<'a> {
    let relative_offset = rest.location_offset() - input.location_offset();
    debug_assert!(relative_offset > 0, "num parser succeeded for empty string");
    let last_consumed_byte_index = relative_offset - 1;

    let input_fragment = *input.fragment();
    let input_as_bytes = input_fragment.as_bytes();
    if relative_offset < input_fragment.len()
        && input_fragment.is_char_boundary(last_consumed_byte_index)
        && input_as_bytes[last_consumed_byte_index] == b'.'
        && can_start_a_var_name(input_as_bytes[relative_offset])
    {
        // The last char consumed by the parser is '.' and the next part looks like
        // a method call. Shift the `rest` boundary to include '.'.
        input.slice(last_consumed_byte_index..)
    } else {
        rest
    }
}

macro_rules! impl_num_literal_for_uint {
    ($($num:ident),+) => {
        $(
        impl NumLiteral for $num {
            fn parse(input: InputSpan<'_>) -> NomResult<'_, Self> {
                let parser = |s: InputSpan<'_>| s.fragment().parse().map_err(ErrorKind::literal);
                map_res(digit1, parser)(input)
            }
        }
        )+
    };
}

impl_num_literal_for_uint!(u8, u16, u32, u64, u128);

macro_rules! impl_num_literal_for_int {
    ($($num:ident),+) => {
        $(
        impl NumLiteral for $num {
            fn parse(input: InputSpan<'_>) -> NomResult<'_, Self> {
                let parser = |s: InputSpan<'_>| s.fragment().parse().map_err(ErrorKind::literal);
                map_res(recognize(tuple((opt(tag_char('-')), digit1))), parser)(input)
            }
        }
        )+
    };
}

impl_num_literal_for_int!(i8, i16, i32, i64, i128);

impl NumLiteral for f32 {
    fn parse(input: InputSpan<'_>) -> NomResult<'_, Self> {
        ensure_no_overlap(float)(input)
    }
}

impl NumLiteral for f64 {
    fn parse(input: InputSpan<'_>) -> NomResult<'_, Self> {
        ensure_no_overlap(double)(input)
    }
}

#[cfg(feature = "num-complex")]
mod complex {
    use nom::{
        branch::alt,
        character::complete::one_of,
        combinator::{map, opt},
        number::complete::{double, float},
        sequence::tuple,
    };
    use num_complex::Complex;
    use num_traits::Num;

    use super::{ensure_no_overlap, NumLiteral};
    use crate::{InputSpan, NomResult};

    fn complex_parser<'a, T: Num>(
        num_parser: impl FnMut(InputSpan<'a>) -> NomResult<'a, T>,
    ) -> impl FnMut(InputSpan<'a>) -> NomResult<'a, Complex<T>> {
        let i_parser = map(one_of("ij"), |_| Complex::new(T::zero(), T::one()));

        let parser = tuple((num_parser, opt(one_of("ij"))));
        let parser = map(parser, |(value, maybe_imag)| {
            if maybe_imag.is_some() {
                Complex::new(T::zero(), value)
            } else {
                Complex::new(value, T::zero())
            }
        });

        alt((i_parser, parser))
    }

    impl NumLiteral for num_complex::Complex32 {
        fn parse(input: InputSpan<'_>) -> NomResult<'_, Self> {
            ensure_no_overlap(complex_parser(float))(input)
        }
    }

    impl NumLiteral for num_complex::Complex64 {
        fn parse(input: InputSpan<'_>) -> NomResult<'_, Self> {
            ensure_no_overlap(complex_parser(double))(input)
        }
    }
}

#[cfg(feature = "num-bigint")]
mod bigint {
    use nom::{
        character::complete::{char as tag_char, digit1},
        combinator::{map_res, opt, recognize},
        sequence::tuple,
    };
    use num_bigint::{BigInt, BigUint};
    use num_traits::Num;

    use super::NumLiteral;
    use crate::{ErrorKind, InputSpan, NomResult};

    impl NumLiteral for BigInt {
        fn parse(input: InputSpan<'_>) -> NomResult<'_, Self> {
            let parser = |s: InputSpan<'_>| {
                BigInt::from_str_radix(s.fragment(), 10).map_err(ErrorKind::literal)
            };
            map_res(recognize(tuple((opt(tag_char('-')), digit1))), parser)(input)
        }
    }

    impl NumLiteral for BigUint {
        fn parse(input: InputSpan<'_>) -> NomResult<'_, Self> {
            let parser = |s: InputSpan<'_>| {
                BigUint::from_str_radix(s.fragment(), 10).map_err(ErrorKind::literal)
            };
            map_res(digit1, parser)(input)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Expr, UnaryOp};

    use assert_matches::assert_matches;
    use core::f32::INFINITY;
    use nom::Err as NomErr;

    #[test]
    fn parsing_numbers_with_dot() {
        #[derive(Debug, Clone, Copy)]
        struct Sample {
            input: &'static str,
            consumed: usize,
            value: f32,
        }

        #[rustfmt::skip]
        const SAMPLES: &[Sample] = &[
            Sample { input: "1.25+3", consumed: 4, value: 1.25 },

            // Cases in which '.' should be consumed.
            Sample { input: "1.", consumed: 2, value: 1.0 },
            Sample { input: "-1.", consumed: 3, value: -1.0 },
            Sample { input: "1. + 2.", consumed: 2, value: 1.0 },
            Sample { input: "1.+2.", consumed: 2, value: 1.0 },
            Sample { input: "1. .sin()", consumed: 2, value: 1.0 },

            // Cases in which '.' should not be consumed.
            Sample { input: "1.sin()", consumed: 1, value: 1.0 },
            Sample { input: "-3.sin()", consumed: 2, value: -3.0 },
            Sample { input: "-3.5.sin()", consumed: 4, value: -3.5 },
        ];

        for &sample in SAMPLES {
            let (rest, number) = <f32 as NumLiteral>::parse(InputSpan::new(sample.input)).unwrap();
            assert!(
                (number - sample.value).abs() < f32::EPSILON,
                "Failed sample: {:?}",
                sample
            );
            assert_eq!(
                rest.location_offset(),
                sample.consumed,
                "Failed sample: {:?}",
                sample
            );
        }
    }

    #[cfg(feature = "std")]
    // ^-- This behavior is specific to `lexical-core` dependency, which is switched on with `std`.
    #[test]
    fn parsing_infinity() {
        let parsed = Untyped::<F32Grammar>::parse_statements("Inf").unwrap();
        let ret = parsed.return_value.unwrap().extra;
        assert_matches!(ret, Expr::Literal(lit) if lit == INFINITY);

        let parsed = Untyped::<F32Grammar>::parse_statements("-Inf").unwrap();
        let ret = parsed.return_value.unwrap().extra;
        assert_matches!(ret, Expr::Literal(lit) if lit == -INFINITY);

        let parsed = Untyped::<F32Grammar>::parse_statements("Infty").unwrap();
        let ret = parsed.return_value.unwrap().extra;
        assert_matches!(ret, Expr::Variable);

        let parsed = Untyped::<F32Grammar>::parse_statements("Infer(1)").unwrap();
        let ret = parsed.return_value.unwrap().extra;
        assert_matches!(ret, Expr::Function { .. });

        let parsed = Untyped::<F32Grammar>::parse_statements("-Infty").unwrap();
        let ret = parsed.return_value.unwrap().extra;
        assert_matches!(ret, Expr::Unary { .. });

        let parsed = Untyped::<F32Grammar>::parse_statements("-Infer(2, 3)").unwrap();
        let ret = parsed.return_value.unwrap().extra;
        assert_matches!(ret, Expr::Unary { .. });
    }

    #[cfg(feature = "num-complex")]
    #[test]
    fn parsing_i() {
        use num_complex::Complex32;

        type C32Grammar = Untyped<NumGrammar<Complex32>>;

        let parsed = C32Grammar::parse_statements("i").unwrap();
        let ret = parsed.return_value.unwrap().extra;
        assert_matches!(ret, Expr::Literal(lit) if lit == Complex32::i());

        let parsed = C32Grammar::parse_statements("i + 5").unwrap();
        let ret = parsed.return_value.unwrap().extra;
        let i_as_lhs = &ret.binary_lhs().unwrap().extra;
        assert_matches!(*i_as_lhs, Expr::Literal(lit) if lit == Complex32::i());

        let parsed = C32Grammar::parse_statements("5 - i").unwrap();
        let ret = parsed.return_value.unwrap().extra;
        let i_as_rhs = &ret.binary_rhs().unwrap().extra;
        assert_matches!(*i_as_rhs, Expr::Literal(lit) if lit == Complex32::i());

        // `i` should not be parsed as a literal if it's a part of larger expression.
        let parsed = C32Grammar::parse_statements("ix + 5").unwrap();
        let ret = parsed.return_value.unwrap().extra;
        let variable = &ret.binary_lhs().unwrap().extra;
        assert_matches!(*variable, Expr::Variable);

        let parsed = C32Grammar::parse_statements("-i + 5").unwrap();
        let ret = parsed.return_value.unwrap().extra;
        let negation_expr = &ret.binary_lhs().unwrap().extra;
        let inner_lhs = match negation_expr {
            Expr::Unary { inner, op } if op.extra == UnaryOp::Neg => &inner.extra,
            _ => panic!("Unexpected LHS: {:?}", negation_expr),
        };
        assert_matches!(inner_lhs, Expr::Literal(lit) if *lit == Complex32::i());

        let parsed = C32Grammar::parse_statements("-ix + 5").unwrap();
        let ret = parsed.return_value.unwrap().extra;
        let var_negation = &ret.binary_lhs().unwrap().extra;
        let negated_var = match var_negation {
            Expr::Unary { inner, op } if op.extra == UnaryOp::Neg => &inner.extra,
            _ => panic!("Unexpected LHS: {:?}", var_negation),
        };
        assert_matches!(negated_var, Expr::Variable);
    }

    #[test]
    fn uint_parsers() {
        let (_, u8_val) = <u8 as NumLiteral>::parse(InputSpan::new("3")).unwrap();
        assert_eq!(u8_val, 3);
        let (_, u16_val) = <u16 as NumLiteral>::parse(InputSpan::new("33333")).unwrap();
        assert_eq!(u16_val, 33_333);
        let (_, u32_val) = <u32 as NumLiteral>::parse(InputSpan::new("1111111111")).unwrap();
        assert_eq!(u32_val, 1_111_111_111);
        let (_, u64_val) =
            <u64 as NumLiteral>::parse(InputSpan::new(&u64::max_value().to_string())).unwrap();
        assert_eq!(u64_val, u64::max_value());
        let (_, u128_val) =
            <u128 as NumLiteral>::parse(InputSpan::new(&u128::max_value().to_string())).unwrap();
        assert_eq!(u128_val, u128::max_value());
    }

    #[test]
    fn int_parsers() {
        let (_, min_val) = <i8 as NumLiteral>::parse(InputSpan::new("-128")).unwrap();
        assert_eq!(min_val, -128);
        let (_, max_val) = <i8 as NumLiteral>::parse(InputSpan::new("127")).unwrap();
        assert_eq!(max_val, 127);

        let err = <i8 as NumLiteral>::parse(InputSpan::new("128")).unwrap_err();
        let err_kind = match &err {
            NomErr::Error(err) => err.kind(),
            _ => panic!("Unexpected error type: {:?}", err),
        };
        assert_matches!(err_kind, ErrorKind::Literal(_));
    }

    #[cfg(feature = "num-bigint")]
    #[test]
    fn bigint_parsers() {
        use num_bigint::{BigInt, BigUint};

        for len in 1..500 {
            let input = "1".repeat(len);
            let (_, value) = <BigUint as NumLiteral>::parse(InputSpan::new(&input)).unwrap();
            assert_eq!(value, BigUint::parse_bytes(input.as_bytes(), 10).unwrap());

            let (_, value) = <BigInt as NumLiteral>::parse(InputSpan::new(&input)).unwrap();
            let expected_value = BigInt::parse_bytes(input.as_bytes(), 10).unwrap();
            assert_eq!(value, expected_value);
            let (_, value) =
                <BigInt as NumLiteral>::parse(InputSpan::new(&format!("-{}", input))).unwrap();
            assert_eq!(value, -expected_value);
        }
    }
}
