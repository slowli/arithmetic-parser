//! Standard grammars.

use nom::{
    bytes::complete::take_while_m_n,
    combinator::{not, peek},
    number::complete::{double, float},
    sequence::terminated,
    Slice,
};
use num_traits::Num;

use core::{f32, f64, fmt, marker::PhantomData};

use crate::{spans::NomResult, Features, Grammar, InputSpan};

// FIXME: `1.foo()` does not work because float parser consumes `1.`

/// Single-type numeric grammar parameterized by the literal type.
#[derive(Debug)]
pub struct NumGrammar<T>(PhantomData<T>);

/// Type alias for a grammar on `f32` literals.
pub type F32Grammar = NumGrammar<f32>;
/// Type alias for a grammar on `f64` literals.
pub type F64Grammar = NumGrammar<f64>;

impl<T: NumLiteral> Grammar for NumGrammar<T> {
    type Lit = T;
    type Type = ();

    const FEATURES: Features = Features {
        type_annotations: false,
        ..Features::all()
    };

    fn parse_literal(input: InputSpan<'_>) -> NomResult<'_, Self::Lit> {
        T::parse(input)
    }

    fn parse_type(_input: InputSpan<'_>) -> NomResult<'_, Self::Type> {
        unimplemented!()
    }
}

/// Numeric literal used in `NumGrammar`s.
pub trait NumLiteral: 'static + Copy + Num + fmt::Debug {
    /// Tries to parse a literal.
    fn parse(input: InputSpan<'_>) -> NomResult<'_, Self>;
}

/// Ensures that the child parser does not consume a part of a larger expression by rejecting
/// if the part following the input is an alphanumeric char or `_`.
///
/// For example, `float` parses `-Inf`, which can lead to parser failure if it's a part of
/// a larger expression (e.g., `-Infer(2, 3)`).
pub fn ensure_no_overlap<'a, T>(
    parser: impl Fn(InputSpan<'a>) -> NomResult<'a, T>,
) -> impl Fn(InputSpan<'a>) -> NomResult<'a, T> {
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
        num_parser: impl Fn(InputSpan<'a>) -> NomResult<'a, T>,
    ) -> impl Fn(InputSpan<'a>) -> NomResult<'a, Complex<T>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Expr, GrammarExt, UnaryOp};

    use assert_matches::assert_matches;
    use core::f32::INFINITY;

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

    #[test]
    fn parsing_infinity() {
        let parsed = F32Grammar::parse_statements("Inf").unwrap();
        let ret = parsed.return_value.unwrap().extra;
        assert_matches!(ret, Expr::Literal(lit) if lit == INFINITY);

        let parsed = F32Grammar::parse_statements("-Inf").unwrap();
        let ret = parsed.return_value.unwrap().extra;
        assert_matches!(ret, Expr::Literal(lit) if lit == -INFINITY);

        let parsed = F32Grammar::parse_statements("Infty").unwrap();
        let ret = parsed.return_value.unwrap().extra;
        assert_matches!(ret, Expr::Variable);

        let parsed = F32Grammar::parse_statements("Infer(1)").unwrap();
        let ret = parsed.return_value.unwrap().extra;
        assert_matches!(ret, Expr::Function { .. });

        let parsed = F32Grammar::parse_statements("-Infty").unwrap();
        let ret = parsed.return_value.unwrap().extra;
        assert_matches!(ret, Expr::Unary { .. });

        let parsed = F32Grammar::parse_statements("-Infer(2, 3)").unwrap();
        let ret = parsed.return_value.unwrap().extra;
        assert_matches!(ret, Expr::Unary { .. });
    }

    #[cfg(feature = "num-complex")]
    #[test]
    fn parsing_i() {
        use num_complex::Complex32;

        let parsed = NumGrammar::<Complex32>::parse_statements("i").unwrap();
        let ret = parsed.return_value.unwrap().extra;
        assert_matches!(ret, Expr::Literal(lit) if lit == Complex32::i());

        let parsed = NumGrammar::<Complex32>::parse_statements("i + 5").unwrap();
        let ret = parsed.return_value.unwrap().extra;
        let i_as_lhs = &ret.binary_lhs().unwrap().extra;
        assert_matches!(*i_as_lhs, Expr::Literal(lit) if lit == Complex32::i());

        let parsed = NumGrammar::<Complex32>::parse_statements("5 - i").unwrap();
        let ret = parsed.return_value.unwrap().extra;
        let i_as_rhs = &ret.binary_rhs().unwrap().extra;
        assert_matches!(*i_as_rhs, Expr::Literal(lit) if lit == Complex32::i());

        // `i` should not be parsed as a literal if it's a part of larger expression.
        let parsed = NumGrammar::<Complex32>::parse_statements("ix + 5").unwrap();
        let ret = parsed.return_value.unwrap().extra;
        let variable = &ret.binary_lhs().unwrap().extra;
        assert_matches!(*variable, Expr::Variable);

        let parsed = NumGrammar::<Complex32>::parse_statements("-i + 5").unwrap();
        let ret = parsed.return_value.unwrap().extra;
        let negation_expr = &ret.binary_lhs().unwrap().extra;
        let inner_lhs = match negation_expr {
            Expr::Unary { inner, op } if op.extra == UnaryOp::Neg => &inner.extra,
            _ => panic!("Unexpected LHS: {:?}", negation_expr),
        };
        assert_matches!(inner_lhs, Expr::Literal(lit) if *lit == Complex32::i());

        let parsed = NumGrammar::<Complex32>::parse_statements("-ix + 5").unwrap();
        let ret = parsed.return_value.unwrap().extra;
        let var_negation = &ret.binary_lhs().unwrap().extra;
        let negated_var = match var_negation {
            Expr::Unary { inner, op } if op.extra == UnaryOp::Neg => &inner.extra,
            _ => panic!("Unexpected LHS: {:?}", var_negation),
        };
        assert_matches!(negated_var, Expr::Variable);
    }
}
