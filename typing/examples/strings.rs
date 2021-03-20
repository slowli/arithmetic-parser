//! A somewhat contrived arithmetic that parses string literals and only allows to add them
//! and compare strings.

use arithmetic_parser::{
    grammars::{Grammar, Parse, ParseLiteral, Typed},
    BinaryOp, InputSpan, NomResult,
};

use arithmetic_typing::{
    arith::*, impl_singleton_literal_type, Assertions, MapLiteralType, NumConstraints,
    Substitutions, TypeEnvironment, TypeResult, ValueType,
};

/// Type of our literals: a string.
#[derive(Debug, Clone, Copy, PartialEq)]
struct StrType;

// FIXME: `NumConstraints` makes little sense here
impl_singleton_literal_type!(StrType, "Str", NumConstraints);

/// Grammar parsing strings as literals.
#[derive(Debug, Clone, Copy)]
struct StrGrammar;

impl ParseLiteral for StrGrammar {
    type Lit = String;

    /// Parses an ASCII string like `"Hello, world!"`.
    fn parse_literal(input: InputSpan<'_>) -> NomResult<'_, Self::Lit> {
        use nom::{
            branch::alt,
            bytes::complete::{escaped_transform, is_not},
            character::complete::char as tag_char,
            combinator::{cut, map, opt},
            sequence::{preceded, terminated},
        };

        let parser = escaped_transform(
            is_not("\\\"\n"),
            '\\',
            alt((
                map(tag_char('\\'), |_| "\\"),
                map(tag_char('"'), |_| "\""),
                map(tag_char('n'), |_| "\n"),
            )),
        );
        map(
            preceded(tag_char('"'), cut(terminated(opt(parser), tag_char('"')))),
            Option::unwrap_or_default,
        )(input)
    }
}

impl Grammar for StrGrammar {
    type Type = ValueType<StrType>;

    fn parse_type(input: InputSpan<'_>) -> NomResult<'_, Self::Type> {
        ValueType::parse(input)
    }
}

#[derive(Debug, Clone, Copy)]
struct StrArithmetic;

impl MapLiteralType<String> for StrArithmetic {
    type Lit = StrType;

    fn type_of_literal(&self, _lit: &String) -> Self::Lit {
        StrType
    }
}

impl TypeArithmetic<String> for StrArithmetic {
    fn process_unary_op<'a>(
        &self,
        substitutions: &mut Substitutions<Self::Lit>,
        spans: UnaryOpSpans<'a, Self::Lit>,
    ) -> TypeResult<'a, Self::Lit> {
        BoolArithmetic::process_unary_op(substitutions, &spans)
    }

    fn process_binary_op<'a>(
        &self,
        substitutions: &mut Substitutions<Self::Lit>,
        spans: BinaryOpSpans<'a, Self::Lit>,
    ) -> TypeResult<'a, Self::Lit> {
        let lhs_ty = &spans.lhs.extra;
        let rhs_ty = &spans.rhs.extra;
        match spans.op.extra {
            BinaryOp::Add => NumArithmetic::unify_binary_op(substitutions, lhs_ty, rhs_ty)
                .map_err(|err| err.with_span(&spans.total)),

            BinaryOp::Gt | BinaryOp::Lt | BinaryOp::Ge | BinaryOp::Le => {
                substitutions
                    .unify(&ValueType::Lit(StrType), lhs_ty)
                    .map_err(|err| err.with_span(&spans.lhs))?;
                substitutions
                    .unify(&ValueType::Lit(StrType), rhs_ty)
                    .map_err(|err| err.with_span(&spans.rhs))?;
                Ok(ValueType::Bool)
            }

            _ => BoolArithmetic::process_binary_op(substitutions, &spans),
        }
    }
}

fn main() -> anyhow::Result<()> {
    let code = r#"
        x = "foo" + "bar";
        // Spreading logic is reused from `NumArithmetic` and just works.
        y = "foo" + ("bar", "quux");
        // Boolean logic works as well.
        assert("bar" != "baz");
        assert("foo" > "bar" && "foo" <= "quux");
    "#;
    let ast = Typed::<StrGrammar>::parse_statements(code)?;

    let mut env = TypeEnvironment::<StrType>::new();
    env.insert_type("assert", Assertions::assert_type().into());
    env.process_with_arithmetic(&StrArithmetic, &ast)?;
    assert_eq!(env["x"], ValueType::Lit(StrType));
    assert_eq!(env["y"].to_string(), "(Str, Str)");

    let bogus_code = r#""foo" - "bar""#;
    let bogus_ast = Typed::<StrGrammar>::parse_statements(bogus_code)?;
    let err = env
        .process_with_arithmetic(&StrArithmetic, &bogus_ast)
        .unwrap_err();
    assert_eq!(err.to_string(), "1:6: Unsupported binary op: subtraction");

    Ok(())
}
