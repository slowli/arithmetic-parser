//! A somewhat contrived arithmetic that parses string literals and only allows to add them
//! and compare strings.

use std::{fmt, str::FromStr};

use arithmetic_parser::{
    grammars::{Parse, ParseLiteral, Typed},
    BinaryOp, InputSpan, NomResult,
};
use arithmetic_typing::{
    arith::*,
    error::{ErrorLocation, OpErrors},
    Annotated, Assertions, PrimitiveType, Substitutions, Type, TypeEnvironment,
};

/// Primitive type: string or boolean.
#[derive(Debug, Clone, Copy, PartialEq)]
enum StrType {
    Str,
    Bool,
}

impl fmt::Display for StrType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Str => "Str",
            Self::Bool => "Bool",
        })
    }
}

impl FromStr for StrType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Str" => Ok(Self::Str),
            "Bool" => Ok(Self::Bool),
            _ => Err(anyhow::anyhow!("Expected `Str` or `Bool`")),
        }
    }
}

impl PrimitiveType for StrType {
    type Constraints = NumConstraints;
}

impl WithBoolean for StrType {
    const BOOL: Self = Self::Bool;
}

impl LinearType for StrType {
    fn is_linear(&self) -> bool {
        matches!(self, Self::Str)
    }
}

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

#[derive(Debug, Clone, Copy)]
struct StrArithmetic;

impl MapPrimitiveType<String> for StrArithmetic {
    type Prim = StrType;

    fn type_of_literal(&self, _lit: &String) -> Self::Prim {
        StrType::Str
    }
}

impl TypeArithmetic<StrType> for StrArithmetic {
    fn process_unary_op<'a>(
        &self,
        substitutions: &mut Substitutions<StrType>,
        context: &UnaryOpContext<StrType>,
        errors: OpErrors<'a, StrType>,
    ) -> Type<StrType> {
        BoolArithmetic.process_unary_op(substitutions, context, errors)
    }

    fn process_binary_op<'a>(
        &self,
        substitutions: &mut Substitutions<StrType>,
        context: &BinaryOpContext<StrType>,
        mut errors: OpErrors<'a, StrType>,
    ) -> Type<StrType> {
        match context.op {
            BinaryOp::Add => NumArithmetic::unify_binary_op(
                substitutions,
                context,
                errors,
                NumConstraints::OP_SETTINGS,
            ),

            BinaryOp::Gt | BinaryOp::Lt | BinaryOp::Ge | BinaryOp::Le => {
                let lhs_ty = &context.lhs;
                let rhs_ty = &context.rhs;

                substitutions.unify(
                    &Type::Prim(StrType::Str),
                    lhs_ty,
                    errors.with_location(ErrorLocation::Lhs),
                );
                substitutions.unify(
                    &Type::Prim(StrType::Str),
                    rhs_ty,
                    errors.with_location(ErrorLocation::Rhs),
                );
                Type::BOOL
            }

            _ => BoolArithmetic.process_binary_op(substitutions, context, errors),
        }
    }
}

type Parser = Typed<Annotated<StrGrammar>>;

fn main() -> anyhow::Result<()> {
    let code = r#"
        x = "foo" + "bar";
        // Spreading logic is reused from `NumArithmetic` and just works.
        y = "foo" + ("bar", "quux");
        // Boolean logic works as well.
        assert("bar" != "baz");
        assert("foo" > "bar" && "foo" <= "quux");
    "#;
    let ast = Parser::parse_statements(code)?;

    let mut env = TypeEnvironment::<StrType>::new();
    env.insert("assert", Assertions::Assert);
    env.process_with_arithmetic(&StrArithmetic, &ast)?;
    assert_eq!(env["x"], Type::Prim(StrType::Str));
    assert_eq!(env["y"].to_string(), "(Str, Str)");

    let bogus_code = r#""foo" - "bar""#;
    let bogus_ast = Parser::parse_statements(bogus_code)?;
    let err = env
        .process_with_arithmetic(&StrArithmetic, &bogus_ast)
        .unwrap_err();
    assert_eq!(err.to_string(), "1:1: Unsupported binary op: subtraction");

    Ok(())
}
