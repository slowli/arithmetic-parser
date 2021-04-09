//! Demonstrates how to use custom type constraints with custom primitive types.

use std::{fmt, ops, str::FromStr};

use arithmetic_parser::{
    grammars::{NumGrammar, NumLiteral, Parse, Typed},
    BinaryOp, InputSpan, NomResult,
};
use arithmetic_typing::{
    arith::*, Annotated, Prelude, PrimitiveType, Substitutions, TypeEnvironment, TypeErrorKind,
    TypeResult, ValueType,
};

/// Literal for arithmetic: either an integer or a byte buffer.
#[derive(Debug, Clone)]
enum NumOrBytes {
    /// Integer number, such as 1.
    Num(u32),
    /// Bytes represented in hex, such as `0xdeadbeef`.
    Bytes(Vec<u8>),
}

impl NumLiteral for NumOrBytes {
    fn parse(input: InputSpan<'_>) -> NomResult<'_, Self> {
        use arithmetic_parser::ErrorKind;
        use nom::{
            branch::alt,
            bytes::complete::{tag, take_while},
            combinator::{cut, map, map_res},
            sequence::preceded,
        };

        let parse_hex = preceded(
            tag("0x"),
            cut(map_res(
                take_while(|c: char| c.is_ascii_hexdigit()),
                |digits: InputSpan<'_>| hex::decode(digits.fragment()).map_err(ErrorKind::literal),
            )),
        );

        alt((map(parse_hex, Self::Bytes), map(u32::parse, Self::Num)))(input)
    }
}

/// Primitive types for `NumOrBytes`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NumOrBytesType {
    Num,
    Bytes,
    Bool,
}

impl fmt::Display for NumOrBytesType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Num => "Num",
            Self::Bytes => "Bytes",
            Self::Bool => "Bool",
        })
    }
}

impl FromStr for NumOrBytesType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Num" => Ok(Self::Num),
            "Bytes" => Ok(Self::Bytes),
            "Bool" => Ok(Self::Bool),
            _ => Err(anyhow::anyhow!("Expected `Num`, `Bytes` or `Bool`")),
        }
    }
}

impl WithBoolean for NumOrBytesType {
    const BOOL: Self = Self::Bool;
}

/// Constraints imposed on `ValueType<NumOrBytesType>`. Besides linearity,
/// we consider its weaker variant: ability to add values of type `T`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Constraints {
    /// No constraints.
    None,
    /// Type is summable (can be used as an argument for binary `+`).
    Summable,
    /// Type is linear (can be used as an argument for all arithmetic ops).
    Lin,
}

impl Default for Constraints {
    fn default() -> Self {
        Self::None
    }
}

impl fmt::Display for Constraints {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::None => "",
            Self::Lin => "Lin",
            Self::Summable => "Sum",
        })
    }
}

impl FromStr for Constraints {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Lin" => Ok(Self::Lin),
            "Sum" => Ok(Self::Summable),
            _ => Err(anyhow::anyhow!("Expected `Lin` or `Sum`")),
        }
    }
}

impl ops::BitOrAssign<&Self> for Constraints {
    fn bitor_assign(&mut self, rhs: &Constraints) {
        *self = match (*self, rhs) {
            (Self::Lin, _) | (_, Self::Lin) => Self::Lin,
            (Self::Summable, _) | (_, Self::Summable) => Self::Summable,
            _ => Self::None,
        };
    }
}

impl TypeConstraints<NumOrBytesType> for Constraints {
    // Constraints are applied recursively, similar to `LinConstraints`.
    fn apply(
        &self,
        ty: &ValueType<NumOrBytesType>,
        substitutions: &mut Substitutions<NumOrBytesType>,
    ) -> Result<(), TypeErrorKind<NumOrBytesType>> {
        if *self == Self::None {
            return Ok(());
        }

        let resolved_ty = if let ValueType::Var(var) = ty {
            substitutions.insert_constraints(var.index(), self);
            substitutions.fast_resolve(ty)
        } else {
            ty
        };

        match resolved_ty {
            // `Var`s are taken care of previously.
            ValueType::Var(_) | ValueType::Prim(NumOrBytesType::Num) => Ok(()),

            ValueType::Some => unreachable!(),

            ValueType::Prim(NumOrBytesType::Bool) | ValueType::Function(_) => Err(
                TypeErrorKind::failed_constraint(ty.to_owned(), self.to_owned()),
            ),

            // Bytes are summable, but not linear.
            ValueType::Prim(NumOrBytesType::Bytes) => {
                if *self == Self::Summable {
                    Ok(())
                } else {
                    Err(TypeErrorKind::failed_constraint(
                        ty.to_owned(),
                        self.to_owned(),
                    ))
                }
            }

            ValueType::Tuple(tuple) => {
                let tuple = tuple.to_owned();
                for element in tuple.element_types() {
                    self.apply(element, substitutions)?;
                }
                Ok(())
            }

            other => Err(TypeErrorKind::UnsupportedType(other.to_owned())),
        }
    }
}

impl PrimitiveType for NumOrBytesType {
    type Constraints = Constraints;
}

#[derive(Debug, Clone, Copy)]
struct NumOrBytesArithmetic;

impl MapPrimitiveType<NumOrBytes> for NumOrBytesArithmetic {
    type Prim = NumOrBytesType;

    fn type_of_literal(&self, lit: &NumOrBytes) -> Self::Prim {
        match lit {
            NumOrBytes::Num(_) => NumOrBytesType::Num,
            NumOrBytes::Bytes(_) => NumOrBytesType::Bytes,
        }
    }
}

impl TypeArithmetic<NumOrBytesType> for NumOrBytesArithmetic {
    fn process_unary_op<'a>(
        &self,
        substitutions: &mut Substitutions<NumOrBytesType>,
        spans: UnaryOpSpans<'a, NumOrBytesType>,
    ) -> TypeResult<'a, NumOrBytesType> {
        NumArithmetic::process_unary_op(substitutions, spans, &Constraints::Lin)
    }

    fn process_binary_op<'a>(
        &self,
        substitutions: &mut Substitutions<NumOrBytesType>,
        spans: BinaryOpSpans<'a, NumOrBytesType>,
    ) -> TypeResult<'a, NumOrBytesType> {
        let op_constraints = if let BinaryOp::Add = spans.op.extra {
            Constraints::Summable
        } else {
            Constraints::Lin
        };
        let settings = OpConstraintSettings {
            lin: &Constraints::Lin,
            ops: &op_constraints,
        };

        let comparable_type = Some(NumOrBytesType::Num);
        NumArithmetic::process_binary_op(substitutions, spans, comparable_type, settings)
    }
}

type Parser = Typed<Annotated<NumGrammar<NumOrBytes>, NumOrBytesType>>;

fn main() -> anyhow::Result<()> {
    let code = r#"
        x = 1 + 3;
        y = 0x1234 + 0x56;
        z = (1, 0x12) + (x, y);

        sum = |xs, init| xs.fold(init, |acc, x| acc + x);
        sum_of_bytes = (0xdead, 0xbe, 0xef).sum(0x);
        sum_of_tuples = ((1, 2), (3, 4)).sum((0, 0));

        product = |xs, init| xs.fold(init, |acc, x| acc * x);
        product_of_ints = (1, 5, 2).product(1);
    "#;
    let ast = Parser::parse_statements(code)?;

    let mut env = TypeEnvironment::<NumOrBytesType>::new();
    env.insert("fold", Prelude::Fold);
    env.process_with_arithmetic(&NumOrBytesArithmetic, &ast)?;

    assert_eq!(env["x"], ValueType::Prim(NumOrBytesType::Num));
    assert_eq!(env["y"].to_string(), "Bytes");
    assert_eq!(env["z"].to_string(), "(Num, Bytes)");
    assert_eq!(env["sum"].to_string(), "for<'T: Sum> (['T; N], 'T) -> 'T");
    assert_eq!(env["sum_of_bytes"].to_string(), "Bytes");
    assert_eq!(env["sum_of_tuples"].to_string(), "(Num, Num)");
    assert_eq!(
        env["product"].to_string(),
        "for<'T: Lin> (['T; N], 'T) -> 'T"
    );
    assert_eq!(env["product_of_ints"].to_string(), "Num");

    let bogus_code = "(0xca, 0xfe).product(0x)";
    let bogus_ast = Parser::parse_statements(bogus_code)?;
    let err = env
        .process_with_arithmetic(&NumOrBytesArithmetic, &bogus_ast)
        .unwrap_err();

    assert_eq!(err.to_string(), "1:0: Type `Bytes` fails constraint Lin");

    let bogus_code = "(|x| x + 1, |x| x - 5).sum(|x| x)";
    let bogus_ast = Parser::parse_statements(bogus_code)?;
    let err = env
        .process_with_arithmetic(&NumOrBytesArithmetic, &bogus_ast)
        .unwrap_err();

    assert_eq!(
        err.to_string(),
        "1:0: Type `(Num) -> Num` fails constraint Sum"
    );

    env.insert("true", ValueType::BOOL);
    let bogus_code = "((1, true), (2, true)).sum((3, true))";
    let bogus_ast = Parser::parse_statements(bogus_code)?;
    let err = env
        .process_with_arithmetic(&NumOrBytesArithmetic, &bogus_ast)
        .unwrap_err();

    assert_eq!(err.to_string(), "1:0: Type `Bool` fails constraint Sum");

    Ok(())
}
