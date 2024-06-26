//! Demonstrates how to use custom type constraints with custom primitive types.

use std::{fmt, str::FromStr};

use arithmetic_parser::{
    grammars::{NumGrammar, NumLiteral, Parse},
    BinaryOp, InputSpan, NomResult,
};
use arithmetic_typing::{
    arith::*, defs::Prelude, error::OpErrors, visit::Visit, Annotated, PrimitiveType, Type,
    TypeEnvironment,
};

/// Literal for arithmetic: either an integer or a byte buffer.
#[allow(dead_code)]
// ^ Variant values would be used in arithmetic implementation, which is omitted since it's not the focus
// of the example.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// Constraints imposed on `Type<NumOrBytesType>`. Besides linearity,
/// we consider its weaker variant: ability to add values of type `T`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Constraints {
    /// Type is summable (can be used as an argument for binary `+`).
    Summable,
    /// Type is linear (can be used as an argument for all arithmetic ops).
    Lin,
}

impl fmt::Display for Constraints {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Lin => "Lin",
            Self::Summable => "Sum",
        })
    }
}

impl Constraint<NumOrBytesType> for Constraints {
    fn visitor<'r>(
        &self,
        substitutions: &'r mut Substitutions<NumOrBytesType>,
        errors: OpErrors<'r, NumOrBytesType>,
    ) -> Box<dyn Visit<NumOrBytesType> + 'r> {
        let predicate: fn(&NumOrBytesType) -> bool = match self {
            Self::Summable => |prim| matches!(prim, NumOrBytesType::Bytes | NumOrBytesType::Num),
            Self::Lin => |prim| *prim == NumOrBytesType::Num,
        };
        StructConstraint::new(*self, predicate)
            .deny_dyn_slices()
            .visitor(substitutions, errors)
    }

    fn clone_boxed(&self) -> Box<dyn Constraint<NumOrBytesType>> {
        Box::new(*self)
    }
}

impl PrimitiveType for NumOrBytesType {}

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
    fn process_unary_op(
        &self,
        substitutions: &mut Substitutions<NumOrBytesType>,
        context: &UnaryOpContext<NumOrBytesType>,
        errors: OpErrors<'_, NumOrBytesType>,
    ) -> Type<NumOrBytesType> {
        NumArithmetic::process_unary_op(substitutions, context, errors, &Constraints::Lin)
    }

    fn process_binary_op(
        &self,
        substitutions: &mut Substitutions<NumOrBytesType>,
        context: &BinaryOpContext<NumOrBytesType>,
        errors: OpErrors<'_, NumOrBytesType>,
    ) -> Type<NumOrBytesType> {
        let op_constraints = if let BinaryOp::Add = context.op {
            Constraints::Summable
        } else {
            Constraints::Lin
        };
        let settings = OpConstraintSettings {
            lin: &Constraints::Lin,
            ops: &op_constraints,
        };

        let comparable_type = Some(NumOrBytesType::Num);
        NumArithmetic::process_binary_op(substitutions, context, errors, comparable_type, settings)
    }
}

type Parser = Annotated<NumGrammar<NumOrBytes>>;

fn main() -> anyhow::Result<()> {
    let code = "
        x = 1 + 3;
        y = 0x1234 + 0x56;
        z = (1, 0x12) + (x, y);

        sum = |xs, init| xs.fold(init, |acc, x| acc + x);
        sum_of_bytes = (0xdead, 0xbe, 0xef).sum(0x);
        sum_of_tuples = ((1, 2), (3, 4)).sum((0, 0));

        product = |xs, init| xs.fold(init, |acc, x| acc * x);
        product_of_ints = (1, 5, 2).product(1);
    ";
    let ast = Parser::parse_statements(code)?;

    let mut env = TypeEnvironment::<NumOrBytesType>::new();
    env.insert("fold", Prelude::Fold);
    env.process_with_arithmetic(&NumOrBytesArithmetic, &ast)?;

    assert_eq!(env["x"], Type::Prim(NumOrBytesType::Num));
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

    assert_eq!(err.to_string(), "1:2: Type `Bytes` fails constraint `Lin`");

    let bogus_code = "(|x| x + 1, |x| x - 5).sum(|x| x)";
    let bogus_ast = Parser::parse_statements(bogus_code)?;
    let err = env
        .process_with_arithmetic(&NumOrBytesArithmetic, &bogus_ast)
        .unwrap_err();

    assert_eq!(
        err.to_string(),
        "1:2: Type `(Num) -> Num` fails constraint `Sum`"
    );

    env.insert("true", Type::BOOL);
    let bogus_code = "((1, true), (2, true)).sum((3, true))";
    let bogus_ast = Parser::parse_statements(bogus_code)?;
    let err = env
        .process_with_arithmetic(&NumOrBytesArithmetic, &bogus_ast)
        .unwrap_err();

    assert_eq!(err.to_string(), "1:6: Type `Bool` fails constraint `Sum`");
    Ok(())
}
