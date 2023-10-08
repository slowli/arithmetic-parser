//! Tests for examples from the eval crate.

use std::{fmt, str::FromStr};

use arithmetic_parser::grammars::{Features, NumGrammar, Parse};
use arithmetic_typing::{
    arith::{
        BinaryOpContext, BoolArithmetic, Constraint, MapPrimitiveType, Num, NumArithmetic,
        Substitutions, TypeArithmetic, UnaryOpContext, WithBoolean,
    },
    defs::{Assertions, Prelude},
    error::{ErrorLocation, OpErrors},
    visit::Visit,
    Annotated, DynConstraints, Function, PrimitiveType, Type, TypeEnvironment, UnknownLen,
};

use crate::Hashed;

const SCHNORR_CODE: &str = include_str!("schnorr.script");
const DSA_CODE: &str = include_str!("dsa.script");
const EL_GAMAL_CODE: &str = include_str!("elgamal.script");

#[derive(Debug, Clone, Copy)]
struct U64Grammar;

impl Parse for U64Grammar {
    type Base = Annotated<NumGrammar<u64>>;
    // ^ We don't use large literals in code, so `u64` is fine

    const FEATURES: Features = Features::all();
}

fn dbg_fn<Prim: PrimitiveType>() -> Function<Prim> {
    Function::builder()
        .with_varargs(Type::Any, UnknownLen::param(0))
        .returning(Type::void())
}

fn prepare_imprecise_env() -> TypeEnvironment {
    let rand_scalar = Function::builder().returning(Type::NUM);
    let hash_to_scalar = Function::builder()
        .with_varargs(DynConstraints::just(Hashed), UnknownLen::param(0))
        .returning(Type::NUM);
    let to_scalar = Function::builder().with_arg(Type::NUM).returning(Type::NUM);

    let mut env: TypeEnvironment = Prelude::iter().chain(Assertions::iter()).collect();
    env.insert("dbg", dbg_fn())
        .insert("GEN", Type::NUM)
        .insert("ORDER", Type::NUM)
        .insert("rand_scalar", rand_scalar)
        .insert("hash_to_scalar", hash_to_scalar)
        .insert("to_scalar", to_scalar);
    env
}

#[test]
fn schnorr_signatures_imprecise() {
    //! Uses imprecise typing, with scalars and group elements not distinguished.

    let code = U64Grammar::parse_statements(SCHNORR_CODE).unwrap();
    let mut env = prepare_imprecise_env();
    env.process_statements(&code).unwrap();

    assert_eq!(env["gen"].to_string(), "() -> { pk: Num, sk: Num }");
    let Type::Object(secret_key) = &env["SecretKey"] else {
        unreachable!();
    };
    assert_eq!(
        secret_key["sign"].to_string(),
        "for<'T: Hash> (Num, 'T) -> { e: Num, s: Num }"
    );
    let Type::Object(public_key) = &env["PublicKey"] else {
        unreachable!();
    };
    assert_eq!(
        public_key["verify"].to_string(),
        "for<'T: Hash, 'U: { e: Num, s: Num }> (Num, 'T, 'U) -> Bool"
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GroupPrim {
    Bool,
    Scalar,
    GroupElement,
}

impl fmt::Display for GroupPrim {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Bool => "Bool",
            Self::Scalar => "Sc",
            Self::GroupElement => "Ge",
        })
    }
}

impl FromStr for GroupPrim {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Bool" => Ok(Self::Bool),
            "Sc" => Ok(Self::Scalar),
            "Ge" => Ok(Self::GroupElement),
            _ => Err(anyhow::anyhow!(
                "Unknown primitive type; expected `Bool`, `Sc` or `Ge`"
            )),
        }
    }
}

impl WithBoolean for GroupPrim {
    const BOOL: Self = Self::Bool;
}

impl PrimitiveType for GroupPrim {}

const SC: Type<GroupPrim> = Type::Prim(GroupPrim::Scalar);
const GE: Type<GroupPrim> = Type::Prim(GroupPrim::GroupElement);

/// Type constraint for types that can be multiplication / division operands.
#[derive(Debug, Clone, Copy)]
struct MulOperand;

impl fmt::Display for MulOperand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Mul")
    }
}

impl Constraint<GroupPrim> for MulOperand {
    fn visitor<'r>(
        &self,
        substitutions: &'r mut Substitutions<GroupPrim>,
        errors: OpErrors<'r, GroupPrim>,
    ) -> Box<dyn Visit<GroupPrim> + 'r> {
        use arithmetic_typing::arith::StructConstraint;

        StructConstraint::new(*self, |prim| *prim != GroupPrim::Bool)
            .deny_dyn_slices()
            .visitor(substitutions, errors)
    }

    fn clone_boxed(&self) -> Box<dyn Constraint<GroupPrim>> {
        Box::new(*self)
    }
}

#[derive(Debug, Clone, Copy)]
struct GroupArithmetic;

impl MapPrimitiveType<u64> for GroupArithmetic {
    type Prim = GroupPrim;

    fn type_of_literal(&self, _: &u64) -> Self::Prim {
        GroupPrim::Scalar
    }
}

impl TypeArithmetic<GroupPrim> for GroupArithmetic {
    // Naive impl: we can only negate separate scalars.
    fn process_unary_op(
        &self,
        substitutions: &mut Substitutions<GroupPrim>,
        context: &UnaryOpContext<GroupPrim>,
        errors: OpErrors<'_, GroupPrim>,
    ) -> Type<GroupPrim> {
        use arithmetic_parser::UnaryOp;

        match context.op {
            UnaryOp::Neg => {
                substitutions.unify(&SC, &context.arg, errors);
                SC
            }
            _ => BoolArithmetic.process_unary_op(substitutions, context, errors),
        }
    }

    // Naive impl: we deal only with primitive types as args.
    fn process_binary_op(
        &self,
        substitutions: &mut Substitutions<GroupPrim>,
        context: &BinaryOpContext<GroupPrim>,
        mut errors: OpErrors<'_, GroupPrim>,
    ) -> Type<GroupPrim> {
        use arithmetic_parser::BinaryOp;

        match context.op {
            BinaryOp::Add | BinaryOp::Sub => {
                substitutions.unify(&SC, &context.lhs, errors.with_location(ErrorLocation::Lhs));
                substitutions.unify(&SC, &context.rhs, errors.with_location(ErrorLocation::Rhs));
                SC
            }

            BinaryOp::Mul | BinaryOp::Div => {
                // To get even more naive, we require that we know the type of at least
                // one of the operands.
                let resolved_lhs = substitutions.fast_resolve(&context.lhs);
                let resolved_rhs = substitutions.fast_resolve(&context.rhs);

                match (resolved_lhs, resolved_rhs) {
                    (sc, _) | (_, sc) if *sc == SC => {
                        substitutions.unify(
                            &SC,
                            &context.lhs,
                            errors.with_location(ErrorLocation::Lhs),
                        );
                        substitutions.unify(
                            &SC,
                            &context.rhs,
                            errors.with_location(ErrorLocation::Rhs),
                        );
                        SC
                    }
                    (ge, _) | (_, ge) if *ge == GE => {
                        substitutions.unify(
                            &GE,
                            &context.lhs,
                            errors.with_location(ErrorLocation::Lhs),
                        );
                        substitutions.unify(
                            &GE,
                            &context.rhs,
                            errors.with_location(ErrorLocation::Rhs),
                        );
                        GE
                    }
                    _ => {
                        MulOperand
                            .visitor(substitutions, errors.with_location(ErrorLocation::Lhs))
                            .visit_type(&context.lhs);
                        MulOperand
                            .visitor(substitutions, errors.with_location(ErrorLocation::Rhs))
                            .visit_type(&context.rhs);
                        substitutions.unify(&context.lhs, &context.rhs, errors);
                        context.lhs.clone()
                    }
                }
            }

            BinaryOp::Power => {
                substitutions.unify(&GE, &context.lhs, errors.by_ref());
                substitutions.unify(&SC, &context.rhs, errors);
                GE
            }

            _ => BoolArithmetic.process_binary_op(substitutions, context, errors),
        }
    }
}

fn prepare_env() -> TypeEnvironment<GroupPrim> {
    let rand_scalar = Function::builder().returning(SC);
    let hash_to_scalar = Function::builder()
        .with_varargs(DynConstraints::just(Hashed), UnknownLen::param(0))
        .returning(SC);
    let to_scalar = Function::builder().with_arg(GE).returning(SC);

    let mut env: TypeEnvironment<GroupPrim> = Prelude::iter().chain(Assertions::iter()).collect();
    env.insert("dbg", dbg_fn())
        .insert("GEN", GE)
        .insert("ORDER", SC)
        .insert("rand_scalar", rand_scalar)
        .insert("hash_to_scalar", hash_to_scalar)
        .insert("to_scalar", to_scalar);
    env
}

#[test]
fn schnorr_signatures() {
    let code = U64Grammar::parse_statements(SCHNORR_CODE).unwrap();
    let mut env = prepare_env();
    env.process_with_arithmetic(&GroupArithmetic, &code)
        .unwrap();

    assert_eq!(env["gen"].to_string(), "() -> { pk: Ge, sk: Sc }");
    let Type::Object(secret_key) = &env["SecretKey"] else {
        unreachable!();
    };
    assert_eq!(
        secret_key["sign"].to_string(),
        "for<'T: Hash> (Sc, 'T) -> { e: Sc, s: Sc }"
    );
    let Type::Object(public_key) = &env["PublicKey"] else {
        unreachable!();
    };
    assert_eq!(
        public_key["verify"].to_string(),
        "for<'T: Hash, 'U: { e: Sc, s: Sc }> (Ge, 'T, 'U) -> Bool"
    );
}

#[test]
fn schnorr_signatures_error() {
    let bogus_code = SCHNORR_CODE.replace("s: r - self * e", "s: R - self * e");
    let code = U64Grammar::parse_statements(bogus_code.as_str()).unwrap();
    let errors = prepare_env()
        .process_with_arithmetic(&GroupArithmetic, &code)
        .unwrap_err();

    assert_eq!(errors.len(), 1);
    let err = errors.into_iter().next().unwrap();
    assert_eq!(*err.main_span().fragment(), "R");
    assert_eq!(
        err.kind().to_string(),
        "Type `Ge` is not assignable to type `Sc`"
    );
}

#[derive(Debug, Clone, Copy)]
struct Mutation {
    from: &'static str,
    to: &'static str,
    expected_msg: &'static str,
}

#[test]
fn schnorr_signatures_mutations() {
    const MUTATIONS: &[Mutation] = &[
        Mutation {
            from: "R = GEN ^ s * self ^ e;",
            to: "R = GEN ^ s * e ^ self;",
            expected_msg: "8:9: Type `Sc` is not assignable to type `Ge`",
        },
        Mutation {
            from: "R = GEN ^ s * self ^ e;",
            to: "R = GEN ^ s + self ^ e;",
            expected_msg: "7:13: Type `Ge` is not assignable to type `Sc`",
        },
        Mutation {
            from: "R = GEN ^ s * self ^ e;",
            to: "R = GEN ^ s * self * e;",
            expected_msg: "8:9: Type `Sc` is not assignable to type `Ge`",
        },
        Mutation {
            from: "R = GEN ^ s * self ^ e;",
            to: "R = (GEN, self) ^ (s, e);",
            expected_msg: "7:13: Type `(Ge, _)` is not assignable to type `Ge`",
        },
    ];

    for &mutation in MUTATIONS {
        let bogus_code = SCHNORR_CODE.replace(mutation.from, mutation.to);
        let code = U64Grammar::parse_statements(bogus_code.as_str()).unwrap();
        let errors = prepare_env()
            .process_with_arithmetic(&GroupArithmetic, &code)
            .unwrap_err();
        let err = errors.into_iter().next().unwrap();
        assert_eq!(err.to_string(), mutation.expected_msg);
    }
}

#[test]
fn dsa_signatures_imprecise() {
    let code = U64Grammar::parse_statements(DSA_CODE).unwrap();
    let mut env = prepare_imprecise_env();
    env.process_statements(&code).unwrap();

    assert_eq!(env["gen"].to_string(), "() -> { pk: Num, sk: Num }");
    let Type::Object(secret_key) = &env["SecretKey"] else {
        unreachable!();
    };
    assert_eq!(
        secret_key["sign"].to_string(),
        "for<'T: Hash> (Num, 'T) -> { r: Num, s: Num }"
    );
    let Type::Object(public_key) = &env["PublicKey"] else {
        unreachable!();
    };
    assert_eq!(
        public_key["verify"].to_string(),
        "for<'T: Hash, 'U: { r: Num, s: Num }> (Num, 'T, 'U) -> Bool"
    );
}

#[test]
fn dsa_signatures() {
    let code = U64Grammar::parse_statements(DSA_CODE).unwrap();
    let mut env = prepare_env();
    env.process_with_arithmetic(&GroupArithmetic, &code)
        .unwrap();

    assert_eq!(env["gen"].to_string(), "() -> { pk: Ge, sk: Sc }");
    let Type::Object(secret_key) = &env["SecretKey"] else {
        unreachable!();
    };
    assert_eq!(
        secret_key["sign"].to_string(),
        "for<'T: Hash> (Sc, 'T) -> { r: Sc, s: Sc }"
    );
    let Type::Object(public_key) = &env["PublicKey"] else {
        unreachable!();
    };
    assert_eq!(
        public_key["verify"].to_string(),
        "for<'T: Hash, 'U: { r: Sc, s: Sc }> (Ge, 'T, 'U) -> Bool"
    );
}

#[test]
fn dsa_signatures_mutations() {
    const MUTATIONS: &[Mutation] = &[
        Mutation {
            from: "r = (GEN ^ k).to_scalar();",
            to: "r = GEN ^ k;",
            expected_msg: "16:40: Type `Ge` is not assignable to type `Sc`",
        },
        Mutation {
            from: "(GEN ^ u1 * self ^ u2).to_scalar() == r",
            to: "GEN ^ u1 * self ^ u2 == r",
            expected_msg: "8:9: Type `Sc` is not assignable to type `Ge`",
        },
        Mutation {
            from: "assert(pk.{PublicKey.verify}(message, signature));",
            to: "assert(message.{PublicKey.verify}(pk, signature));",
            expected_msg: "37:12: Type `Sc` is not assignable to type `Ge`",
        },
    ];

    for &mutation in MUTATIONS {
        let bogus_code = DSA_CODE.replace(mutation.from, mutation.to);
        let code = U64Grammar::parse_statements(bogus_code.as_str()).unwrap();
        let errors = prepare_env()
            .process_with_arithmetic(&GroupArithmetic, &code)
            .unwrap_err();
        let err = errors.into_iter().next().unwrap();
        assert_eq!(err.to_string(), mutation.expected_msg);
    }
}

#[test]
fn el_gamal_encryption_imprecise() {
    let code = U64Grammar::parse_statements(EL_GAMAL_CODE).unwrap();
    let mut env = prepare_imprecise_env();
    env.process_statements(&code).unwrap();

    assert_eq!(env["gen"].to_string(), "() -> { pk: Num, sk: Num }");
    let Type::Object(public_key) = &env["PublicKey"] else {
        unreachable!();
    };
    assert_eq!(
        public_key["encrypt"].to_string(),
        "(Num, Num) -> { B: Num, R: Num }"
    );
    let Type::Object(secret_key) = &env["SecretKey"] else {
        unreachable!();
    };
    assert_eq!(
        secret_key["decrypt"].to_string(),
        "for<'T: Ops, 'U: { B: 'T, R: 'T }> ('T, 'U) -> 'T"
    );
}

#[test]
fn el_gamal_encryption() {
    let code = U64Grammar::parse_statements(EL_GAMAL_CODE).unwrap();
    let mut env = prepare_env();
    env.process_with_arithmetic(&GroupArithmetic, &code)
        .unwrap();

    assert_eq!(env["gen"].to_string(), "() -> { pk: Ge, sk: Sc }");
    let Type::Object(public_key) = &env["PublicKey"] else {
        unreachable!();
    };
    assert_eq!(
        public_key["encrypt"].to_string(),
        "(Ge, Ge) -> { B: Ge, R: Ge }"
    );
    let Type::Object(secret_key) = &env["SecretKey"] else {
        unreachable!();
    };
    assert_eq!(
        secret_key["decrypt"].to_string(),
        "for<'T: { B: Ge, R: Ge }> (Sc, 'T) -> Ge"
    );
    assert_eq!(
        env["encrypt_and_combine"].to_string(),
        "(Ge, [Ge; N]) -> { B: Ge, R: Ge }"
    );
}

#[test]
fn rfold() {
    let code = include_str!("rfold.script");
    let code = U64Grammar::parse_statements(code).unwrap();
    let mut env: TypeEnvironment = Prelude::iter().chain(Assertions::iter()).collect();
    env.insert("INF", Type::NUM)
        .process_with_arithmetic(&NumArithmetic::with_comparisons(), &code)
        .unwrap();

    assert_eq!(
        env["rfold"].to_string(),
        "([Num; N], 'T, ('T, Num) -> 'T) -> 'T"
    );
    assert_eq!(env["min"], Type::NUM);
    assert_eq!(env["max"], Type::NUM);
}

#[test]
fn quick_sort() {
    let code = include_str!("quick_sort.script");
    let code = U64Grammar::parse_statements(code).unwrap();

    let rand_num = Function::builder()
        .with_arg(Type::NUM)
        .with_arg(Type::NUM)
        .returning(Type::NUM);

    let mut env: TypeEnvironment = Prelude::iter().chain(Assertions::iter()).collect();
    env.insert("array", Prelude::array(Num::Num))
        .insert("rand_num", rand_num)
        .process_with_arithmetic(&NumArithmetic::with_comparisons(), &code)
        .unwrap();

    assert_eq!(env["sort"].to_string(), "([Num]) -> [Num]");
}
