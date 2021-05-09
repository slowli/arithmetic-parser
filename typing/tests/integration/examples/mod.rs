//! Tests for examples from the eval crate.

use std::{fmt, str::FromStr};

use arithmetic_parser::grammars::{Features, NumGrammar, Parse};
use arithmetic_typing::{
    arith::{
        BinaryOpContext, BoolArithmetic, ConstraintSet, MapPrimitiveType, NumArithmetic,
        NumConstraints, TypeArithmetic, UnaryOpContext, WithBoolean,
    },
    error::{ErrorLocation, OpErrors},
    Annotated, Assertions, ErrorKind, FnType, Prelude, PrimitiveType, Substitutions, Type,
    TypeEnvironment, UnknownLen,
};

const SCHNORR_CODE: &str = include_str!("schnorr.script");
const DSA_CODE: &str = include_str!("dsa.script");
const EL_GAMAL_CODE: &str = include_str!("elgamal.script");

#[derive(Debug, Clone, Copy)]
struct U64Grammar;

impl Parse<'_> for U64Grammar {
    type Base = Annotated<NumGrammar<u64>>;
    // ^ We don't use large literals in code, so `u64` is fine

    const FEATURES: Features = Features::all();
}

fn dbg_fn<Prim: PrimitiveType>() -> FnType<Prim> {
    FnType::builder()
        .with_varargs(Type::any(), UnknownLen::param(0))
        .returning(Type::void())
}

fn prepare_imprecise_env() -> TypeEnvironment {
    let rand_scalar = FnType::builder().returning(Type::NUM);
    let hash_to_scalar = FnType::builder()
        .with_varargs(
            ConstraintSet::just(NumConstraints::Lin),
            UnknownLen::param(0),
        )
        .returning(Type::NUM);
    let to_scalar = FnType::builder().with_arg(Type::NUM).returning(Type::NUM);

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

    assert_eq!(env["gen"].to_string(), "() -> (Num, Num)");
    assert_eq!(
        env["sign"].to_string(),
        "for<'T: Lin> ('T, Num) -> (Num, Num)"
    );
    assert_eq!(
        env["verify"].to_string(),
        "for<'T: Lin> ((Num, Num), 'T, Num) -> Bool"
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
                    (Type::Tuple(_), Type::Tuple(_)) => {
                        substitutions.unify(&context.lhs, &context.rhs, errors.by_ref());
                        context.lhs.clone()
                    }

                    _ => {
                        // FIXME: add `other` variant to `ErrorKind`?
                        let err = ErrorKind::unsupported(context.op);
                        errors.push(err);
                        substitutions.new_type_var()
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
    let rand_scalar = FnType::builder().returning(SC);
    // TODO: too wide typing; we don't want to hash fns.
    let hash_to_scalar = FnType::builder()
        .with_varargs(Type::any(), UnknownLen::param(0))
        .returning(SC);
    let to_scalar = FnType::builder().with_arg(GE).returning(SC);

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

    assert_eq!(env["gen"].to_string(), "() -> (Sc, Ge)");
    assert_eq!(env["sign"].to_string(), "('T, Sc) -> (Sc, Sc)");
    assert_eq!(env["verify"].to_string(), "((Sc, Sc), 'T, Ge) -> Bool");
}

#[test]
fn schnorr_signatures_error() {
    let bogus_code = SCHNORR_CODE.replace("(e, r - sk * e)", "(e, R - sk * e)");
    let code = U64Grammar::parse_statements(bogus_code.as_str()).unwrap();
    let errors = prepare_env()
        .process_with_arithmetic(&GroupArithmetic, &code)
        .unwrap_err();

    assert_eq!(errors.len(), 1);
    let err = errors.into_iter().next().unwrap();
    assert_eq!(*err.span().fragment(), "R");
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
            from: "R = GEN ^ s * pk ^ e;",
            to: "R = GEN ^ s * e ^ pk;",
            expected_msg: "17:5: Type `Sc` is not assignable to type `Ge`",
        },
        Mutation {
            from: "R = GEN ^ s * pk ^ e;",
            to: "R = GEN ^ s + pk ^ e;",
            expected_msg: "16:9: Type `Ge` is not assignable to type `Sc`",
        },
        Mutation {
            from: "R = GEN ^ s * pk ^ e;",
            to: "R = GEN ^ s * pk * e;",
            expected_msg: "17:5: Type `Sc` is not assignable to type `Ge`",
        },
        Mutation {
            from: "R = GEN ^ s * pk ^ e;",
            to: "R = (GEN, pk) ^ (s, e);",
            expected_msg: "16:9: Type `(Ge, _)` is not assignable to type `Ge`",
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

    assert_eq!(env["gen"].to_string(), "() -> (Num, Num)");
    assert_eq!(
        env["sign"].to_string(),
        "for<'T: Lin> ('T, Num) -> (Num, Num)"
    );
    assert_eq!(
        env["verify"].to_string(),
        "for<'T: Lin> ((Num, Num), 'T, Num) -> Bool"
    );
}

#[test]
fn dsa_signatures() {
    let code = U64Grammar::parse_statements(DSA_CODE).unwrap();
    let mut env = prepare_env();
    env.process_with_arithmetic(&GroupArithmetic, &code)
        .unwrap();

    assert_eq!(env["gen"].to_string(), "() -> (Sc, Ge)");
    assert_eq!(env["sign"].to_string(), "('T, Sc) -> (Sc, Sc)");
    assert_eq!(env["verify"].to_string(), "((Sc, Sc), 'T, Ge) -> Bool");
}

#[test]
fn dsa_signatures_mutations() {
    const MUTATIONS: &[Mutation] = &[
        Mutation {
            from: "r = (GEN ^ k).to_scalar();",
            to: "r = GEN ^ k;",
            expected_msg: "11:36: Type `Ge` is not assignable to type `Sc`",
        },
        Mutation {
            from: "(GEN ^ u1 * pk ^ u2).to_scalar() == r",
            to: "GEN ^ u1 * pk ^ u2 == r",
            expected_msg: "17:5: Type `Sc` is not assignable to type `Ge`",
        },
        Mutation {
            from: "assert(signature.verify(message, pk));",
            to: "assert(signature.verify(pk, message));",
            expected_msg: "30:33: Type `Sc` is not assignable to type `Ge`",
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

    assert_eq!(env["gen"].to_string(), "() -> (Num, Num)");
    assert_eq!(env["encrypt"].to_string(), "(Num, Num) -> (Num, Num)");
    assert_eq!(
        env["decrypt"].to_string(),
        "for<'T: Ops> (('T, 'T), 'T) -> 'T"
    );
}

#[test]
fn el_gamal_encryption() {
    let code = U64Grammar::parse_statements(EL_GAMAL_CODE).unwrap();
    let mut env = prepare_env();
    env.process_with_arithmetic(&GroupArithmetic, &code)
        .unwrap();

    assert_eq!(env["gen"].to_string(), "() -> (Sc, Ge)");
    assert_eq!(env["encrypt"].to_string(), "(Ge, Ge) -> (Ge, Ge)");
    assert_eq!(env["decrypt"].to_string(), "((Ge, Ge), Sc) -> Ge");

    // `Ge` annotations are a temporary measure until constraints are revisited.
    let additional_code = r#"
        ONE = GEN ^ 0;
        encrypt_and_combine = |messages, pk| {
            messages.map(|msg| msg.encrypt(pk)).fold(
                (ONE, ONE),
                |(R_acc: Ge, B_acc: Ge), (R, B)| (R_acc * R, B_acc * B),
            )
        };

        messages = (1, 2, 3, 4, 5).map(|_| GEN ^ rand_scalar());
        assert_eq(
            encrypt_and_combine(messages, pk).decrypt(sk),
            messages.fold(ONE, |acc: Ge, msg| acc * msg)
        );
    "#;
    let additional_code = U64Grammar::parse_statements(additional_code).unwrap();
    env.process_with_arithmetic(&GroupArithmetic, &additional_code)
        .unwrap();

    assert_eq!(
        env["encrypt_and_combine"].to_string(),
        "([Ge; N], Ge) -> (Ge, Ge)"
    );
}

#[test]
fn rfold() {
    let code = include_str!("rfold.script");
    let code = U64Grammar::parse_statements(code).unwrap();
    let mut env: TypeEnvironment = Prelude::iter().chain(Assertions::iter()).collect();
    let tuple_len = FnType::builder()
        .with_arg(Type::slice(Type::any(), UnknownLen::param(0)))
        .returning(Type::NUM);
    env.insert("MIN", Type::NUM)
        .insert("MAX", Type::NUM)
        .insert("len", tuple_len);
    env.process_with_arithmetic(&NumArithmetic::with_comparisons(), &code)
        .unwrap();

    assert_eq!(
        env["rfold"].to_string(),
        "([Num; N], 'T, ('T, Num) -> 'T) -> 'T"
    );
    assert_eq!(env["min"], Type::NUM);
    assert_eq!(env["max"], Type::NUM);
}
