//! Tests for errors.

use assert_matches::assert_matches;

use arithmetic_parser::grammars::Parse;
use arithmetic_typing::{
    arith::{NumArithmetic, NumConstraints},
    error::{ErrorContext, ErrorKind, ErrorLocation, TupleContext},
    Prelude, TupleIndex, TupleLen, Type, TypeEnvironment,
};

use super::{assert_incompatible_types, hash_fn_type, zip_fn_type, ErrorsExt, F32Grammar};

mod annotations;
mod multiple;
mod recovery;

fn fn_arg(index: usize) -> ErrorLocation {
    ErrorLocation::FnArg(Some(TupleIndex::Start(index)))
}

fn tuple_element(index: usize) -> ErrorLocation {
    ErrorLocation::TupleElement(Some(TupleIndex::Start(index)))
}

#[test]
fn type_recursion() {
    let code = "bog = |x| x + (x, 2);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(*err.span().fragment(), "x + (x, 2)");
    assert!(err.location().is_empty());
    assert_matches!(err.context(), ErrorContext::BinaryOp(_));
    assert_matches!(
        err.kind(),
        ErrorKind::RecursiveType(ref ty) if ty.to_string() == "('T, Num)"
    );
    assert_eq!(
        err.kind().to_string(),
        "Cannot unify type 'T with a type containing it: ('T, Num)"
    );
}

#[test]
fn indirect_type_recursion() {
    let code = r#"
        add = |x, y| x + y; // this function is fine
        bog = |x| add(x, (1, x)); // ...but its application is not
    "#;

    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();
    assert_matches!(
        err.kind(),
        ErrorKind::RecursiveType(ref ty) if ty.to_string() == "(Num, 'T)"
    );
}

#[test]
fn recursion_via_fn() {
    let code = "func = |bog| bog(1, bog);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();
    assert_matches!(
        err.kind(),
        ErrorKind::RecursiveType(ref ty) if ty.to_string() == "(Num, 'T) -> _"
    );
}

#[test]
fn unknown_method() {
    let code = "bar = 3.do_something();";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(*err.span().fragment(), "do_something");
    assert!(err.location().is_empty());
    assert_matches!(err.context(), ErrorContext::None);
    assert_matches!(err.kind(), ErrorKind::UndefinedVar(name) if name == "do_something");
    assert_eq!(
        err.kind().to_string(),
        "Variable `do_something` is not defined"
    );
}

#[test]
fn immediately_invoked_function_with_invalid_arg() {
    let code = "flag = (|x| x + x)(4 == 7);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(*err.span().fragment(), "4 == 7");
    assert_eq!(err.location(), [fn_arg(0)]);
    assert_matches!(
        err.context(),
        ErrorContext::FnCall { definition, call_signature }
            if definition.to_string() == "for<'T: Ops> ('T) -> 'T"
            && call_signature.to_string() == "(Bool) -> Bool"
    );
    assert_matches!(
        err.kind(),
        ErrorKind::FailedConstraint { ty, .. } if *ty == Type::BOOL
    );
    assert_eq!(err.kind().to_string(), "Type `Bool` fails constraint `Ops`");
}

#[test]
fn destructuring_error_on_assignment() {
    let bogus_code = "(x, y, ...zs) = (1,);";
    let block = F32Grammar::parse_statements(bogus_code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(*err.span().fragment(), "(x, y, ...zs)");
    assert!(err.location().is_empty());
    assert_matches!(
        err.context(),
        ErrorContext::Assignment { lhs, rhs }
            if lhs.to_string() == "(_, _, ...[_; _])" && rhs.to_string() == "(Num)"
    );
    assert_matches!(
        err.kind(),
        ErrorKind::TupleLenMismatch { lhs, rhs, .. }
            if lhs.to_string() == "_ + 2" && *rhs == TupleLen::from(1)
    );
    assert_eq!(
        err.kind().to_string(),
        "Expected a tuple with _ + 2 elements, got one with 1 elements"
    );
}

#[test]
fn incorrect_tuple_length_returned_from_fn() {
    let code = "double = |x| (x, x); (z,) = double(5);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_matches!(
        err.kind(),
        ErrorKind::TupleLenMismatch {
            lhs,
            rhs,
            context: TupleContext::Generic,
        } if *lhs == TupleLen::from(1) && *rhs == TupleLen::from(2)
    );
}

#[test]
fn parametric_fn_passed_as_arg_with_unsatisfiable_requirements() {
    let code = r#"
        concat = |x| { |y| (x, y) };
        partial = concat(3); // (U) -> (Num, U)

        bogus = |fun| fun(1) == 4;
        bogus(partial);
    "#;

    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_incompatible_types(
        &err.kind(),
        &Type::NUM,
        &Type::Tuple(vec![Type::NUM; 2].into()),
    );
}

#[test]
fn parametric_fn_passed_as_arg_with_recursive_requirements() {
    let code = r#"
        concat = |x| { |y| (x, y) };
        partial = concat(3); // (U) -> (Num, U)
        bogus = |fun| { |x| fun(x) == x };
        bogus(partial);
    "#;

    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_matches!(err.kind(), ErrorKind::RecursiveType(_));
}

#[test]
fn function_passed_as_arg_invalid_arity() {
    let code = r#"
        mapper = |(x, y), map| (map(x), map(y));
        mapper((1, 2), |x, y| x + y);
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(*err.span().fragment(), "|x, y| x + y");
    assert_eq!(err.location(), [fn_arg(1)]);
    let expected_call_signature = "((Num, Num), for<'T: Ops> ('T, 'T) -> 'T) -> (Num, Num)";
    assert_matches!(
        err.context(),
        ErrorContext::FnCall { definition, call_signature }
            if definition.to_string() == "(('T, 'T), ('T) -> 'U) -> ('U, 'U)"
            && call_signature.to_string() == expected_call_signature
    );

    assert_matches!(
        err.kind(),
        ErrorKind::TupleLenMismatch {
            lhs,
            rhs,
            context: TupleContext::FnArgs,
        } if *lhs == TupleLen::from(2) && *rhs == TupleLen::from(1)
    );
    assert_eq!(
        err.kind().to_string(),
        "Function expects 2 args, but is called with 1 args"
    );
}

#[test]
fn function_passed_as_arg_invalid_arg_type() {
    let code = r#"
        mapper = |(x, y), map| (map(x), map(y));
        mapper((1, 2), |(x, _)| x);
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_matches!(
        err.kind(),
        ErrorKind::TypeMismatch(Type::Tuple(t), rhs)
            if t.len() == TupleLen::from(2) && *rhs == Type::NUM
    );
}

#[test]
fn function_passed_as_arg_invalid_input() {
    let code = r#"
        mapper = |(x, y), map| (map(x), map(y));
        mapper((1, 2 != 3), |x| x + 2);
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(*err.span().fragment(), "2 != 3");
    assert_eq!(err.location(), [fn_arg(0), tuple_element(1)]);
    assert_matches!(
        err.context(),
        ErrorContext::FnCall { definition, call_signature }
            if definition.to_string() == "(('T, 'T), ('T) -> 'U) -> ('U, 'U)"
            || call_signature.to_string() == "((Num, Bool), (Num) -> Num) -> (Num, Num)"
    );
    assert_incompatible_types(&err.kind(), &Type::NUM, &Type::BOOL);
    assert_eq!(
        err.kind().to_string(),
        "Type `Bool` is not assignable to type `Num`"
    );
}

#[test]
fn incorrect_arg_in_slices() {
    let code = "(1, 2 == 3).map(|x| x);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::Map);

    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(*err.span().fragment(), "2 == 3");
    assert_eq!(err.location(), [fn_arg(0), tuple_element(1)]);
    assert_matches!(
        err.context(),
        ErrorContext::FnCall { definition, call_signature }
            if definition.to_string() == "(['T; N], ('T) -> 'U) -> ['U; N]"
            && call_signature.to_string() == "((Num, Bool), ('T) -> 'T) -> (Num, Num)"
    );
    assert_incompatible_types(&err.kind(), &Type::NUM, &Type::BOOL);
}

#[test]
fn unifying_length_vars_error() {
    let code = "(1, 2).zip_with((3, 4, 5));";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("zip_with", zip_fn_type());
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(*err.span().fragment(), "(3, 4, 5)");
    assert_eq!(err.location(), [fn_arg(1)]);
    assert_matches!(
        err.context(),
        ErrorContext::FnCall { definition, call_signature }
            if *definition == type_env["zip_with"]
            && call_signature.to_string().starts_with("((Num, Num), (Num, Num, Num)) ->")
    );

    assert_matches!(
        err.kind(),
        ErrorKind::TupleLenMismatch {
            lhs,
            rhs,
            context: TupleContext::Generic,
        } if *lhs == TupleLen::from(2) && *rhs == TupleLen::from(3)
    );
}

#[test]
fn cannot_destructure_dynamic_slice() {
    let code = "(x, y) = (1, 2, 3).filter(|x| x != 1);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("filter", Prelude::Filter);
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_matches!(
        err.kind(),
        ErrorKind::TupleLenMismatch { lhs, .. } if *lhs == TupleLen::from(2)
    );
}

#[test]
fn comparisons_when_switched_off() {
    let code = "(1, 2, 3).filter(|x| x > 1)";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("filter", Prelude::Filter);
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(*err.span().fragment(), "x > 1");
    assert!(err.location().is_empty());
    assert_matches!(err.context(), ErrorContext::BinaryOp(_));
    assert_matches!(err.kind(), ErrorKind::UnsupportedFeature(_));
    assert_eq!(
        err.kind().to_string(),
        "Unsupported binary op: greater comparison"
    );
}

#[test]
fn constraint_error() {
    let code = "add = |x, y| x + y; add(1 == 2, 1 == 3)";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env
        .process_with_arithmetic(&NumArithmetic::with_comparisons(), &block)
        .unwrap_err()
        .single();

    assert_eq!(*err.span().fragment(), "1 == 2");
    assert_matches!(
        err.kind(),
        ErrorKind::FailedConstraint {
            ty,
            constraint: NumConstraints::Ops,
        } if *ty == Type::BOOL
    );
}

#[test]
fn any_type_with_bound_with_bogus_function_call() {
    let code = "hash(1, |x| x + 1)";
    let block = F32Grammar::parse_statements(code).unwrap();

    let mut type_env = TypeEnvironment::new();
    let err = type_env
        .insert("hash", hash_fn_type())
        .process_statements(&block)
        .unwrap_err()
        .single();

    assert_eq!(*err.span().fragment(), "|x| x + 1");
    assert_eq!(err.location(), [fn_arg(1)]);
    assert_matches!(
        err.context(),
        ErrorContext::FnCall { call_signature, .. }
            if call_signature.to_string() == "(Num, (Num) -> Num) -> Num"
    );

    assert_matches!(
        err.kind(),
        ErrorKind::FailedConstraint {
            ty: Type::Function(_),
            ..
        }
    );
}

#[test]
fn any_type_with_bound_in_tuple() {
    let mut type_env = TypeEnvironment::new();
    type_env.insert("some_lin", Type::Any(NumConstraints::Lin));

    let bogus_call = "some_lin(1)";
    let bogus_call = F32Grammar::parse_statements(bogus_call).unwrap();
    let err = type_env
        .process_statements(&bogus_call)
        .unwrap_err()
        .single();

    assert_matches!(
        err.kind(),
        ErrorKind::FailedConstraint {
            ty: Type::Function(_),
            ..
        }
    );

    let destructure = "(x, y) = some_lin; !x";
    let destructure = F32Grammar::parse_statements(destructure).unwrap();
    let err = type_env
        .process_statements(&destructure)
        .unwrap_err()
        .single();

    assert_eq!(*err.span().fragment(), "!x");
    assert_eq!(err.location(), []);
    assert_matches!(err.context(), ErrorContext::UnaryOp(_));
    assert_matches!(
        err.kind(),
        ErrorKind::FailedConstraint { ty, .. } if *ty == Type::BOOL
    );
    assert_eq!(err.kind().to_string(), "Type `Bool` fails constraint `Lin`");
}

#[test]
fn locating_type_with_failed_constraint() {
    let mut type_env = TypeEnvironment::new();
    for &code in &["5 + (1, true)", "(1, true) + 5"] {
        let block = F32Grammar::parse_statements(code).unwrap();
        let err = type_env
            .insert("true", Type::BOOL)
            .process_statements(&block)
            .unwrap_err()
            .single();

        assert_eq!(*err.span().fragment(), "true");
        assert_eq!(err.location()[1..], [tuple_element(1)]);
        assert_matches!(err.context(), ErrorContext::BinaryOp(_));
        assert_matches!(err.kind(), ErrorKind::FailedConstraint { .. });
        assert_eq!(err.kind().to_string(), "Type `Bool` fails constraint `Lin`");
    }
}

#[test]
fn locating_tuple_middle_with_failed_constraint() {
    let code = "|xs| { (_: Num, ...flags) = xs; flags.map(|flag| !flag); hash(xs) }";
    let block = F32Grammar::parse_statements(code).unwrap();
    let err = TypeEnvironment::new()
        .insert("map", Prelude::Map)
        .insert("hash", hash_fn_type())
        .process_statements(&block)
        .unwrap_err()
        .single();

    assert_eq!(*err.span().fragment(), "xs");
    assert_eq!(*err.root_span().fragment(), "hash(xs)");
    assert_eq!(err.location(), [fn_arg(0), TupleIndex::Middle.into()]);
    assert_matches!(
        err.context(),
        ErrorContext::FnCall { call_signature, .. }
            if call_signature.to_string() == "((Num, ...[Bool; _])) -> Num"
    );
    assert_matches!(err.kind(), ErrorKind::FailedConstraint { ty, .. } if *ty == Type::BOOL);
}
