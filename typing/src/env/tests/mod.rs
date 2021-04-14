use arithmetic_parser::grammars::{NumGrammar, Parse, Typed};
use assert_matches::assert_matches;

use super::*;
use crate::{
    arith::NumConstraints,
    error::{ErrorLocation, TupleLenMismatchContext, TypeErrorKind},
    Annotated, Num, Prelude, TupleLen, UnknownLen,
};

mod annotations;
mod error_handling;

type F32Grammar = Typed<Annotated<NumGrammar<f32>>>;

fn assert_incompatible_types<Prim: PrimitiveType>(
    err: &TypeErrorKind<Prim>,
    first: &Type<Prim>,
    second: &Type<Prim>,
) {
    let (x, y) = match err {
        TypeErrorKind::TypeMismatch(x, y) => (x, y),
        _ => panic!("Unexpected error type: {:?}", err),
    };
    assert!(
        (x == first && y == second) || (x == second && y == first),
        "Unexpected incompatible types: {:?}, expected: {:?}",
        (x, y),
        (first, second)
    );
}

fn hash_fn_type() -> FnType<Num> {
    FnType {
        args: Slice::new(Type::Any(NumConstraints::Lin), UnknownLen::param(0)).into(),
        return_type: Type::NUM,
        params: None,
    }
}

#[test]
fn hash_fn_type_display() {
    assert_eq!(hash_fn_type().to_string(), "(...[any Lin; N]) -> Num");
}

/// `zip` function signature.
pub fn zip_fn_type() -> FnType<Num> {
    FnType::builder()
        .with_arg(Type::param(0).repeat(UnknownLen::param(0)))
        .with_arg(Type::param(1).repeat(UnknownLen::param(0)))
        .returning(Type::slice(
            (Type::param(0), Type::param(1)),
            UnknownLen::param(0),
        ))
        .with_static_lengths(&[0])
        .into()
}

#[test]
fn zip_fn_type_display() {
    let zip_fn_string = zip_fn_type().to_string();
    assert_eq!(
        zip_fn_string,
        "for<len! N> (['T; N], ['U; N]) -> [('T, 'U); N]"
    );
}

#[test]
fn statements_with_a_block() {
    let code = "y = { x = 3; 2 * x }; x ^ y == 6 * x;";
    let block = F32Grammar::parse_statements(code).unwrap();

    let mut type_env: TypeEnvironment<Num> = vec![("x", Type::NUM)].into_iter().collect();
    type_env.process_statements(&block).unwrap();
    assert_eq!(*type_env.get("y").unwrap(), Type::NUM);
}

#[test]
fn boolean_statements() {
    let code = "y = x == x ^ 2; y = y || { x = 3; x != 7 };";
    let block = F32Grammar::parse_statements(code).unwrap();

    let mut type_env: TypeEnvironment<Num> = vec![("x", Type::NUM)].into_iter().collect();
    type_env.process_statements(&block).unwrap();
    assert_eq!(type_env["y"], Type::BOOL);
}

#[test]
fn spreading_binary_ops() {
    let code = r#"
        x = 3 * (1, 2);
        y = (1, x, 3) * 4;
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["x"].to_string(), "(Num, Num)");
    assert_eq!(type_env["y"].to_string(), "(Num, (Num, Num), Num)");
}

#[test]
fn function_definition() {
    let code = r#"
        sign = |x, msg| {
            (r, R) = hash() * (1, 3);
            c = hash(R, msg);
            (R, r + c * x)
        };
    "#;

    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("hash", hash_fn_type());

    type_env.process_statements(&block).unwrap();
    assert_eq!(
        type_env.get("sign").unwrap().to_string(),
        "for<'T: Lin> (Num, 'T) -> (Num, Num)"
    );
}

#[test]
fn non_linear_types_in_function() {
    let code = r#"
        compare = |x, y| x == y;
        compare_hash = |x, z| x == 2 ^ hash(z);
        add_hashes = |x, y| hash(x, y) + hash(y, x);
    "#;

    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("hash", hash_fn_type());

    type_env.process_statements(&block).unwrap();
    assert_eq!(
        type_env.get("compare").unwrap().to_string(),
        "('T, 'T) -> Bool"
    );
    assert_eq!(
        type_env.get("compare_hash").unwrap().to_string(),
        "for<'T: Lin> (Num, 'T) -> Bool"
    );
    assert_eq!(
        type_env.get("add_hashes").unwrap().to_string(),
        "for<'T: Lin, 'U: Lin> ('T, 'U) -> Num"
    );
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
        TypeErrorKind::RecursiveType(ref ty) if ty.to_string() == "('T, Num)"
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
        TypeErrorKind::RecursiveType(ref ty) if ty.to_string() == "(Num, 'T)"
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
        TypeErrorKind::RecursiveType(ref ty) if ty.to_string() == "(Num, 'T) -> _"
    );
}

#[test]
fn method_basics() {
    let code = r#"
        foo = 3.plus(4);
        do_something = |x| x - 5;
        bar = foo.do_something();
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let plus_type = FnType::new(vec![Type::NUM; 2].into(), Type::NUM);
    type_env.insert("plus", plus_type);
    type_env.process_statements(&block).unwrap();

    assert_eq!(*type_env.get("bar").unwrap(), Type::NUM);
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
    assert_matches!(err.kind(), TypeErrorKind::UndefinedVar(name) if name == "do_something");
    assert_eq!(
        err.kind().to_string(),
        "Variable `do_something` is not defined"
    );
}

#[test]
fn immediately_invoked_function() {
    let code = "flag = (|x| x + 3)(4) == 7;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(*type_env.get("flag").unwrap(), Type::BOOL);
}

#[test]
fn immediately_invoked_function_with_invalid_arg() {
    let code = "flag = (|x| x + x)(4 == 7);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(*err.span().fragment(), "4 == 7");
    assert_eq!(err.location(), [ErrorLocation::FnArg(0)]);
    assert_matches!(
        err.context(),
        ErrorContext::FnCall { definition, call_signature }
            if definition.to_string() == "for<'T: Ops> ('T) -> 'T"
            && call_signature.to_string() == "(Bool) -> Bool"
    );
    assert_matches!(
        err.kind(),
        TypeErrorKind::FailedConstraint { ty, .. } if *ty == Type::BOOL
    );
    assert_eq!(err.kind().to_string(), "Type `Bool` fails constraint `Ops`");
}

#[test]
fn variable_scoping() {
    let code = "x = 5; y = { x = (1, x); x * (2, 3) };";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        *type_env.get("y").unwrap(),
        Type::Tuple(vec![Type::NUM; 2].into())
    );
}

#[test]
fn destructuring_for_tuple_on_assignment() {
    let code = r#"
        (x, ...ys) = (1, 2, 3);
        (...zs, fn, flag) = (4, 5, 6, |x| x + 3, 1 == 1);
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["x"], Type::NUM);
    assert_eq!(type_env["ys"], Type::slice(Type::NUM, TupleLen::from(2)));
    assert_eq!(type_env["zs"], Type::slice(Type::NUM, TupleLen::from(3)));
    assert_matches!(type_env["fn"], Type::Function(_));
    assert_eq!(type_env["flag"], Type::BOOL);
}

#[test]
fn destructuring_with_unnamed_middle() {
    let code = r#"
        (x, y, ...) = (1, 2, || 3);
        (..., z) = (1 == 1,);
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["x"], Type::NUM);
    assert_eq!(type_env["y"], Type::NUM);
    assert_eq!(type_env["z"], Type::BOOL);
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
        TypeErrorKind::TupleLenMismatch { lhs, rhs, .. }
            if lhs.to_string() == "_ + 2" && *rhs == TupleLen::from(1)
    );
    assert_eq!(
        err.kind().to_string(),
        "Expected a tuple with _ + 2 elements, got one with 1 elements"
    );
}

#[test]
fn destructuring_for_fn_args() {
    let code = r#"
        shift = |shift: Num, ...xs| xs + shift;
        shift(1, 2, 3, 4) == (3, 4, 5);
        shift(1, (2, 3), (4, 5)) == ((3, 4), (5, 6));
        3.shift(5)
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let res = type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["shift"].to_string(),
        "for<'T: Lin> (Num, ...['T; N]) -> ['T; N]"
    );
    assert_eq!(res.to_string(), "(Num)");

    {
        let bogus_code = "shift(1, 2, (3, 4))";
        let bogus_block = F32Grammar::parse_statements(bogus_code).unwrap();
        let err = type_env
            .process_statements(&bogus_block)
            .unwrap_err()
            .single();
        assert_matches!(err.kind(), TypeErrorKind::TypeMismatch(..));
    }
    {
        let bogus_code = "shift(1, 1 == 2, 1 == 1)";
        let bogus_block = F32Grammar::parse_statements(bogus_code).unwrap();
        let err = type_env
            .process_statements(&bogus_block)
            .unwrap_err()
            .single();
        assert_matches!(err.kind(), TypeErrorKind::FailedConstraint { .. });
    }

    let bogus_code = "(x, _, _) = 1.shift(2, 3);";
    let bogus_block = F32Grammar::parse_statements(bogus_code).unwrap();
    let err = type_env
        .process_statements(&bogus_block)
        .unwrap_err()
        .single();
    assert_matches!(
        err.kind(),
        TypeErrorKind::TupleLenMismatch {
            lhs,
            rhs,
            ..
        } if *lhs == TupleLen::from(3) && *rhs == TupleLen::from(2)
    )
}

#[test]
fn fn_args_can_be_unified_with_concrete_length() {
    let code = "weird = |...xs| { (x, y) = xs; x };";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["weird"].to_string(), "('T, 'T) -> 'T");
}

#[test]
fn exact_lengths_for_gathering_fn() {
    let code = r#"
        gather = |...xs| xs;
        (x, y) = gather(1, 2);
        (head, ...) = gather(1 == 1, 1 == 2, 1 == 3);
        (_, ...tail) = gather(4, 5, 6);
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["gather"].to_string(), "(...['T; N]) -> ['T; N]");
    assert_eq!(type_env["x"], Type::NUM);
    assert_eq!(type_env["y"], Type::NUM);
    assert_eq!(type_env["head"], Type::BOOL);
    assert_eq!(type_env["tail"], Type::slice(Type::NUM, TupleLen::from(2)));
}

#[test]
fn inferring_type_from_embedded_function() {
    let code = "double = |x| { (x, || (x, 2 * x)) };";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();
    assert_eq!(
        type_env["double"].to_string(),
        "(Num) -> (Num, () -> (Num, Num))"
    );
}

#[test]
fn free_and_bound_type_vars() {
    let code = r#"
        concat = |x| { |y| (x, y) };
        x = concat(2)(5);
        partial = concat(3);
        y = (partial(2), partial((1, 1)));
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env.get("concat").unwrap().to_string(),
        "('T) -> ('U) -> ('T, 'U)"
    );
    assert_eq!(type_env.get("x").unwrap().to_string(), "(Num, Num)");
    assert_eq!(
        type_env.get("partial").unwrap().to_string(),
        "('U) -> (Num, 'U)"
    );
    assert_eq!(
        type_env.get("y").unwrap().to_string(),
        "((Num, Num), (Num, (Num, Num)))"
    );
}

#[test]
fn attributing_type_vars_to_correct_fn() {
    let code = "double = |x| { (x, || (x, x)) };";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();
    assert_eq!(
        type_env["double"].to_string(),
        "('T) -> ('T, () -> ('T, 'T))"
    );
}

#[test]
fn defining_and_calling_embedded_function() {
    let code = r#"
        call_double = |x| {
            double = |x| (x, x);
            double(x) == (1, 3)
        };
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["call_double"].to_string(), "(Num) -> Bool");
}

#[test]
fn varargs_in_embedded_fn() {
    let code = r#"
        create_sum = |init| |...xs| xs.fold(init, |acc, x| acc + x);
        sum = create_sum(0);
        sum(1, 2, 3, 4) == 10;
        other_sum = create_sum((0, 0));
        other_sum((1, 2), (3, 4)) == (4, 6);
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("fold", Prelude::Fold);
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["create_sum"].to_string(),
        "for<'T: Ops> ('T) -> (...['T; N]) -> 'T"
    );
    assert_eq!(type_env["sum"].to_string(), "(...[Num; N]) -> Num");
    assert_eq!(
        type_env["other_sum"].to_string(),
        "(...[(Num, Num); N]) -> (Num, Num)"
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
        TypeErrorKind::TupleLenMismatch {
            lhs,
            rhs,
            context: TupleLenMismatchContext::Assignment,
        } if *lhs == TupleLen::from(1) && *rhs == TupleLen::from(2)
    );
}

#[test]
fn function_as_arg() {
    let code = "mapper = |(x, y), map| (map(x), map(y));";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();
    assert_eq!(
        type_env["mapper"].to_string(),
        "(('T, 'T), ('T) -> 'U) -> ('U, 'U)"
    );
}

#[test]
fn function_as_arg_with_more_constraints() {
    let code = "mapper = |(x, y), map| map(x) + map(y);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();
    assert_eq!(
        type_env["mapper"].to_string(),
        "for<'U: Ops> (('T, 'T), ('T) -> 'U) -> 'U"
    );
}

#[test]
fn function_as_arg_with_even_more_constraints() {
    let code = "mapper = |(x, y), map| map(x * map(y));";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();
    assert_eq!(
        type_env["mapper"].to_string(),
        "for<'T: Ops> (('T, 'T), ('T) -> 'T) -> 'T"
    );
}

#[test]
fn function_arg_with_multiple_args() {
    let code = "test_fn = |x, fun| fun(x, x * 3);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();
    assert_eq!(
        type_env["test_fn"].to_string(),
        "(Num, (Num, Num) -> 'T) -> 'T"
    );
}

#[test]
fn function_as_arg_within_tuple() {
    let code = r#"
        test_fn = |struct, y| {
            (fn, x) = struct;
            fn(x / 3) * y
        };
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env.get("test_fn").unwrap().to_string(),
        "for<'T: Ops> (((Num) -> 'T, Num), 'T) -> 'T"
    );
}

#[test]
fn function_instantiations_are_independent() {
    let code = r#"
        identity = |x| x;
        x = (identity(5), identity(1 == 2));
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env.get("x").unwrap().to_string(), "(Num, Bool)");
}

#[test]
fn function_passed_as_arg() {
    let code = r#"
        mapper = |(x, y), map| (map(x), map(y));
        tuple = mapper((1, 2), |x| x + 3);
        create_fn = |x| { || x };
        tuple_of_fns = mapper((1, 2), create_fn);
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env.get("tuple").unwrap().to_string(), "(Num, Num)");
    assert_eq!(
        type_env.get("tuple_of_fns").unwrap().to_string(),
        "(() -> Num, () -> Num)"
    );
}

#[test]
fn curried_function_passed_as_arg() {
    let code = r#"
        mapper = |(x, y), map| (map(x), map(y));
        concat = |x| { |y| (x, y) };
        r = mapper((1, 2), concat(1 == 1));
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env.get("r").unwrap().to_string(),
        "((Bool, Num), (Bool, Num))"
    );
}

#[test]
fn parametric_fn_passed_as_arg_with_different_constraints() {
    let code = r#"
        concat = |x| { |y| (x, y) };
        partial = concat(3); // (U) -> (Num, U)

        first = |fun| fun(5);
        r = first(partial); // (Num, Num)
        second = |fun, b| fun(b) == (3, b);
        second(partial, 1 == 1);
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env.get("r").unwrap().to_string(), "(Num, Num)");
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

    assert_matches!(err.kind(), TypeErrorKind::RecursiveType(_));
}

#[test]
fn type_param_is_placed_correctly_with_fn_arg() {
    let code = "foo = |fun| { |x| fun(x) == x };";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();

    type_env.process_statements(&block).unwrap();
    assert_eq!(
        type_env.get("foo").unwrap().to_string(),
        "(('T) -> 'T) -> ('T) -> Bool"
    );
}

#[test]
fn type_params_in_fn_with_multiple_fn_args() {
    let code = "test = |x, foo, bar| foo(x) == bar(x * x);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();

    type_env.process_statements(&block).unwrap();
    assert_eq!(
        type_env.get("test").unwrap().to_string(),
        "for<'T: Ops> ('T, ('T) -> 'U, ('T) -> 'U) -> Bool"
    );
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
    assert_eq!(err.location(), [ErrorLocation::FnArg(1)]);
    let expected_call_signature = "((Num, Num), for<'T: Ops> ('T, 'T) -> 'T) -> (Num, Num)";
    assert_matches!(
        err.context(),
        ErrorContext::FnCall { definition, call_signature }
            if definition.to_string() == "(('T, 'T), ('T) -> 'U) -> ('U, 'U)"
            && call_signature.to_string() == expected_call_signature
    );

    assert_matches!(
        err.kind(),
        TypeErrorKind::TupleLenMismatch {
            lhs,
            rhs,
            context: TupleLenMismatchContext::FnArgs,
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
        TypeErrorKind::TypeMismatch(Type::Tuple(t), rhs)
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
    assert_eq!(
        err.location(),
        [ErrorLocation::FnArg(0), ErrorLocation::TupleElement(1)]
    );
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
fn unifying_slice_and_tuple() {
    let code = "xs = (1, 2).map(|x| x + 5);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::Map);
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["xs"], Type::Tuple(vec![Type::NUM; 2].into()));
}

#[test]
fn function_accepting_slices() {
    let code = "inc = |xs| xs.map(|x| x + 5); z = (1, 2, 3).inc();";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::Map);
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["inc"].to_string(), "([Num; N]) -> [Num; N]");
    assert_eq!(type_env["z"], Type::slice(Type::NUM, TupleLen::from(3)));
}

#[test]
fn incorrect_arg_in_slices() {
    let code = "(1, 2 == 3).map(|x| x);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::Map);

    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(*err.span().fragment(), "2 == 3");
    assert_eq!(
        err.location(),
        [ErrorLocation::FnArg(0), ErrorLocation::TupleElement(1)]
    );
    assert_matches!(
        err.context(),
        ErrorContext::FnCall { definition, call_signature }
            if definition.to_string() == "(['T; N], ('T) -> 'U) -> ['U; N]"
            && call_signature.to_string() == "((Num, Bool), ('T) -> 'T) -> (Num, Num)"
    );
    assert_incompatible_types(&err.kind(), &Type::NUM, &Type::BOOL);
}

#[test]
fn slice_narrowed_to_tuple() {
    let code = "foo = |xs, fn| { (x, y, _) = xs.map(fn); y - x };";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::Map);
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["foo"].to_string(),
        "for<'U: Ops> (('T, 'T, 'T), ('T) -> 'U) -> 'U"
    );
}

#[test]
fn unifying_length_vars_error() {
    let code = "(1, 2).zip_with((3, 4, 5));";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("zip_with", zip_fn_type());
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(*err.span().fragment(), "(3, 4, 5)");
    assert_eq!(err.location(), [ErrorLocation::FnArg(1)]);
    assert_matches!(
        err.context(),
        ErrorContext::FnCall { definition, call_signature }
            if *definition == type_env["zip_with"]
            && call_signature.to_string().starts_with("((Num, Num), (Num, Num, Num)) ->")
    );

    assert_matches!(
        err.kind(),
        TypeErrorKind::TupleLenMismatch {
            lhs,
            rhs,
            context: TupleLenMismatchContext::Assignment,
        } if *lhs == TupleLen::from(2) && *rhs == TupleLen::from(3)
    );
}

#[test]
fn unifying_length_vars() {
    let code = "foo = |xs, ys| xs.zip_with(ys).map(|(x, y)| x + y);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::Map);
    type_env.insert("zip_with", zip_fn_type());
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["foo"].to_string(),
        "for<len! N; 'T: Ops> (['T; N], ['T; N]) -> ['T; N]"
    );
}

#[test]
fn dynamically_sized_slices_basics() {
    let code = "filtered = (1, 2, 3).filter(|x| x != 1);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("filter", Prelude::Filter);
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["filtered"].to_string(), "[Num]");

    let new_code = r#"
        mapped = filtered.map(|x| x + 3);
        sum = mapped.fold(0, |acc, x| acc + x);
    "#;
    type_env
        .insert("map", Prelude::Map)
        .insert("fold", Prelude::Fold);
    for line in new_code.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let block = F32Grammar::parse_statements(line).unwrap();
        type_env.process_statements(&block).unwrap();
    }

    assert_eq!(type_env["mapped"], type_env["filtered"]);
    assert_eq!(type_env["sum"], Type::NUM);
}

#[test]
fn dynamically_sized_slices_with_map() {
    let code = r#"
        foo = |xs| xs.filter(|x| x != 1).map(|x| x / 2);
        // `foo` must be callable both with tuples and dynamically sized slices.
        (1, 2, 3).foo();
        (5, 6, 7).filter(|x| x != 6 && x != 7).foo();
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("filter", Prelude::Filter);
    type_env.insert("map", Prelude::Map);
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["foo"].to_string(), "([Num; N]) -> [Num]");
}

#[test]
fn mix_of_static_and_dynamic_slices() {
    let code = "|xs| { _unused: [_] = xs; xs + (1, 2) }";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let output = type_env.process_statements(&block).unwrap();
    assert_eq!(output.to_string(), "((Num, Num)) -> (Num, Num)");
}

#[test]
fn mix_of_static_and_dynamic_slices_via_fn() {
    let code = "|xs| { xs.filter(|x| x != 1); xs + (1, 2) }";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let output = type_env
        .insert("filter", Prelude::Filter)
        .process_statements(&block)
        .unwrap();
    assert_eq!(output.to_string(), "((Num, Num)) -> (Num, Num)");
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
        TypeErrorKind::TupleLenMismatch { lhs, .. } if *lhs == TupleLen::from(2)
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
    assert_matches!(err.kind(), TypeErrorKind::UnsupportedFeature(_));
    assert_eq!(
        err.kind().to_string(),
        "Unsupported binary op: greater comparison"
    );
}

#[test]
fn comparisons_when_switched_on() {
    let code = "(1, 2, 3).filter(|x| x > 1)";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("filter", Prelude::Filter);

    let output = type_env
        .process_with_arithmetic(&NumArithmetic::with_comparisons(), &block)
        .unwrap();
    let slice = match &output {
        Type::Tuple(tuple) => tuple.as_slice().unwrap(),
        _ => panic!("Unexpected output: {:?}", output),
    };

    assert_eq!(*slice.element(), Type::NUM);
    assert_matches!(slice.len().components(), (Some(UnknownLen::Dynamic), 0));
}

#[test]
fn comparison_type_errors() {
    let code = "(2 <= 1) + 3";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env
        .process_with_arithmetic(&NumArithmetic::with_comparisons(), &block)
        .unwrap_err()
        .single();

    assert_eq!(*err.span().fragment(), "(2 <= 1)");
    assert_eq!(err.location(), [ErrorLocation::Lhs]);
    assert_matches!(err.context(), ErrorContext::BinaryOp(_));
    assert_matches!(
        err.kind(),
        TypeErrorKind::FailedConstraint { ty, .. } if *ty == Type::BOOL
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
        TypeErrorKind::FailedConstraint {
            ty,
            constraint: NumConstraints::Ops,
        } if *ty == Type::BOOL
    );
}

#[test]
fn constraint_passed_to_wrapping_fn() {
    let code = "add = |x, y| x + y; |x| add(x, x)";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let double_fn = type_env
        .process_with_arithmetic(&NumArithmetic::with_comparisons(), &block)
        .unwrap();

    assert_eq!(double_fn.to_string(), "for<'T: Ops> ('T) -> 'T");
}

#[test]
fn any_can_be_unified_with_anything() {
    let code = r#"
        any == 5 && any == (2, 3);
        any(2, (3, 5)) == any(7);
        any + 2;
        any * (3, 5);
        any.map(|x| x + 3) == (2, 4, 5); // could be valid
        any.map(|x| x + 3) == (1 == 2); // cannot be valid; the output is a slice of numbers
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env
        .insert("any", Type::any())
        .insert("map", Prelude::Map)
        .process_statements(&block)
        .unwrap_err()
        .single();

    let lhs_slice = match err.kind() {
        TypeErrorKind::TypeMismatch(Type::Tuple(tuple), bool) if *bool == Type::BOOL => {
            tuple.as_slice().unwrap()
        }
        _ => panic!("Unexpected error: {:?}", err),
    };
    assert_eq!(*lhs_slice.element(), Type::NUM);
}

#[test]
fn any_propagates_via_fn_params() {
    let code = r#"
        identity = |x| x;
        identity(any) == 5 && identity(any) == (2, 3, 1 == 1);
        val = identity(any);
        val == 5 && val == (7, 8) // works: `val: any`
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env
        .insert("any", Type::any())
        .process_statements(&block)
        .unwrap();

    assert_eq!(type_env["val"], Type::any());
}

#[test]
fn any_can_be_copied_and_unified_with_anything() {
    let code = r#"
        any = any_other;
        any == 5 && any == (2, 3);
        any + 3;
        any(2, (3, 5)) == any(7);
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env
        .insert("any_other", Type::any())
        .process_statements(&block)
        .unwrap();
}

#[test]
fn any_can_be_destructured_and_unified_with_anything() {
    let code = r#"
        (any, ...) = some_tuple;
        any == 5 && any == (2, 3);
        any + 3;
        any(2, (3, 5)) == any(7);
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env
        .insert("some_tuple", Type::any().repeat(3))
        .process_statements(&block)
        .unwrap();
}

#[test]
fn unifying_types_containing_any() {
    let code = "digest(1, (2, 3), |x| x + 1)";
    let block = F32Grammar::parse_statements(code).unwrap();

    let digest_type = FnType::builder()
        .with_arg(Type::NUM)
        .with_varargs(Type::any(), UnknownLen::param(0))
        .returning(Type::NUM);
    let mut type_env = TypeEnvironment::new();
    let output = type_env
        .insert("digest", digest_type)
        .process_statements(&block)
        .unwrap();

    assert_eq!(output, Type::NUM);

    let bogus_code = "digest(2 == 3, 5)";
    let bogus_block = F32Grammar::parse_statements(bogus_code).unwrap();
    let err = type_env
        .process_statements(&bogus_block)
        .unwrap_err()
        .single();

    assert_incompatible_types(err.kind(), &Type::NUM, &Type::BOOL);
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
    assert_eq!(err.location(), [ErrorLocation::FnArg(1)]);
    assert_matches!(
        err.context(),
        ErrorContext::FnCall { call_signature, .. }
            if call_signature.to_string() == "(Num, (Num) -> Num) -> Num"
    );

    assert_matches!(
        err.kind(),
        TypeErrorKind::FailedConstraint {
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
        TypeErrorKind::FailedConstraint {
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
        TypeErrorKind::FailedConstraint { ty, .. } if *ty == Type::BOOL
    );
    assert_eq!(err.kind().to_string(), "Type `Bool` fails constraint `Lin`");
    // TODO: store where constraint comes from?
}
