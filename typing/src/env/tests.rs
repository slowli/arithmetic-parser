use arithmetic_parser::grammars::{NumGrammar, Parse, Typed};
use assert_matches::assert_matches;

use super::*;
use crate::{arith::LinConstraints, Annotated, FnArgs, Num, Prelude, TupleLength};

pub type F32Grammar = Typed<Annotated<NumGrammar<f32>>>;

pub fn assert_incompatible_types<Prim: PrimitiveType>(
    err: &TypeErrorKind<Prim>,
    first: &ValueType<Prim>,
    second: &ValueType<Prim>,
) {
    let (x, y) = match err {
        TypeErrorKind::IncompatibleTypes(x, y) => (x, y),
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
        args: FnArgs::List(vec![].into()), // FIXME
        return_type: ValueType::NUM,
        type_params: Vec::new(),
        len_params: Vec::new(),
    }
}

/// `zip` function signature:
///
/// ```text
/// fn<len N; T, U>([T; N], [U; N]) -> [(T, U); N]
/// ```
pub fn zip_fn_type() -> FnType<Num> {
    FnType::builder()
        .with_len_params(iter::once(0))
        .with_type_params(0..=1)
        .with_arg(ValueType::Param(0).repeat(TupleLength::Param(0)))
        .with_arg(ValueType::Param(1).repeat(TupleLength::Param(0)))
        .returning(ValueType::slice(
            (ValueType::Param(0), ValueType::Param(1)),
            TupleLength::Param(0),
        ))
}

#[test]
fn zip_fn_type_display() {
    let zip_fn_string = zip_fn_type().to_string();
    assert_eq!(
        zip_fn_string,
        "fn<len N; T, U>([T; N], [U; N]) -> [(T, U); N]"
    );
}

#[test]
fn statements_with_a_block() {
    let code = "y = { x = 3; 2 * x }; x ^ y == 6 * x;";
    let block = F32Grammar::parse_statements(code).unwrap();

    let mut type_env: TypeEnvironment<Num> = vec![("x", ValueType::NUM)].into_iter().collect();
    type_env.process_statements(&block).unwrap();
    assert_eq!(*type_env.get("y").unwrap(), ValueType::NUM);
}

#[test]
fn boolean_statements() {
    let code = "y = x == x ^ 2; y = y || { x = 3; x != 7 };";
    let block = F32Grammar::parse_statements(code).unwrap();

    let mut type_env: TypeEnvironment<Num> = vec![("x", ValueType::NUM)].into_iter().collect();
    type_env.process_statements(&block).unwrap();
    assert_eq!(type_env["y"], ValueType::BOOL);
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
    type_env.insert("hash", hash_fn_type().into());

    type_env.process_statements(&block).unwrap();
    assert_eq!(
        type_env.get("sign").unwrap().to_string(),
        "fn<T>(Num, T) -> (Num, Num)"
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
    type_env.insert("hash", hash_fn_type().into());

    type_env.process_statements(&block).unwrap();
    assert_eq!(
        type_env.get("compare").unwrap().to_string(),
        "fn<T>(T, T) -> Bool"
    );
    assert_eq!(
        type_env.get("compare_hash").unwrap().to_string(),
        "fn<T>(Num, T) -> Bool"
    );
    assert_eq!(
        type_env.get("add_hashes").unwrap().to_string(),
        "fn<T, U>(T, U) -> Num"
    );
}

#[test]
fn type_recursion() {
    let code = "bog = |x| x + (x, 2);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err();
    assert_eq!(*err.span().fragment(), "x + (x, 2)");
    assert_matches!(err.kind(), TypeErrorKind::RecursiveType(ref ty) if ty.to_string() == "(T, Num)");
}

#[test]
fn indirect_type_recursion() {
    let code = r#"
        add = |x, y| x + y; // this function is fine
        bog = |x| add(x, (1, x)); // ...but its application is not
    "#;

    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err();
    assert_matches!(
        err.kind(),
        TypeErrorKind::RecursiveType(ref ty) if ty.to_string() == "(Num, T)"
    );
}

#[test]
fn recursion_via_fn() {
    let code = "func = |bog| bog(1, bog);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err();
    assert_matches!(
        err.kind(),
        TypeErrorKind::RecursiveType(ref ty) if ty.to_string() == "fn(Num, T) -> _"
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
    let plus_type = FnType::new(FnArgs::List(vec![ValueType::NUM; 2].into()), ValueType::NUM);
    type_env.insert("plus", plus_type.into());
    type_env.process_statements(&block).unwrap();

    assert_eq!(*type_env.get("bar").unwrap(), ValueType::NUM);
}

#[test]
fn unknown_method() {
    let code = "bar = 3.do_something();";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err();

    assert_eq!(*err.span().fragment(), "do_something");
    assert_matches!(err.kind(), TypeErrorKind::UndefinedVar(name) if name == "do_something");
}

#[test]
fn immediately_invoked_function() {
    let code = "flag = (|x| x + 3)(4) == 7;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(*type_env.get("flag").unwrap(), ValueType::BOOL);
}

#[test]
fn immediately_invoked_function_with_invalid_arg() {
    let code = "flag = (|x| x + x)(4 == 7);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err();

    assert_matches!(
        err.kind(),
        TypeErrorKind::FailedConstraint { ty, .. } if *ty == ValueType::BOOL
    );
}

#[test]
fn variable_scoping() {
    let code = "x = 5; y = { x = (1, x); x * (2, 3) };";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        *type_env.get("y").unwrap(),
        ValueType::Tuple(vec![ValueType::NUM; 2].into())
    );
}

#[test]
fn unsupported_destructuring_for_tuple() {
    let code = "(x, ...ys) = (1, 2, 3);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err();

    assert_eq!(*err.span().fragment(), "...ys");
    assert_matches!(err.kind(), TypeErrorKind::UnsupportedDestructure);
}

#[test]
fn unsupported_destructuring_for_fn_args() {
    let code = "foo = |y, ...xs| xs + y;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err();

    assert_eq!(*err.span().fragment(), "...xs");
    assert_matches!(err.kind(), TypeErrorKind::UnsupportedDestructure);
}

#[test]
fn inferring_value_type_from_embedded_function() {
    let code = "double = |x| { (x, || (x, 2 * x)) };";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();
    assert_eq!(
        type_env["double"].to_string(),
        "fn(Num) -> (Num, fn() -> (Num, Num))"
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
        "fn<T>(T) -> fn<U>(U) -> (T, U)"
    );
    assert_eq!(type_env.get("x").unwrap().to_string(), "(Num, Num)");
    assert_eq!(
        type_env.get("partial").unwrap().to_string(),
        "fn<U>(U) -> (Num, U)"
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
        "fn<T>(T) -> (T, fn() -> (T, T))"
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
    assert_eq!(type_env["call_double"].to_string(), "fn(Num) -> Bool");
}

#[test]
fn incorrect_function_arity() {
    let code = "double = |x| (x, x); (z,) = double(5);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err();

    assert_matches!(
        err.kind(),
        TypeErrorKind::IncompatibleLengths(TupleLength::Exact(1), TupleLength::Exact(2))
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
        "fn<T, U>((T, T), fn(T) -> U) -> (U, U)"
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
        "fn<T, U: Lin>((T, T), fn(T) -> U) -> U"
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
        "fn<T: Lin>((T, T), fn(T) -> T) -> T"
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
        "fn<T>(Num, fn(Num, Num) -> T) -> T"
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
        "fn<T: Lin>((fn(Num) -> T, Num), T) -> T"
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
        "(fn() -> Num, fn() -> Num)"
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
        partial = concat(3); // fn<U>(U) -> (Num, U)

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
        partial = concat(3); // fn<U>(U) -> (Num, U)

        bogus = |fun| fun(1) == 4;
        bogus(partial);
    "#;

    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err();

    assert_incompatible_types(
        &err.kind(),
        &ValueType::NUM,
        &ValueType::Tuple(vec![ValueType::NUM; 2].into()),
    );
}

#[test]
fn parametric_fn_passed_as_arg_with_recursive_requirements() {
    let code = r#"
        concat = |x| { |y| (x, y) };
        partial = concat(3); // fn<U>(U) -> (Num, U)
        bogus = |fun| { |x| fun(x) == x };
        bogus(partial);
    "#;

    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err();

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
        "fn<T>(fn(T) -> T) -> fn(T) -> Bool"
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
        "fn<T: Lin, U>(T, fn(T) -> U, fn(T) -> U) -> Bool"
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
    let err = type_env.process_statements(&block).unwrap_err();

    assert_matches!(
        err.kind(),
        TypeErrorKind::ArgLenMismatch {
            expected: 1,
            actual: 2
        }
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
    let err = type_env.process_statements(&block).unwrap_err();

    assert_incompatible_types(
        &err.kind(),
        &ValueType::NUM,
        &ValueType::Tuple(vec![ValueType::Some; 2].into()),
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
    let err = type_env.process_statements(&block).unwrap_err();

    assert_incompatible_types(&err.kind(), &ValueType::NUM, &ValueType::BOOL);
}

#[test]
fn unifying_slice_and_tuple() {
    let code = "xs = (1, 2).map(|x| x + 5);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::map_type().into());
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["xs"],
        ValueType::Tuple(vec![ValueType::NUM; 2].into())
    );
}

#[test]
fn function_accepting_slices() {
    let code = "inc = |xs| xs.map(|x| x + 5); z = (1, 2, 3).inc();";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::map_type().into());
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["inc"].to_string(),
        "fn<len N>([Num; N]) -> [Num; N]"
    );
    assert_eq!(
        type_env["z"],
        ValueType::slice(ValueType::NUM, TupleLength::Exact(3))
    );
}

#[test]
fn incorrect_arg_in_slices() {
    let code = "(1, 2 == 3).map(|x| x);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::map_type().into());

    let err = type_env.process_statements(&block).unwrap_err();

    // FIXME: error span is incorrect here; should be `(1, 2 == 3)`
    assert_incompatible_types(&err.kind(), &ValueType::NUM, &ValueType::BOOL);
}

#[test]
fn slice_narrowed_to_tuple() {
    let code = "foo = |xs, fn| { (x, y, _) = xs.map(fn); y - x };";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::map_type().into());
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["foo"].to_string(),
        "fn<T, U: Lin>((T, T, T), fn(T) -> U) -> U"
    );
}

#[test]
fn unifying_length_vars_error() {
    let code = "(1, 2).zip_with((3, 4, 5));";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("zip_with", zip_fn_type().into());

    let err = type_env.process_statements(&block).unwrap_err();
    assert_matches!(
        err.kind(),
        TypeErrorKind::IncompatibleLengths(TupleLength::Exact(2), TupleLength::Exact(3))
    );
}

#[test]
fn unifying_length_vars() {
    let code = "foo = |xs, ys| xs.zip_with(ys).map(|(x, y)| x + y);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::map_type().into());
    type_env.insert("zip_with", zip_fn_type().into());
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["foo"].to_string(),
        "fn<len N; T: Lin>([T; N], [T; N]) -> [T; N]"
    );
}

#[test]
fn dynamically_sized_slices_basics() {
    let code = "filtered = (1, 2, 3).filter(|x| x != 1);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("filter", Prelude::filter_type().into());
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["filtered"].to_string(), "[Num; _]");

    let new_code = r#"
        mapped = filtered.map(|x| x + 3);
        filtered + mapped; // check that `map` preserves type
        sum = mapped.fold(0, |acc, x| acc + x);
    "#;
    type_env
        .insert("map", Prelude::map_type().into())
        .insert("fold", Prelude::fold_type().into());
    for line in new_code.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let block = F32Grammar::parse_statements(line).unwrap();
        type_env.process_statements(&block).unwrap();
    }

    assert_eq!(type_env["mapped"], type_env["filtered"]);
    assert_eq!(type_env["sum"], ValueType::NUM);
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
    type_env.insert("filter", Prelude::filter_type().into());
    type_env.insert("map", Prelude::map_type().into());
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["foo"].to_string(),
        "fn<len N, M*>([Num; N]) -> [Num; M]"
    );
}

#[test]
fn cannot_destructure_dynamic_slice() {
    let code = "(x, y) = (1, 2, 3).filter(|x| x != 1);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("filter", Prelude::filter_type().into());
    let err = type_env.process_statements(&block).unwrap_err();

    assert_matches!(
        err.kind(),
        TypeErrorKind::IncompatibleLengths(TupleLength::Exact(2), TupleLength::Var(_))
    );
}

#[test]
fn comparisons_when_switched_off() {
    let code = "(1, 2, 3).filter(|x| x > 1)";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("filter", Prelude::filter_type().into());
    let err = type_env.process_statements(&block).unwrap_err();

    assert_eq!(*err.span().fragment(), ">");
    assert_matches!(err.kind(), TypeErrorKind::Unsupported(_));
}

#[test]
fn comparisons_when_switched_on() {
    let code = "(1, 2, 3).filter(|x| x > 1)";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("filter", Prelude::filter_type().into());

    let output = type_env
        .process_with_arithmetic(&NumArithmetic::with_comparisons(), &block)
        .unwrap();
    let slice = match &output {
        ValueType::Tuple(tuple) => tuple.as_slice().unwrap(),
        _ => panic!("Unexpected output: {:?}", output),
    };

    assert_eq!(*slice.element(), ValueType::NUM);
    assert_matches!(slice.len(), TupleLength::Var(_));
}

#[test]
fn comparison_type_error() {
    let code = "(2 <= 1) + 3";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env
        .process_with_arithmetic(&NumArithmetic::with_comparisons(), &block)
        .unwrap_err();

    assert_matches!(
        err.kind(),
        TypeErrorKind::FailedConstraint { ty, .. } if *ty == ValueType::BOOL
    );
}

#[test]
fn constraint_error() {
    let code = "add = |x, y| x + y; add(1 == 2, 1 == 3)";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env
        .process_with_arithmetic(&NumArithmetic::with_comparisons(), &block)
        .unwrap_err();

    assert_matches!(
        err.kind(),
        TypeErrorKind::FailedConstraint {
            ty,
            constraint: LinConstraints::LIN,
        } if *ty == ValueType::BOOL
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

    assert_eq!(double_fn.to_string(), "fn<T: Lin>(T) -> T");
}
