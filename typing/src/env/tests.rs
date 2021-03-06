use super::*;
use crate::FnArgs;
use std::collections::BTreeMap;

use arithmetic_parser::{
    grammars::{Grammar, NumLiteral, Parse, ParseLiteral, Typed},
    InputSpan, NomResult,
};
use assert_matches::assert_matches;

fn hash_fn_type() -> FnType {
    FnType {
        args: FnArgs::Any,
        return_type: ValueType::Number,
        type_params: BTreeMap::new(),
    }
}

#[derive(Debug, Clone, Copy)]
struct NumGrammar;

impl ParseLiteral for NumGrammar {
    type Lit = f32;

    fn parse_literal(input: InputSpan<'_>) -> NomResult<'_, Self::Lit> {
        f32::parse(input)
    }
}

impl Grammar for NumGrammar {
    type Type = ValueType;

    fn parse_type(_input: InputSpan<'_>) -> NomResult<'_, Self::Type> {
        unimplemented!()
    }
}

#[test]
fn statements_with_a_block() {
    let code = "y = { x = 3; 2 * x }; x ^ y == 6 * x;";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();

    let mut type_context = TypeEnvironment::new();
    type_context.insert_type("x", ValueType::Number);
    type_context.process_statements(&block.statements).unwrap();
    assert_eq!(*type_context.get_type("y").unwrap(), ValueType::Number);
}

#[test]
fn boolean_statements() {
    let code = "y = x == x ^ 2; y = y || { x = 3; x != 7 };";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();

    let mut type_context = TypeEnvironment::new();
    type_context.insert_type("x", ValueType::Number);
    type_context.process_statements(&block.statements).unwrap();
    assert_eq!(*type_context.get_type("y").unwrap(), ValueType::Bool);
}

#[test]
fn function_definition() {
    // FIXME: revert to single `hash()`
    let code = r#"
        sign = |x, msg| {
            (r, R) = (hash(), hash()) * (1, 3);
            c = hash(R, msg);
            (R, r + c * x)
        };
    "#;

    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();

    let hash_type = hash_fn_type();
    type_context.insert_type("hash", ValueType::Function(Box::new(hash_type)));

    type_context.process_statements(&block.statements).unwrap();
    assert_eq!(
        type_context.get_type("sign").unwrap().to_string(),
        "fn<T: ?Lin>(Num, T) -> (Num, Num)"
    );
}

#[test]
fn non_linear_types_in_function() {
    let code = r#"
        compare = |x, y| x == y;
        compare_hash = |x, z| x == 2 ^ hash(z);
        add_hashes = |x, y| hash(x, y) + hash(y, x);
    "#;

    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();

    let hash_type = hash_fn_type();
    type_context.insert_type("hash", ValueType::Function(Box::new(hash_type)));

    type_context.process_statements(&block.statements).unwrap();
    assert_eq!(
        type_context.get_type("compare").unwrap().to_string(),
        "fn<T: ?Lin>(T, T) -> Bool"
    );
    assert_eq!(
        type_context.get_type("compare_hash").unwrap().to_string(),
        "fn<T: ?Lin>(Num, T) -> Bool"
    );
    assert_eq!(
        type_context.get_type("add_hashes").unwrap().to_string(),
        "fn<T: ?Lin, U: ?Lin>(T, U) -> Num"
    );
}

#[test]
fn type_recursion() {
    let code = "bog = |x| x + (x, 2);";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    let err = type_context
        .process_statements(&block.statements)
        .unwrap_err();
    assert_eq!(*err.fragment(), "x + (x, 2)");
    assert_matches!(err.extra, TypeError::RecursiveType(ref ty) if ty.to_string() == "(T, Num)");
}

#[test]
fn indirect_type_recursion() {
    let code = r#"
        add = |x, y| x + y; // this function is fine
        bog = |x| add(x, (1, x)); // ...but its application is not
    "#;

    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    let err = type_context
        .process_statements(&block.statements)
        .unwrap_err();
    assert_matches!(
        err.extra,
        TypeError::RecursiveType(ref ty) if ty.to_string() == "(Num, T)"
    );
}

#[test]
fn recursion_via_fn() {
    let code = "func = |bog| bog(1, bog);";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    let err = type_context
        .process_statements(&block.statements)
        .unwrap_err();
    assert_matches!(
        err.extra,
        TypeError::RecursiveType(ref ty) if ty.to_string() == "fn(Num, T) -> _"
    );
}

#[test]
fn method_basics() {
    let code = r#"
        foo = 3.plus(4);
        do_something = |x| x - 5;
        bar = foo.do_something();
    "#;
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    let plus_type = FnType::new(
        vec![ValueType::Number, ValueType::Number],
        ValueType::Number,
    );
    type_context.insert_type("plus", plus_type.into());
    type_context.process_statements(&block.statements).unwrap();

    assert_eq!(*type_context.get_type("bar").unwrap(), ValueType::Number);
}

#[test]
fn unknown_method() {
    let code = "bar = 3.do_something();";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    let err = type_context
        .process_statements(&block.statements)
        .unwrap_err();

    assert_eq!(*err.fragment(), "do_something");
    assert_matches!(err.extra, TypeError::UndefinedVar(name) if name == "do_something");
}

#[test]
fn immediately_invoked_function() {
    let code = "flag = (|x| x + 3)(4) == 7;";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    type_context.process_statements(&block.statements).unwrap();

    assert_eq!(*type_context.get_type("flag").unwrap(), ValueType::Bool);
}

#[test]
fn immediately_invoked_function_with_invalid_arg() {
    let code = "flag = (|x| x + x)(4 == 7);";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    let err = type_context
        .process_statements(&block.statements)
        .unwrap_err();

    assert_matches!(err.extra, TypeError::NonLinearType(ValueType::Bool));
}

#[test]
fn variable_scoping() {
    let code = "x = 5; y = { x = (1, x); x * (2, 3) };";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    type_context.process_statements(&block.statements).unwrap();

    assert_eq!(
        *type_context.get_type("y").unwrap(),
        ValueType::Tuple(vec![ValueType::Number, ValueType::Number])
    );
}

#[test]
fn unsupported_destructuring_for_tuple() {
    let code = "(x, ...ys) = (1, 2, 3);";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    let err = type_context
        .process_statements(&block.statements)
        .unwrap_err();

    assert_eq!(*err.fragment(), "...ys");
    assert_matches!(err.extra, TypeError::UnsupportedDestructure);
}

#[test]
fn unsupported_destructuring_for_fn_args() {
    let code = "foo = |y, ...xs| xs + y;";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    let err = type_context
        .process_statements(&block.statements)
        .unwrap_err();

    assert_eq!(*err.fragment(), "...xs");
    assert_matches!(err.extra, TypeError::UnsupportedDestructure);
}

#[test]
fn inferring_value_type_from_embedded_function() {
    let code = "double = |x| { (x, || (x, 2 * x)) };";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    type_context.process_statements(&block.statements).unwrap();
    assert_eq!(
        type_context["double"].to_string(),
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
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    type_context.process_statements(&block.statements).unwrap();

    assert_eq!(
        type_context.get_type("concat").unwrap().to_string(),
        "fn<T: ?Lin>(T) -> fn<U: ?Lin>(U) -> (T, U)"
    );
    assert_eq!(
        type_context.get_type("x").unwrap().to_string(),
        "(Num, Num)"
    );
    assert_eq!(
        type_context.get_type("partial").unwrap().to_string(),
        "fn<U: ?Lin>(U) -> (Num, U)"
    );
    assert_eq!(
        type_context.get_type("y").unwrap().to_string(),
        "((Num, Num), (Num, (Num, Num)))"
    );
}

#[test]
fn attributing_type_vars_to_correct_fn() {
    let code = "double = |x| { (x, || (x, x)) };";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    type_context.process_statements(&block.statements).unwrap();
    assert_eq!(
        type_context["double"].to_string(),
        "fn<T: ?Lin>(T) -> (T, fn() -> (T, T))"
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
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    type_context.process_statements(&block.statements).unwrap();
    assert_eq!(type_context["call_double"].to_string(), "fn(Num) -> Bool");
}

#[test]
fn incorrect_function_arity() {
    let code = "double = |x| (x, x); (z,) = double(5);";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    let err = type_context
        .process_statements(&block.statements)
        .unwrap_err();
    assert_matches!(err.extra, TypeError::TupleLenMismatch(2, 1));
}

#[test]
fn function_as_arg() {
    let code = "mapper = |(x, y), map| (map(x), map(y));";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    type_context.process_statements(&block.statements).unwrap();
    assert_eq!(
        type_context["mapper"].to_string(),
        "fn<T: ?Lin, U: ?Lin>((T, T), fn(T) -> U) -> (U, U)"
    );
}

#[test]
fn function_as_arg_with_more_constraints() {
    let code = "mapper = |(x, y), map| map(x) + map(y);";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    type_context.process_statements(&block.statements).unwrap();
    assert_eq!(
        type_context["mapper"].to_string(),
        "fn<T: ?Lin, U>((T, T), fn(T) -> U) -> U"
    );
}

#[test]
fn function_as_arg_with_even_more_constraints() {
    let code = "mapper = |(x, y), map| map(x * map(y));";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    type_context.process_statements(&block.statements).unwrap();
    assert_eq!(
        type_context["mapper"].to_string(),
        "fn<T>((T, T), fn(T) -> T) -> T"
    );
}

#[test]
fn function_arg_with_multiple_args() {
    let code = "test_fn = |x, fun| fun(x, x * 3);";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    type_context.process_statements(&block.statements).unwrap();
    assert_eq!(
        type_context["test_fn"].to_string(),
        "fn<T: ?Lin>(Num, fn(Num, Num) -> T) -> T"
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
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    type_context.process_statements(&block.statements).unwrap();

    assert_eq!(
        type_context.get_type("test_fn").unwrap().to_string(),
        "fn<T>((fn(Num) -> T, Num), T) -> T"
    );
}

#[test]
fn function_instantiations_are_independent() {
    let code = r#"
        identity = |x| x;
        x = (identity(5), identity(1 == 2));
    "#;
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    type_context.process_statements(&block.statements).unwrap();

    assert_eq!(
        type_context.get_type("x").unwrap().to_string(),
        "(Num, Bool)"
    );
}

#[test]
fn function_passed_as_arg() {
    let code = r#"
        mapper = |(x, y), map| (map(x), map(y));
        tuple = mapper((1, 2), |x| x + 3);
        create_fn = |x| { || x };
        tuple_of_fns = mapper((1, 2), create_fn);
    "#;
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    type_context.process_statements(&block.statements).unwrap();

    assert_eq!(
        type_context.get_type("tuple").unwrap().to_string(),
        "(Num, Num)"
    );
    assert_eq!(
        type_context.get_type("tuple_of_fns").unwrap().to_string(),
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
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    type_context.process_statements(&block.statements).unwrap();

    assert_eq!(
        type_context.get_type("r").unwrap().to_string(),
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
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    type_context.process_statements(&block.statements).unwrap();

    assert_eq!(
        type_context.get_type("r").unwrap().to_string(),
        "(Num, Num)"
    );
}

#[test]
fn parametric_fn_passed_as_arg_with_unsatisfiable_requirements() {
    let code = r#"
        concat = |x| { |y| (x, y) };
        partial = concat(3); // fn<U>(U) -> (Num, U)

        bogus = |fun| fun(1) == 4;
        bogus(partial);
    "#;

    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    let err = type_context
        .process_statements(&block.statements)
        .unwrap_err();

    assert_matches!(err.extra, TypeError::IncompatibleTypes(_, _));
}

#[test]
fn parametric_fn_passed_as_arg_with_recursive_requirements() {
    let code = r#"
        concat = |x| { |y| (x, y) };
        partial = concat(3); // fn<U>(U) -> (Num, U)
        bogus = |fun| { |x| fun(x) == x };
        bogus(partial);
    "#;

    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    let err = type_context
        .process_statements(&block.statements)
        .unwrap_err();

    assert_matches!(err.extra, TypeError::RecursiveType(_));
}

#[test]
fn type_param_is_placed_correctly_with_fn_arg() {
    let code = "foo = |fun| { |x| fun(x) == x };";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();

    type_context.process_statements(&block.statements).unwrap();
    assert_eq!(
        type_context.get_type("foo").unwrap().to_string(),
        "fn<T: ?Lin>(fn(T) -> T) -> fn(T) -> Bool"
    );
}

#[test]
fn type_params_in_fn_with_multiple_fn_args() {
    let code = "test = |x, foo, bar| foo(x) == bar(x * x);";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();

    type_context.process_statements(&block.statements).unwrap();
    assert_eq!(
        type_context.get_type("test").unwrap().to_string(),
        "fn<T, U: ?Lin>(T, fn(T) -> U, fn(T) -> U) -> Bool"
    );
}

#[test]
fn function_passed_as_arg_invalid_arity() {
    let code = r#"
        mapper = |(x, y), map| (map(x), map(y));
        mapper((1, 2), |x, y| x + y);
    "#;
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    let err = type_context
        .process_statements(&block.statements)
        .unwrap_err();

    assert_matches!(
        err.extra,
        TypeError::ArgLenMismatch {
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
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    let err = type_context
        .process_statements(&block.statements)
        .unwrap_err();

    assert_matches!(err.extra, TypeError::IncompatibleTypes(..));
}

#[test]
fn function_passed_as_arg_invalid_input() {
    let code = r#"
        mapper = |(x, y), map| (map(x), map(y));
        mapper((1, 2 != 3), |x| x + 2);
    "#;
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_context = TypeEnvironment::new();
    let err = type_context
        .process_statements(&block.statements)
        .unwrap_err();

    assert_matches!(err.extra, TypeError::IncompatibleTypes(..));
}
