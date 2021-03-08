use super::*;
use crate::{ConstParamDescription, FnArgs, TupleLength, TypeParamDescription};
use std::collections::BTreeMap;

use arithmetic_parser::{
    grammars::{Grammar, NumLiteral, Parse, ParseLiteral, Typed},
    InputSpan, NomResult,
};
use assert_matches::assert_matches;

pub fn assert_incompatible_types(err: &TypeError, first: &ValueType, second: &ValueType) {
    let (x, y) = match err {
        TypeError::IncompatibleTypes(x, y) => (x, y),
        _ => panic!("Unexpected error type: {:?}", err),
    };
    assert!(
        (x == first && y == second) || (x == second && y == first),
        "Unexpected incompatible types: {:?}, expected: {:?}",
        (x, y),
        (first, second)
    );
}

fn hash_fn_type() -> FnType {
    FnType {
        args: FnArgs::Any,
        return_type: ValueType::Number,
        type_params: BTreeMap::new(),
        const_params: BTreeMap::new(),
    }
}

/// `map` function signature:
///
/// ```text
/// fn<const N; T, U>([T; N], fn(T) -> U) -> [U; N]
/// ```
pub fn map_fn_type() -> FnType {
    let map_fn = FnType {
        args: FnArgs::List(vec![ValueType::TypeParam(0)]),
        return_type: ValueType::TypeParam(1),
        type_params: BTreeMap::new(),
        const_params: BTreeMap::new(),
    };

    let param_description = TypeParamDescription {
        maybe_non_linear: true,
    };

    FnType {
        args: FnArgs::List(vec![
            ValueType::Slice {
                element: Box::new(ValueType::TypeParam(0)),
                length: TupleLength::Param(0),
            },
            map_fn.into(),
        ]),
        return_type: ValueType::Slice {
            element: Box::new(ValueType::TypeParam(1)),
            length: TupleLength::Param(0),
        },
        type_params: (0..2).map(|i| (i, param_description)).collect(),
        const_params: vec![(0, ConstParamDescription)].into_iter().collect(),
    }
}

#[test]
fn map_fn_type_display() {
    let map_fn_string = map_fn_type().to_string();
    assert_eq!(
        map_fn_string,
        "fn<const N; T: ?Lin, U: ?Lin>([T; N], fn(T) -> U) -> [U; N]"
    );
}

/// `zip` function signature:
///
/// ```text
/// fn<const N; T, U>([T; N], [U; N]) -> [(T, U); N]
/// ```
fn zip_fn_type() -> FnType {
    let param_description = TypeParamDescription {
        maybe_non_linear: true,
    };

    FnType {
        args: FnArgs::List(vec![
            ValueType::Slice {
                element: Box::new(ValueType::TypeParam(0)),
                length: TupleLength::Param(0),
            },
            ValueType::Slice {
                element: Box::new(ValueType::TypeParam(1)),
                length: TupleLength::Param(0),
            },
        ]),
        return_type: ValueType::Slice {
            element: Box::new(ValueType::Tuple(vec![
                ValueType::TypeParam(0),
                ValueType::TypeParam(1),
            ])),
            length: TupleLength::Param(0),
        },
        type_params: (0..2).map(|i| (i, param_description)).collect(),
        const_params: vec![(0, ConstParamDescription)].into_iter().collect(),
    }
}

#[test]
fn zip_fn_type_display() {
    let zip_fn_string = zip_fn_type().to_string();
    assert_eq!(
        zip_fn_string,
        "fn<const N; T: ?Lin, U: ?Lin>([T; N], [U; N]) -> [(T, U); N]"
    );
}

fn filter_fn_type() -> FnType {
    let filter_fn_type = FnType {
        args: FnArgs::List(vec![ValueType::TypeParam(0)]),
        return_type: ValueType::Bool,
        type_params: BTreeMap::new(),
        const_params: BTreeMap::new(),
    };

    let param_description = TypeParamDescription {
        maybe_non_linear: true,
    };
    FnType {
        args: FnArgs::List(vec![
            ValueType::Slice {
                element: Box::new(ValueType::TypeParam(0)),
                length: TupleLength::Param(0),
            },
            filter_fn_type.into(),
        ]),
        return_type: ValueType::Slice {
            element: Box::new(ValueType::TypeParam(0)),
            length: TupleLength::Dynamic,
        },
        type_params: vec![(0, param_description)].into_iter().collect(),
        const_params: vec![(0, ConstParamDescription)].into_iter().collect(),
    }
}

#[test]
fn filter_fn_type_display() {
    let filter_fn_string = filter_fn_type().to_string();
    assert_eq!(
        filter_fn_string,
        "fn<const N; T: ?Lin>([T; N], fn(T) -> Bool) -> [T]"
    );
}

#[derive(Debug, Clone, Copy)]
pub struct NumGrammar;

impl ParseLiteral for NumGrammar {
    type Lit = f32;

    fn parse_literal(input: InputSpan<'_>) -> NomResult<'_, Self::Lit> {
        f32::parse(input)
    }
}

impl Grammar for NumGrammar {
    type Type = ValueType;

    fn parse_type(input: InputSpan<'_>) -> NomResult<'_, Self::Type> {
        ValueType::parse(input)
    }
}

#[test]
fn statements_with_a_block() {
    let code = "y = { x = 3; 2 * x }; x ^ y == 6 * x;";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();

    let mut type_env = TypeEnvironment::new();
    type_env.insert_type("x", ValueType::Number);
    type_env.process_statements(&block.statements).unwrap();
    assert_eq!(*type_env.get_type("y").unwrap(), ValueType::Number);
}

#[test]
fn boolean_statements() {
    let code = "y = x == x ^ 2; y = y || { x = 3; x != 7 };";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();

    let mut type_env = TypeEnvironment::new();
    type_env.insert_type("x", ValueType::Number);
    type_env.process_statements(&block.statements).unwrap();
    assert_eq!(type_env["y"], ValueType::Bool);
}

#[test]
fn spreading_binary_ops() {
    let code = r#"
        x = 3 * (1, 2);
        y = (1, x, 3) * 4;
    "#;
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block.statements).unwrap();

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

    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert_type("hash", hash_fn_type().into());

    type_env.process_statements(&block.statements).unwrap();
    assert_eq!(
        type_env.get_type("sign").unwrap().to_string(),
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
    let mut type_env = TypeEnvironment::new();

    let hash_type = hash_fn_type();
    type_env.insert_type("hash", ValueType::Function(Box::new(hash_type)));

    type_env.process_statements(&block.statements).unwrap();
    assert_eq!(
        type_env.get_type("compare").unwrap().to_string(),
        "fn<T: ?Lin>(T, T) -> Bool"
    );
    assert_eq!(
        type_env.get_type("compare_hash").unwrap().to_string(),
        "fn<T: ?Lin>(Num, T) -> Bool"
    );
    assert_eq!(
        type_env.get_type("add_hashes").unwrap().to_string(),
        "fn<T: ?Lin, U: ?Lin>(T, U) -> Num"
    );
}

#[test]
fn type_recursion() {
    let code = "bog = |x| x + (x, 2);";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block.statements).unwrap_err();
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
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block.statements).unwrap_err();
    assert_matches!(
        err.extra,
        TypeError::RecursiveType(ref ty) if ty.to_string() == "(Num, T)"
    );
}

#[test]
fn recursion_via_fn() {
    let code = "func = |bog| bog(1, bog);";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block.statements).unwrap_err();
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
    let mut type_env = TypeEnvironment::new();
    let plus_type = FnType::new(
        vec![ValueType::Number, ValueType::Number],
        ValueType::Number,
    );
    type_env.insert_type("plus", plus_type.into());
    type_env.process_statements(&block.statements).unwrap();

    assert_eq!(*type_env.get_type("bar").unwrap(), ValueType::Number);
}

#[test]
fn unknown_method() {
    let code = "bar = 3.do_something();";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block.statements).unwrap_err();

    assert_eq!(*err.fragment(), "do_something");
    assert_matches!(err.extra, TypeError::UndefinedVar(name) if name == "do_something");
}

#[test]
fn immediately_invoked_function() {
    let code = "flag = (|x| x + 3)(4) == 7;";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block.statements).unwrap();

    assert_eq!(*type_env.get_type("flag").unwrap(), ValueType::Bool);
}

#[test]
fn immediately_invoked_function_with_invalid_arg() {
    let code = "flag = (|x| x + x)(4 == 7);";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block.statements).unwrap_err();

    assert_matches!(err.extra, TypeError::NonLinearType(ValueType::Bool));
}

#[test]
fn variable_scoping() {
    let code = "x = 5; y = { x = (1, x); x * (2, 3) };";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block.statements).unwrap();

    assert_eq!(
        *type_env.get_type("y").unwrap(),
        ValueType::Tuple(vec![ValueType::Number, ValueType::Number])
    );
}

#[test]
fn unsupported_destructuring_for_tuple() {
    let code = "(x, ...ys) = (1, 2, 3);";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block.statements).unwrap_err();

    assert_eq!(*err.fragment(), "...ys");
    assert_matches!(err.extra, TypeError::UnsupportedDestructure);
}

#[test]
fn unsupported_destructuring_for_fn_args() {
    let code = "foo = |y, ...xs| xs + y;";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block.statements).unwrap_err();

    assert_eq!(*err.fragment(), "...xs");
    assert_matches!(err.extra, TypeError::UnsupportedDestructure);
}

#[test]
fn inferring_value_type_from_embedded_function() {
    let code = "double = |x| { (x, || (x, 2 * x)) };";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block.statements).unwrap();
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
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block.statements).unwrap();

    assert_eq!(
        type_env.get_type("concat").unwrap().to_string(),
        "fn<T: ?Lin>(T) -> fn<U: ?Lin>(U) -> (T, U)"
    );
    assert_eq!(type_env.get_type("x").unwrap().to_string(), "(Num, Num)");
    assert_eq!(
        type_env.get_type("partial").unwrap().to_string(),
        "fn<U: ?Lin>(U) -> (Num, U)"
    );
    assert_eq!(
        type_env.get_type("y").unwrap().to_string(),
        "((Num, Num), (Num, (Num, Num)))"
    );
}

#[test]
fn attributing_type_vars_to_correct_fn() {
    let code = "double = |x| { (x, || (x, x)) };";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block.statements).unwrap();
    assert_eq!(
        type_env["double"].to_string(),
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
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block.statements).unwrap();
    assert_eq!(type_env["call_double"].to_string(), "fn(Num) -> Bool");
}

#[test]
fn incorrect_function_arity() {
    let code = "double = |x| (x, x); (z,) = double(5);";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block.statements).unwrap_err();

    assert_matches!(
        err.extra,
        TypeError::IncompatibleLengths(TupleLength::Exact(2), TupleLength::Exact(1))
    );
}

#[test]
fn function_as_arg() {
    let code = "mapper = |(x, y), map| (map(x), map(y));";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block.statements).unwrap();
    assert_eq!(
        type_env["mapper"].to_string(),
        "fn<T: ?Lin, U: ?Lin>((T, T), fn(T) -> U) -> (U, U)"
    );
}

#[test]
fn function_as_arg_with_more_constraints() {
    let code = "mapper = |(x, y), map| map(x) + map(y);";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block.statements).unwrap();
    assert_eq!(
        type_env["mapper"].to_string(),
        "fn<T: ?Lin, U>((T, T), fn(T) -> U) -> U"
    );
}

#[test]
fn function_as_arg_with_even_more_constraints() {
    let code = "mapper = |(x, y), map| map(x * map(y));";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block.statements).unwrap();
    assert_eq!(
        type_env["mapper"].to_string(),
        "fn<T>((T, T), fn(T) -> T) -> T"
    );
}

#[test]
fn function_arg_with_multiple_args() {
    let code = "test_fn = |x, fun| fun(x, x * 3);";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block.statements).unwrap();
    assert_eq!(
        type_env["test_fn"].to_string(),
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
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block.statements).unwrap();

    assert_eq!(
        type_env.get_type("test_fn").unwrap().to_string(),
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
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block.statements).unwrap();

    assert_eq!(type_env.get_type("x").unwrap().to_string(), "(Num, Bool)");
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
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block.statements).unwrap();

    assert_eq!(
        type_env.get_type("tuple").unwrap().to_string(),
        "(Num, Num)"
    );
    assert_eq!(
        type_env.get_type("tuple_of_fns").unwrap().to_string(),
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
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block.statements).unwrap();

    assert_eq!(
        type_env.get_type("r").unwrap().to_string(),
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
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block.statements).unwrap();

    assert_eq!(type_env.get_type("r").unwrap().to_string(), "(Num, Num)");
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
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block.statements).unwrap_err();

    assert_incompatible_types(
        &err.extra,
        &ValueType::Number,
        &ValueType::Tuple(vec![ValueType::Number; 2]),
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

    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block.statements).unwrap_err();

    assert_matches!(err.extra, TypeError::RecursiveType(_));
}

#[test]
fn type_param_is_placed_correctly_with_fn_arg() {
    let code = "foo = |fun| { |x| fun(x) == x };";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();

    type_env.process_statements(&block.statements).unwrap();
    assert_eq!(
        type_env.get_type("foo").unwrap().to_string(),
        "fn<T: ?Lin>(fn(T) -> T) -> fn(T) -> Bool"
    );
}

#[test]
fn type_params_in_fn_with_multiple_fn_args() {
    let code = "test = |x, foo, bar| foo(x) == bar(x * x);";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();

    type_env.process_statements(&block.statements).unwrap();
    assert_eq!(
        type_env.get_type("test").unwrap().to_string(),
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
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block.statements).unwrap_err();

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
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block.statements).unwrap_err();

    assert_incompatible_types(
        &err.extra,
        &ValueType::Number,
        &ValueType::Tuple(vec![ValueType::Any; 2]),
    );
}

#[test]
fn function_passed_as_arg_invalid_input() {
    let code = r#"
        mapper = |(x, y), map| (map(x), map(y));
        mapper((1, 2 != 3), |x| x + 2);
    "#;
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block.statements).unwrap_err();

    assert_incompatible_types(&err.extra, &ValueType::Number, &ValueType::Bool);
}

#[test]
fn unifying_slice_and_tuple() {
    let code = "xs = (1, 2).map(|x| x + 5);";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert_type("map", map_fn_type().into());
    type_env.process_statements(&block.statements).unwrap();

    assert_eq!(
        type_env["xs"],
        ValueType::Tuple(vec![ValueType::Number, ValueType::Number])
    );
}

#[test]
fn function_accepting_slices() {
    let code = "inc = |xs| xs.map(|x| x + 5); z = (1, 2, 3).inc();";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert_type("map", map_fn_type().into());
    type_env.process_statements(&block.statements).unwrap();

    assert_eq!(
        type_env["inc"].to_string(),
        "fn<const N>([Num; N]) -> [Num; N]"
    );
    assert_eq!(
        type_env["z"],
        ValueType::Slice {
            element: Box::new(ValueType::Number),
            length: TupleLength::Exact(3)
        }
    );
}

#[test]
fn incorrect_arg_in_slices() {
    let code = "(1, 2 == 3).map(|x| x);";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert_type("map", map_fn_type().into());

    let err = type_env.process_statements(&block.statements).unwrap_err();

    // FIXME: error span is incorrect here; should be `(1, 2 == 3)`
    assert_incompatible_types(&err.extra, &ValueType::Number, &ValueType::Bool);
}

#[test]
fn slice_narrowed_to_tuple() {
    let code = "foo = |xs, fn| { (x, y, _) = xs.map(fn); y - x };";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert_type("map", map_fn_type().into());
    type_env.process_statements(&block.statements).unwrap();

    assert_eq!(
        type_env["foo"].to_string(),
        "fn<T: ?Lin, U>((T, T, T), fn(T) -> U) -> U"
    );
}

#[test]
fn unifying_length_vars_error() {
    let code = "(1, 2).zip_with((3, 4, 5));";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert_type("zip_with", zip_fn_type().into());

    let err = type_env.process_statements(&block.statements).unwrap_err();
    assert_matches!(
        err.extra,
        TypeError::IncompatibleLengths(TupleLength::Exact(2), TupleLength::Exact(3))
    );
}

#[test]
fn unifying_length_vars() {
    let code = "foo = |xs, ys| xs.zip_with(ys).map(|(x, y)| x + y);";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert_type("map", map_fn_type().into());
    type_env.insert_type("zip_with", zip_fn_type().into());
    type_env.process_statements(&block.statements).unwrap();

    assert_eq!(
        type_env["foo"].to_string(),
        "fn<const N; T>([T; N], [T; N]) -> [T; N]"
    );
}

#[test]
fn dynamically_sized_slices_basics() {
    let code = "filtered = (1, 2, 3).filter(|x| x != 1);";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert_type("filter", filter_fn_type().into());
    type_env.process_statements(&block.statements).unwrap();

    assert_eq!(type_env["filtered"].to_string(), "[Num]");
}

#[test]
fn dynamically_sized_slices_with_map() {
    let code = r#"
        foo = |xs| xs.filter(|x| x != 1).map(|x| x / 2);
        // `foo` must be callable both with tuples and dynamically sized slices.
        (1, 2, 3).foo();
        (5, 6, 7).filter(|x| x != 6 && x != 7).foo();
    "#;
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert_type("filter", filter_fn_type().into());
    type_env.insert_type("map", map_fn_type().into());
    type_env.process_statements(&block.statements).unwrap();

    assert_eq!(
        type_env["foo"].to_string(),
        "fn<const N>([Num; N]) -> [Num]"
    );
}

#[test]
fn cannot_destructure_dynamic_slice() {
    let code = "(x, y) = (1, 2, 3).filter(|x| x != 1);";
    let block = Typed::<NumGrammar>::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert_type("filter", filter_fn_type().into());
    let err = type_env.process_statements(&block.statements).unwrap_err();

    assert_matches!(
        err.extra,
        TypeError::IncompatibleLengths(TupleLength::Dynamic, TupleLength::Exact(2))
    );
}
