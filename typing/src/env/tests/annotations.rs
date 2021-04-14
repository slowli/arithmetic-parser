//! Tests with explicit type annotations.

use assert_matches::assert_matches;

use std::collections::HashSet;

use super::{assert_incompatible_types, zip_fn_type, F32Grammar};
use crate::{
    arith::NumArithmetic,
    ast::AstConversionError,
    error::{ErrorContext, ErrorKind, ErrorLocation, TupleLenMismatchContext},
    Assertions, Prelude, TupleLen, Type, TypeEnvironment, UnknownLen,
};
use arithmetic_parser::grammars::Parse;

#[test]
fn type_hint_within_tuple() {
    let code = "foo = |x, fun| { (y: Num, z) = x; fun(y + z) };";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["foo"].to_string(),
        "((Num, Num), (Num) -> 'T) -> 'T"
    );
}

#[test]
fn type_hint_in_fn_arg() {
    let code = r#"
        foo = |tuple: (Num, _), fun| {
            (x, flag) = tuple;
            flag && fun() == x
        };
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["foo"].to_string(),
        "((Num, Bool), () -> Num) -> Bool"
    );
}

#[test]
fn contradicting_type_hint() {
    let code = "x: (Num, _) = (1, 2, 3);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_matches!(
        err.kind(),
        ErrorKind::TupleLenMismatch {
            lhs,
            rhs,
            context: TupleLenMismatchContext::Assignment,
        } if *lhs == TupleLen::from(2) && *rhs == TupleLen::from(3)
    );
}

#[test]
fn valid_type_hint_with_slice() {
    let code = "x: [Num; _] = (1, 2, 3);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["x"].to_string(), "(Num, Num, Num)");
}

#[test]
fn contradicting_type_hint_with_slice() {
    let code = "x: [Num; _] = (1, 2 == 3);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_incompatible_types(&err.kind(), &Type::NUM, &Type::BOOL);
}

#[test]
fn valid_type_hint_with_fn_arg() {
    let code = "foo = |xs, map_fn: (Num) -> _| xs.map(map_fn);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::Map);
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["foo"].to_string(),
        "([Num; N], (Num) -> 'T) -> ['T; N]"
    );
}

#[test]
fn invalid_type_hint_with_fn_arg() {
    let code = "foo = |xs, map_fn: (_, _) -> _| xs.map(map_fn);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::Map);
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_matches!(
        err.kind(),
        ErrorKind::TupleLenMismatch {
            lhs,
            rhs,
            context: TupleLenMismatchContext::FnArgs,
        } if *lhs == TupleLen::from(2) && *rhs == TupleLen::from(1)
    );
}

#[test]
fn valid_type_hint_with_fn_declaration() {
    let code = "foo: (Num) -> _ = |x| x + 3;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::Map);
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["foo"].to_string(), "(Num) -> Num");
}

#[test]
fn invalid_type_hint_with_fn_declaration() {
    let code = "foo: (_) -> Bool = |x| x + 3;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::Map);
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_incompatible_types(&err.kind(), &Type::NUM, &Type::BOOL);
}

#[test]
fn widening_type_hint_with_generic_slice_arg() {
    // Without a type annotation on `xs` it would be interpreted as a number.
    let code = "foo = |xs: [_; _]| xs + 1;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::Map);
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["foo"].to_string(),
        "for<'T: Lin> (['T; N]) -> ['T; N]"
    );
}

#[test]
fn widening_type_hint_with_slice_arg() {
    // Without a type annotation on `xs` it would be interpreted as a number.
    let code = "foo = |xs: [Num; _]| xs + 1;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::Map);
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["foo"].to_string(), "([Num; N]) -> [Num; N]");
}

#[test]
fn unsupported_type_param_in_generic_fn() {
    let code = "identity: (('Arg,)) -> ('Arg,) = |x| x;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert!(err.location().is_empty());
    assert_matches!(err.context(), ErrorContext::Assignment { .. });
    assert_eq!(*err.span().fragment(), "(('Arg,)) -> ('Arg,)");
    assert_matches!(err.kind(), ErrorKind::UnsupportedParam);
}

#[test]
fn unsupported_type_param_location() {
    let code_samples = &[
        "identity: (_, (('Arg,)) -> ('Arg,)) = (3, |x| x);",
        "(_, identity: (('Arg,)) -> ('Arg,)) = (3, |x| x);",
    ];

    for &code in code_samples {
        let block = F32Grammar::parse_statements(code).unwrap();
        let mut type_env = TypeEnvironment::new();
        let err = type_env.process_statements(&block).unwrap_err().single();

        assert_eq!(err.location(), [ErrorLocation::TupleElement(1)]);
        assert_matches!(err.context(), ErrorContext::Assignment { .. });
        assert_eq!(*err.span().fragment(), "(('Arg,)) -> ('Arg,)");
        assert_matches!(err.kind(), ErrorKind::UnsupportedParam);
    }
}

#[test]
fn unsupported_const_param_in_generic_fn() {
    let code = "identity: ([Num; N]) -> [Num; N] = |x| x;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(*err.span().fragment(), "([Num; N]) -> [Num; N]");
    assert_matches!(err.kind(), ErrorKind::UnsupportedParam);
}

#[test]
fn fn_narrowed_via_type_hint() {
    let code = r#"
        identity: (Num) -> _ = |x| x;
        identity((1, 2));
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(type_env["identity"].to_string(), "(Num) -> Num");
    assert_incompatible_types(
        &err.kind(),
        &Type::NUM,
        &Type::Tuple(vec![Type::NUM; 2].into()),
    )
}

#[test]
fn fn_incorrectly_narrowed_via_type_hint() {
    let code = "identity: (Num) -> Bool = |x| x;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_incompatible_types(&err.kind(), &Type::NUM, &Type::BOOL);
}

#[test]
fn fn_instantiated_via_type_hint() {
    let code = r#"
        identity: (_) -> _ = |x| x;
        identity(5) == 5;
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["identity"].to_string(), "(Num) -> Num");
    assert!(type_env["identity"].is_concrete());
}

#[test]
fn assigning_to_dynamically_sized_slice() {
    let code = "slice: [Num] = (1, 2, 3);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["slice"].to_string(), "[Num]");
    assert!(type_env["slice"].is_concrete());

    let bogus_code = "(x, y) = slice;";
    let bogus_block = F32Grammar::parse_statements(bogus_code).unwrap();
    let err = type_env
        .process_statements(&bogus_block)
        .unwrap_err()
        .single();

    assert_matches!(err.kind(), ErrorKind::TupleLenMismatch { .. });
}

#[test]
fn assigning_to_a_slice_and_then_narrowing() {
    let code = r#"
        // The arg type annotation is required because otherwise `xs` type will be set
        // to `[Num]` by unifying it with the type var.
        slice_fn = |xs| {
            _unused: [Num] = xs;
            (x, y, z) = xs;
            x + y * z
        };
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["slice_fn"].to_string(), "((Num, Num, Num)) -> Num");
}

#[test]
fn adding_dynamically_typed_slices() {
    let code = r#"
        x: [Num] = (1, 2);
        y: [Num] = (3, 4, 5);
        x + y
    "#;

    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let errors: Vec<_> = type_env
        .process_statements(&block)
        .unwrap_err()
        .into_iter()
        .collect();

    assert_eq!(errors.len(), 2);
    assert_eq!(*errors[0].span().fragment(), "x");
    assert_matches!(errors[0].kind(), ErrorKind::DynamicLen(_));
    assert_eq!(*errors[1].span().fragment(), "y");
    assert_matches!(errors[1].kind(), ErrorKind::DynamicLen(_));
}

#[test]
fn unifying_dynamic_slices_error() {
    let code = r#"
        x: [Num] = (1, 2);
        y: [Num] = (3, 4, 5);
        x.zip_with(y)
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("zip_with", zip_fn_type());
    let errors: Vec<_> = type_env
        .process_statements(&block)
        .unwrap_err()
        .into_iter()
        .collect();

    assert_eq!(errors.len(), 2);
    assert_matches!(errors[0].kind(), ErrorKind::DynamicLen(_));
    assert_matches!(errors[1].kind(), ErrorKind::DynamicLen(_));
}

#[test]
fn unifying_tuples_with_middle() {
    let code = r#"
        xs: (Num, ...[Num; _]) = (1, 2, 3);
        ys: (...[_; _], _) = (4, 5, |x| x);
        zs: (_, ...[Num], _) = (6, 7, 8, 9);

        // Check basic destructuring.
        (...xs_head: Num, _) = xs;
        (_, _, _: (_) -> _) = ys;
        (_, ...zs_middle, _) = zs;
    "#;

    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["xs"].to_string(), "(Num, Num, Num)");
    assert_eq!(type_env["ys"].to_string(), "(Num, Num, ('T) -> 'T)");
    assert_eq!(type_env["zs"].to_string(), "(Num, ...[Num], Num)");
    assert_eq!(type_env["xs_head"].to_string(), "(Num, Num)");
    assert_eq!(type_env["zs_middle"].to_string(), "[Num]");
}

#[test]
fn unifying_tuples_with_dyn_lengths() {
    let code = r#"
        xs: (_, ...[_], _) = (true, 1, 2, 3, 4);
        (_, _, ...ys) = xs; // should work
        zs: (...[_; _], _, Num) = xs; // should not work (Bool and Num cannot be unified)
    "#;

    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("true", Type::BOOL);
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(*err.span().fragment(), "(...[_; _], _, Num)");
    assert_eq!(err.location(), [ErrorLocation::TupleElement(3)]); // FIXME
    assert_matches!(
        err.context(),
        ErrorContext::Assignment { lhs, rhs }
            if lhs.to_string() == "(...[Num], Num, Num)" && *rhs == type_env["xs"]
    );
    assert_incompatible_types(err.kind(), &Type::BOOL, &Type::NUM);

    assert_eq!(type_env["xs"].to_string(), "(Bool, ...[Num], Num)");
    assert_eq!(type_env["ys"].to_string(), "[Num]");
}

#[test]
fn fn_with_varargs() {
    let code = r#"
        sum = |...xs: Num| xs.fold(0, |acc, x| acc + x);
        sum_spec: (Num, Num, Num) -> _ = sum;
        tuple_sum = |init, ...xs: (_, _)| xs.fold((init, init), |acc, x| acc + x);
    "#;

    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("fold", Prelude::Fold);
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["sum"].to_string(), "(...[Num; N]) -> Num");
    assert_eq!(type_env["sum_spec"].to_string(), "(Num, Num, Num) -> Num");
    assert_eq!(
        type_env["tuple_sum"].to_string(),
        "for<'T: Ops> ('T, ...[('T, 'T); N]) -> ('T, 'T)"
    );
}

#[test]
fn any_type() {
    let code = "test: any = 1; test(1, 2) && test == (3, |x| x)";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let output = type_env.process_statements(&block).unwrap();

    assert_eq!(output, Type::BOOL);
}

#[test]
fn type_with_tuple_of_any() {
    let code = r#"
        test: [any; _] = (1, 1 == 2, |x| x + 1);
        (x, y, z) = test;
        test(1) // should fail: slices are not callable
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(*err.span().fragment(), "test(1)");
    assert_matches!(
        err.kind(),
        ErrorKind::TypeMismatch(lhs, rhs)
            if lhs.to_string() == "(Num) -> _" && rhs.to_string() == "(any, any, any)"
    );
    assert_eq!(type_env["x"], Type::any());
    assert_eq!(type_env["y"], Type::any());
    assert_eq!(type_env["z"], Type::any());
}

#[test]
fn type_with_any_fn() {
    let code = r#"
        fun: (any) -> _ = |x| x == 1;
        fun(1 == 1) && fun((1, 2, 3)) && fun(|x| x + 1);
        fun(1, 2) // should fail: function is only callable with 1 arg
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(*err.span().fragment(), "fun(1, 2)");
    assert_matches!(
        err.kind(),
        ErrorKind::TupleLenMismatch { lhs, rhs, .. }
            if *lhs == TupleLen::from(1) && *rhs == TupleLen::from(2)
    );
    assert_eq!(type_env["fun"].to_string(), "(any) -> Bool");
}

#[test]
fn any_fn_with_constraints() {
    let code = r#"
        lin_tuple: [any Lin; _] = (1, (2, 5), 9);
        bogus_tuple: [any Lin; _] = (1, 2, 3 == 4);
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(*err.span().fragment(), "[any Lin; _]");
    assert_matches!(
        err.kind(),
        ErrorKind::FailedConstraint { ty, .. } if *ty == Type::BOOL
    );
    assert_eq!(
        type_env["lin_tuple"].to_string(),
        "(any Lin, any Lin, any Lin)"
    );
}

#[test]
fn mix_of_any_and_specific_types() {
    let code_samples = &[
        "|xs| { _unused: [any; _] = xs; xs + (1, 2) }",
        "|xs| { _unused: [any] = xs; xs + (1, 2) }",
        "|xs| { _unused: [any Lin; _] = xs; xs + (1, 2) }",
        "|xs| { _unused: [any Lin] = xs; xs + (1, 2) }",
    ];

    for &code in code_samples {
        let block = F32Grammar::parse_statements(code).unwrap();
        let mut type_env = TypeEnvironment::new();
        let output = type_env.process_statements(&block).unwrap();
        assert_eq!(output.to_string(), "((Num, Num)) -> (Num, Num)");
    }
}

#[test]
fn mix_of_any_and_specific_types_in_fns() {
    let code = r#"
        accepts_fn = |fn| { _unused: (any) -> any = fn; (fn(2), fn(3)) + (4, 5) };
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["accepts_fn"].to_string(),
        "((any) -> Num) -> (Num, Num)"
    );
}

#[test]
fn annotations_for_fns_with_slices() {
    let code = r#"
        first: (_) -> [_] = |(x, y)| (x,);
        inc: ([_]) -> [_] = |xs: [Num; _]| xs.map(|x| x + 1);
        bogus: ([_]) -> [_] = |(x, y)| (x,);
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::Map);
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(err.span().location_line(), 4);
    assert_matches!(
        err.kind(),
        ErrorKind::TupleLenMismatch { lhs, rhs, .. }
            if *lhs == TupleLen::from(2) && *rhs == TupleLen::from(UnknownLen::Dynamic)
    );

    assert_eq!(type_env["first"].to_string(), "((_, _)) -> [_]");
    assert_eq!(type_env["inc"].to_string(), "([Num]) -> [Num]");
}

#[test]
fn annotations_for_fns_with_slices_in_contravariant_position() {
    let code = r#"
        |fn| {
            _unused: ([_]) -> [_] = fn;
            fn((1, 2)) + (3, 4);
            _unused: ([_]) -> [Num] = fn;
            (x, ...) = fn((1, 2, 3));
            x
        }
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let output = type_env.process_statements(&block).unwrap();

    assert_eq!(output.to_string(), "(([Num]) -> (Num, Num)) -> Num");
}

#[test]
fn recovery_after_bogus_annotations() {
    let code = r#"
        fun: for<'T: Bogus, 'U: Lin> ('T) -> () = |x| assert(x > 1 && x < 10);
        other_fun = |x: 'T| x + 1;
        other_fun((4, 5));
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("assert", Assertions::Assert);
    let errors = type_env
        .process_with_arithmetic(&NumArithmetic::with_comparisons(), &block)
        .unwrap_err();

    let expected_messages = &[
        "2:22: Error instantiating type from annotation: Unknown constraint `Bogus`",
        "2:30: Error instantiating type from annotation: Unused type param `U`",
        "2:14: Params in declared function types are not supported yet",
        "3:25: Error instantiating type from annotation: \
        Type param `T` is not scoped by function definition",
        "4:19: Type `(Num, Num)` is not assignable to type `Num`",
    ];
    let expected_messages: HashSet<_> = expected_messages.iter().copied().collect();
    let actual_messages: Vec<_> = errors.iter().map(ToString::to_string).collect();
    let actual_messages: HashSet<_> = actual_messages.iter().map(String::as_str).collect();
    assert_eq!(actual_messages, expected_messages);
}

#[test]
fn bogus_annotation_in_fn_definition() {
    let code = "|x: Bogus| x + 1";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(*err.span().fragment(), "Bogus");
    assert_matches!(
        err.kind(),
        ErrorKind::AstConversion(AstConversionError::UnknownType(ty))
            if ty == "Bogus"
    );
}
