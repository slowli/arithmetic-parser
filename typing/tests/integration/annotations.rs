//! Tests with explicit type annotations.

use arithmetic_parser::grammars::Parse;
use arithmetic_typing::{
    defs::Prelude,
    error::{ErrorContext, ErrorKind, ErrorPathFragment},
    TupleLen, Type, TypeEnvironment, UnknownLen,
};
use assert_matches::assert_matches;

use crate::{assert_incompatible_types, hash_fn_type, ErrorsExt, F32Grammar, Hashed};

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
    let code = "
        foo = |tuple: (Num, _), fun| {
            (x, flag) = tuple;
            flag && fun() == x
        };
    ";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["foo"].to_string(),
        "((Num, Bool), () -> Num) -> Bool"
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
fn valid_type_hint_with_fn_declaration() {
    let code = "foo: (Num) -> _ = |x| x + 3;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::Map);
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["foo"].to_string(), "(Num) -> Num");
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
fn fn_narrowed_via_type_hint() {
    let code = "
        identity: (Num) -> _ = |x| x;
        identity((1, 2));
    ";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(type_env["identity"].to_string(), "(Num) -> Num");
    assert_incompatible_types(
        err.kind(),
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

    assert_incompatible_types(err.kind(), &Type::NUM, &Type::BOOL);
}

#[test]
fn fn_instantiated_via_type_hint() {
    let code = "
        identity: (_) -> _ = |x| x;
        identity(5) == 5;
    ";
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
    let code = "
        slice_fn = |xs| {
            _unused: [Num] = xs;
            (x, y, z) = xs;
            x + y * z
        };
    ";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["slice_fn"].to_string(), "((Num, Num, Num)) -> Num");
}

#[test]
fn unifying_tuples_with_middle() {
    let code = "
        xs: (Num, ...[Num; _]) = (1, 2, 3);
        ys: (...[_; _], _) = (4, 5, |x| x);
        zs: (_, ...[Num], _) = (6, 7, 8, 9);

        // Check basic destructuring.
        (...xs_head: Num, _) = xs;
        (_, _, _: (_) -> _) = ys;
        (_, ...zs_middle, _) = zs;
    ";

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
    let code = "
        xs: (_, ...[_], _) = (true, 1, 2, 3, 4);
        (_, _, ...ys) = xs; // should work
        zs: (...[_; _], _, Num) = xs; // should not work (Bool and Num cannot be unified)
    ";

    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("true", Type::BOOL);
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(err.main_location().span(code), "(...[_; _], _, Num)");
    assert_eq!(err.path(), [ErrorPathFragment::TupleElement(None)]);
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
    let code = "
        sum = |...xs: Num| xs.fold(0, |acc, x| acc + x);
        sum_spec: (Num, Num, Num) -> _ = sum;
        tuple_sum = |init, ...xs: (_, _)| xs.fold((init, init), |acc, x| acc + x);
    ";

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
    let code = "
        test: [any; _] = (1, 1 == 2, |x| x + 1);
        (x, y, z) = test;
        test(1) // should fail: slices are not callable
    ";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(err.main_location().span(code), "test(1)");
    assert_matches!(
        err.kind(),
        ErrorKind::TypeMismatch(lhs, rhs)
            if lhs.to_string() == "(Num) -> _" && rhs.to_string() == "(any, any, any)"
    );
    assert_eq!(type_env["x"], Type::Any);
    assert_eq!(type_env["y"], Type::Any);
    assert_eq!(type_env["z"], Type::Any);
}

#[test]
fn type_with_any_fn() {
    let code = "
        fun: (any) -> _ = |x| x == 1;
        fun(1 == 1) && fun((1, 2, 3)) && fun(|x| x + 1);
        fun(1, 2) // should fail: function is only callable with 1 arg
    ";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(err.main_location().span(code), "fun(1, 2)");
    assert_matches!(
        err.kind(),
        ErrorKind::TupleLenMismatch { lhs, rhs, .. }
            if *lhs == TupleLen::from(1) && *rhs == TupleLen::from(2)
    );
    assert_eq!(type_env["fun"].to_string(), "(any) -> Bool");
}

#[test]
fn dyn_type_in_slice() {
    let code = "
        lin_tuple: [dyn Lin; _] = (1, (2, 5), 9);
        bogus_tuple: [dyn Lin; _] = (1, 2, 3 == 4);
    ";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(err.main_location().span(code), "[dyn Lin; _]");
    assert_matches!(
        err.kind(),
        ErrorKind::FailedConstraint { ty, .. } if **ty == Type::BOOL
    );
    assert_eq!(
        type_env["lin_tuple"].to_string(),
        "(dyn Lin, dyn Lin, dyn Lin)"
    );
}

#[test]
fn mix_of_any_and_specific_types() {
    let code_samples = &[
        "|xs| { _unused: [any; _] = xs; xs + (1, 2) }",
        "|xs| { _unused: [any] = xs; xs + (1, 2) }",
        "|xs| { _unused: [dyn; _] = xs; xs + (1, 2) }",
        "|xs| { _unused: [dyn] = xs; xs + (1, 2) }",
        "|xs| { _unused: [dyn Lin; _] = xs; xs + (1, 2) }",
        "|xs| { _unused: [dyn Lin] = xs; xs + (1, 2) }",
    ];

    for &code in code_samples {
        let block = F32Grammar::parse_statements(code).unwrap();
        let mut type_env = TypeEnvironment::new();
        let output = type_env.process_statements(&block).unwrap();
        assert_eq!(output.to_string(), "((Num, Num)) -> (Num, Num)");
    }
}

#[test]
fn constraint_with_dyn_object_and_cast() {
    let code = "hash(#{ x: 1 } as dyn { x: Num } + Hash)";
    let block = F32Grammar::parse_statements(code).unwrap();
    let output = TypeEnvironment::new()
        .insert_object_safe_constraint(Hashed)
        .insert("hash", hash_fn_type())
        .process_statements(&block)
        .unwrap();

    assert_eq!(output, Type::NUM);
}

#[test]
fn generalizing_dyn_constraint() {
    let code = "#{ x: 1, y: 1 } as dyn { x: Num, y: Num } as dyn { x: _ }";
    let block = F32Grammar::parse_statements(code).unwrap();
    let output = TypeEnvironment::new().process_statements(&block).unwrap();

    assert_eq!(output.to_string(), "dyn { x: Num }");
}

#[test]
fn dyn_annotation_on_fn_arg() {
    let code = "
        test = |obj: dyn { x: Num, y: Num }| obj.x + obj.y;
        test(#{ x: 1, y: 2 });
        test(#{ x: 1, y: 2, z: 3 });

        pt: dyn { x: Num, y: Num } = #{ x: 1, y: 2, z: 3 };
        test(pt);
        pt: dyn { x: Num, y: Num, z: Num } = #{ x: 1, y: 2, z: 3 };
        test(pt);
    ";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["test"].to_string(),
        "(dyn { x: Num, y: Num }) -> Num"
    );
    assert_matches!(type_env["pt"], Type::Dyn(_));
}

#[test]
fn partial_dyn_annotation_on_fn_arg() {
    let code = "
        test = |obj: dyn { x: _, y: _ }| obj.x + obj.y;
        test(#{ x: 1, y: 2 });
        test(#{ x: 1, y: 2, z: 3 });
        test(#{ x: (1, 2), y: (3, 4), z: 3 });

        pt: dyn { x: Num, y: Num } = #{ x: 1, y: 2 };
        test(pt);
        pt: dyn { x: Num, y: _ } = #{ x: 1, y: 2 };
        test(pt);
        pt: dyn { x: _, y: _ } = #{ x: 1, y: 2 };
        test(pt);
    ";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["test"].to_string(),
        "for<'T: Ops> (dyn { x: 'T, y: 'T }) -> 'T"
    );
}

#[test]
fn dyn_annotation_propagation() {
    let code = "
        test = |objs: [dyn { x: _ }; _]| objs.map(|obj| obj.x + 1);
        (#{ x: 1, y: 2 }, #{ x: 2 }, #{ x: 3, y: (1,) }).test();
    ";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env
        .insert("map", Prelude::Map)
        .process_statements(&block)
        .unwrap();

    assert_eq!(
        type_env["test"].to_string(),
        "([dyn { x: Num }; N]) -> [Num; N]"
    );
}

#[test]
fn mix_of_any_and_specific_types_in_fns() {
    let code = "
        accepts_fn = |fn| { _unused: (any) -> any = fn; (fn(2), fn(3)) + (4, 5) };
    ";
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
    let code = "
        first: (_) -> [_] = |(x, y)| (x,);
        inc: ([_]) -> [_] = |xs: [Num; _]| xs.map(|x| x + 1);
        bogus: ([_]) -> [_] = |(x, y)| (x,);
    ";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.insert("map", Prelude::Map);
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(err.main_location().location_line(), 4);
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
    let code = "
        |fn| {
            _unused: ([_]) -> [_] = fn;
            fn((1, 2)) + (3, 4);
            _unused: ([_]) -> [Num] = fn;
            (x, ...) = fn((1, 2, 3));
            x
        }
    ";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let output = type_env.process_statements(&block).unwrap();

    assert_eq!(output.to_string(), "(([Num]) -> (Num, Num)) -> Num");
}

#[test]
fn custom_constraint_if_added_to_env() {
    let code = "x: dyn Hash = (1, 2);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut env = TypeEnvironment::new();
    env.insert_object_safe_constraint(Hashed)
        .process_statements(&block)
        .unwrap();

    assert_eq!(env["x"].to_string(), "dyn Hash");
}

#[test]
fn type_cast_basics() {
    let code = "(1, 2, 3) as [_]";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let output = type_env.process_statements(&block).unwrap();

    assert_eq!(output.to_string(), "[Num]");

    let any_code = "(1, (2, 3), 1 == 5) as dyn Hash";
    let any_block = F32Grammar::parse_statements(any_code).unwrap();
    let output = type_env
        .insert_object_safe_constraint(Hashed)
        .process_statements(&any_block)
        .unwrap();

    assert_eq!(output.to_string(), "dyn Hash");
}

#[test]
fn transmuting_type_via_casts() {
    let code = "(1, 2, 3) as any as (Num, Num)";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let output = type_env.process_statements(&block).unwrap();

    assert_eq!(output.to_string(), "(Num, Num)");
}

#[test]
fn indexing_with_annotations() {
    let code = "|xs: (_, _)| xs.0 + xs.1";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let output = type_env.process_statements(&block).unwrap();

    assert_eq!(output.to_string(), "for<'T: Ops> (('T, 'T)) -> 'T");
}

#[test]
fn object_annotations() {
    let code_samples = &["obj: { x: _ } = #{ x: 1 };", "obj: { x: Num } = #{ x: 1 };"];

    for &code in code_samples {
        let block = F32Grammar::parse_statements(code).unwrap();
        let mut type_env = TypeEnvironment::new();
        type_env.process_statements(&block).unwrap();
        assert_eq!(type_env["obj"].to_string(), "{ x: Num }");
    }
}

#[test]
fn object_annotations_in_function() {
    let code = "
        test = |obj: { x: _ }| obj.x == 1;
        test(#{ x: 1 });
    ";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["test"].to_string(), "({ x: Num }) -> Bool");

    let bogus_code = "test(#{ x: 1, y: 2 })";
    let bogus_block = F32Grammar::parse_statements(bogus_code).unwrap();
    let err = type_env
        .process_statements(&bogus_block)
        .unwrap_err()
        .single();

    assert_matches!(err.kind(), ErrorKind::FieldsMismatch { .. });
}

#[test]
fn object_destructure_with_narrowing_annotation() {
    let code = "
        test = |{ x -> x_: Num, y }| x_ + y;
        test(#{ x: 1, y: 2 });
    ";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["test"].to_string(),
        "for<'T: { x: Num, y: Num }> ('T) -> Num"
    );
}

#[test]
fn embedded_object_type_annotation() {
    let code = "
        test = |{ pt -> pt: { x: Num, y: Num } }| pt.x + pt.y;
        test(#{ pt: #{ x: 1, y: 2 } });
    ";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["test"].to_string(),
        "for<'T: { pt: { x: Num, y: Num } }> ('T) -> Num"
    );
}

#[test]
fn wildcard_var_influences_typing() {
    let code = "|(_: Num, y)| y == 1";
    let block = F32Grammar::parse_statements(code).unwrap();
    let output = TypeEnvironment::new().process_statements(&block).unwrap();
    assert_eq!(output.to_string(), "((Num, Num)) -> Bool");
}
