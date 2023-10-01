//! Tests focused on objects.

use assert_matches::assert_matches;

use arithmetic_parser::grammars::Parse;
use arithmetic_typing::{
    arith::NumArithmetic,
    defs::{Assertions, Prelude},
    error::{ErrorKind, TupleContext},
    TupleLen, Type, TypeEnvironment,
};

use crate::{hash_fn_type, ErrorsExt, F32Grammar};

#[test]
fn object_expr_basics() {
    let code = "x = 1; #{ x, y: (x + 1, x + 2) }";
    let block = F32Grammar::parse_statements(code).unwrap();
    let output = TypeEnvironment::new().process_statements(&block).unwrap();
    assert_eq!(output.to_string(), "{ x: Num, y: (Num, Num) }");
}

#[test]
fn object_field_access() {
    let code = "|obj| obj.x == 1";
    let block = F32Grammar::parse_statements(code).unwrap();
    let output = TypeEnvironment::new().process_statements(&block).unwrap();
    assert_eq!(output.to_string(), "for<'T: { x: Num }> ('T) -> Bool");

    let code = "|pt| pt.x + pt.y";
    let block = F32Grammar::parse_statements(code).unwrap();
    let output = TypeEnvironment::new().process_statements(&block).unwrap();
    assert_eq!(
        output.to_string(),
        "for<'T: { x: 'U, y: 'U }, 'U: Ops> ('T) -> 'U"
    );

    let code = "|pt| (pt.x, pt.y).fold(0, |acc, x| acc + x)";
    let block = F32Grammar::parse_statements(code).unwrap();
    let output = TypeEnvironment::new()
        .insert("fold", Prelude::Fold)
        .process_statements(&block)
        .unwrap();
    assert_eq!(
        output.to_string(),
        "for<'T: { x: Num, y: Num }> ('T) -> Num"
    );

    let code = "|pt| { pt.x + pt.y == 1; pt }";
    let block = F32Grammar::parse_statements(code).unwrap();
    let output = TypeEnvironment::new().process_statements(&block).unwrap();
    assert_eq!(output.to_string(), "for<'T: { x: Num, y: Num }> ('T) -> 'T");
}

#[test]
fn recursive_object_type_via_function_field() {
    let code = r#"
        call_len = |obj| (obj.len)(obj);
        pt = #{ x: 3, y: 4, len: |{ x, y }| x + y };
        // direct call should work
        {pt.len}(pt) == 7 && (pt.len)(#{ x: 1, y: 2 }) == 3;
        // call via a function should work as well
        call_len(pt) == 7;
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut env = TypeEnvironment::new();
    env.process_statements(&block).unwrap();

    assert_eq!(
        env["call_len"].to_string(),
        "for<'T: { len: ('T) -> 'U }> ('T) -> 'U"
    );

    let bogus_code = r#"
        pt = #{ x: 3, len: |{ x, y }| x as Num + y };
        call_len(pt)
    "#;
    let bogus_block = F32Grammar::parse_statements(bogus_code).unwrap();
    let err = env.process_statements(&bogus_block).unwrap_err().single();

    assert_matches!(err.kind(), ErrorKind::MissingFields { fields, .. } if fields.contains("y"));
}

#[test]
fn object_function_defs() {
    let code = r#"
        call_fn = |obj| (obj.fn)(#{ x: 1 }) as Num;
        obj = #{ fn: |{ x, y }| x as Num + y };
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut env = TypeEnvironment::new();
    env.process_statements(&block).unwrap();

    assert_eq!(
        env["call_fn"].to_string(),
        "for<'T: { fn: ({ x: Num }) -> Num }> ('T) -> Num"
    );
    assert_eq!(
        env["obj"].to_string(),
        "{ fn: for<'T: { x: Num, y: Num }> ('T) -> Num }"
    );

    let bogus_call = "call_fn(obj)";
    let bogus_block = F32Grammar::parse_statements(bogus_call).unwrap();
    let err = env.process_statements(&bogus_block).unwrap_err().single();

    assert_matches!(err.kind(), ErrorKind::MissingFields { fields, .. } if fields.contains("y"));
}

#[test]
fn recursive_object_definitions() {
    let code = r#"
        normalize_pt = |pt| #{ x: pt.x, y: pt.y } / ({pt.len}(pt) as Num);
        len = |{ x, y }| x + y;
        normalize_pt(#{ x: 3, y: 4, len }) == #{ x: 0.6, y: 0.8 };
        normalize_pt(#{ x: 3, y: 4, z: 0, len }) == #{ x: 0.6, y: 0.8 };
        normalize_pt(#{
            x: 3, y: 4,
            len: |pt| pt.x + 3,
        })
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut env = TypeEnvironment::new();
    let output = env.process_statements(&block).unwrap();

    assert_eq!(output.to_string(), "{ x: Num, y: Num }");

    let bogus_code = r#"
        normalize_pt(#{
            x: (3, 1), y: 4,
            len: |pt| pt.x + 3,
        })
    "#;
    let bogus_block = F32Grammar::parse_statements(bogus_code).unwrap();
    let err = env.process_statements(&bogus_block).unwrap_err().single();

    assert_matches!(err.kind(), ErrorKind::TypeMismatch(lhs, _) if *lhs == Type::NUM);

    let bogus_code = r#"
        normalize_pt(#{
            x: 3, y: 4,
            len: |x, y| x + y,
        })
    "#;
    let bogus_block = F32Grammar::parse_statements(bogus_code).unwrap();
    let errors = env.process_statements(&bogus_block).unwrap_err();

    assert!(errors.iter().any(|err| matches!(
        err.kind(),
        ErrorKind::TupleLenMismatch {
            lhs,
            rhs,
            context: TupleContext::FnArgs,
        } if *lhs == TupleLen::from(2) && *rhs == TupleLen::from(1)
    )));

    let bogus_code = r#"
        normalize_pt(#{
            x: 3, y: 4,
            len: |pt| pt.z + 3,
        })
    "#;
    let bogus_block = F32Grammar::parse_statements(bogus_code).unwrap();
    let err = env.process_statements(&bogus_block).unwrap_err().single();

    assert_matches!(err.kind(), ErrorKind::MissingFields { fields, .. } if fields.contains("z"));
}

#[test]
fn applying_object_constraints() {
    let code = r#"
        sum_coords = |pt| pt.x + pt.y;
        sum_coords(#{ x: 1, y: -3 }) == -2;
        sum_coords(#{ x: (1, 2), y: (-3, 4) })
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let output = TypeEnvironment::new().process_statements(&block).unwrap();
    assert_eq!(output.to_string(), "(Num, Num)");
}

#[test]
fn extra_fields_are_retained_with_constraints() {
    let code = r#"
        test = |obj| { obj.x == 1; obj };
        test(#{ x: 1, y: 2 }).y == 2;
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    TypeEnvironment::new().process_statements(&block).unwrap();
}

#[test]
fn additional_object_constraint() {
    let code = r#"
        require_x = |obj| obj.x == 1;
        require_y = |obj| require_x(obj) && obj.y != 2;
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["require_x"].to_string(),
        "for<'T: { x: Num }> ('T) -> Bool"
    );
    assert_eq!(
        type_env["require_y"].to_string(),
        "for<'T: { x: Num, y: Num }> ('T) -> Bool"
    );
}

#[test]
fn additional_object_constraints_through_multiple_fns() {
    let code = r#"
        require_x = |obj| obj.x == 1;
        require_y = |obj| obj.y == (2, 3);
        test = |obj| require_x(obj) && require_y(obj);

        test(#{ x: 1, y: (4, 5), z: 1 == 1 });
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["test"].to_string(),
        "for<'T: { x: Num, y: (Num, Num) }> ('T) -> Bool"
    );
}

#[test]
fn interleaving_object_constraints() {
    let code = r#"
        require_x = |obj| { obj.z == 1; obj.x };
        require_y = |obj| obj.x * obj.y;
        test = |obj| require_x(obj) + require_y(obj);

        test(#{ x: 1, y: 3, z: 1 });
        test(#{ x: (1, 2), z: 1, y: (2, 3) });
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["test"].to_string(),
        "for<'T: { x: 'U, y: 'U, z: Num }, 'U: Ops> ('T) -> 'U"
    );
}

#[test]
fn interleaving_object_constraints_complex_case() {
    let code = r#"
        require_x = |obj| { obj.z == 1; obj.x };
        require_y = |obj| obj.x == (obj.y, obj.z);
        test = |obj| { require_x(obj); require_y(obj) };

        test({ x = (1, 2); #{ x, y: x.0, z: x.1 } });
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["test"].to_string(),
        "for<'T: { x: ('U, Num), y: 'U, z: Num }> ('T) -> Bool"
    );
}

#[test]
fn functional_fields_in_objects() {
    let code = r#"
        obj = #{ x: 1, run: |x, y| x + y };
        run = obj.run;
        run((1, 2), (3, 4)) == (4, 6);
        (obj.run)(obj.x, 5)
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let output = type_env.process_statements(&block).unwrap();
    assert_eq!(output.to_string(), "Num");

    assert_eq!(
        type_env["obj"].to_string(),
        "{ run: for<'T: Ops> ('T, 'T) -> 'T, x: Num }"
    );
}

#[test]
fn functional_fields_in_object_constraints() {
    let code = "test = |obj| (obj.run)(obj.x, 1);";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut env = TypeEnvironment::new();
    env.insert("push", Prelude::Push);
    env.process_statements(&block).unwrap();
    assert_eq!(
        env["test"].to_string(),
        "for<'T: { run: ('U, Num) -> 'V, x: 'U }> ('T) -> 'V"
    );

    let code_samples = &[
        ("test(#{ x: 1, run: |x: Num, y: Num| x + y })", "Num"),
        ("test(#{ x: 1, run: |x, y| x + y })", "Num"),
        ("test(#{ run: push, x: (5, 6) })", "(Num, Num, Num)"),
    ];
    for &(run_code, expected_output) in code_samples {
        let run_block = F32Grammar::parse_statements(run_code).unwrap();
        let output = env.process_statements(&run_block).unwrap();
        assert_eq!(output.to_string(), expected_output);
    }
}

#[test]
fn object_and_ordinary_constraints() {
    let code = "fun = |obj| obj.x && hash(obj) == 0;";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env
        .insert("true", Type::BOOL)
        .insert("hash", hash_fn_type())
        .process_statements(&block)
        .unwrap();
    assert_eq!(
        type_env["fun"].to_string(),
        "for<'T: { x: Bool } + Hash> ('T) -> Bool"
    );

    let use_code = "fun(#{ x: true }) && fun(#{ x: true, y: 5 })";
    let use_block = F32Grammar::parse_statements(use_code).unwrap();
    type_env.process_statements(&use_block).unwrap();

    let bogus_code = "fun(#{ x: true, y: || 1 })";
    let bogus_block = F32Grammar::parse_statements(bogus_code).unwrap();
    let err = type_env
        .process_statements(&bogus_block)
        .unwrap_err()
        .single();
    assert_matches!(err.kind(), ErrorKind::FailedConstraint { .. });
}

#[test]
fn embedded_objects() {
    let code = r#"
        obj = #{
            x: #{ val: (1, 2, 3), len: 3 },
            y: 3,
        };
        obj.x.val.0 + obj.y;
        x = obj.x;
        x.val.fold(x.len, |acc, x| acc + x)
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let output = type_env
        .insert("fold", Prelude::Fold)
        .process_statements(&block)
        .unwrap();
    assert_eq!(output.to_string(), "Num");
}

#[test]
fn embedded_object_constraints() {
    let code = "|obj| (obj.x.len, obj.y == 3)";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let output = type_env.process_statements(&block).unwrap();

    assert_eq!(
        output.to_string(),
        "for<'T: { x: 'U, y: Num }, 'U: { len: 'V }> ('T) -> ('V, Bool)"
    );
}

#[test]
fn creating_object_in_closure() {
    let code = "(1, 2, 3).map(|x| #{ x, y: x + 1 })";
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let output = type_env
        .insert("map", Prelude::Map)
        .process_statements(&block)
        .unwrap();

    let tuple = match output {
        Type::Tuple(tuple) if tuple.len() == TupleLen::from(3) => tuple,
        _ => panic!("Unexpected output: {output:?}"),
    };
    let (_, element) = tuple.element_types().next().unwrap();
    assert_eq!(element.to_string(), "{ x: Num, y: Num }");
}

#[test]
fn creating_and_consuming_object_in_closure() {
    let code = r#"
        (1, 2, 3).map(|x| #{ x, y: x + 1 }).fold(0, |acc, pt| acc + pt.x / pt.y)
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let output = type_env
        .insert("map", Prelude::Map)
        .insert("fold", Prelude::Fold)
        .process_statements(&block)
        .unwrap();

    assert_eq!(output, Type::NUM);
}

#[test]
fn folding_to_object() {
    let code = r#"
        |xs| xs.fold(#{ min: INF, max: -INF }, |acc, x| #{
            min: if(x < acc.min, x, acc.min),
            max: if(x > acc.max, x, acc.max),
        })
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let output = type_env
        .insert("INF", Type::NUM)
        .insert("if", Prelude::If)
        .insert("fold", Prelude::Fold)
        .process_with_arithmetic(&NumArithmetic::with_comparisons(), &block)
        .unwrap();

    assert_eq!(output.to_string(), "([Num; N]) -> { max: Num, min: Num }");
}

#[test]
fn shared_type_vars_in_objects() {
    let code = r#"
        fun = |x, obj| x == obj.x;
        fun(5, #{ x: 4 });
        fun(5, #{ x: 4, y: 2 });
        fun((1, true), #{ x: (5, true) });
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env
        .insert("true", Type::BOOL)
        .process_statements(&block)
        .unwrap();

    assert_eq!(
        type_env["fun"].to_string(),
        "for<'U: { x: 'T }> ('T, 'U) -> Bool"
    );
}

#[test]
fn shared_type_vars_in_objects_curried() {
    let code = r#"
        fun = |x| |obj| x == obj.x;
        fun(5)(#{ x: 4 });
        fun(5)(#{ x: 4, y: 2 });
        fun((1, true))(#{ x: (5, true) });
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env
        .insert("true", Type::BOOL)
        .process_statements(&block)
        .unwrap();

    assert_eq!(
        type_env["fun"].to_string(),
        "for<'U: { x: 'T }> ('T) -> ('U) -> Bool"
    );

    let bogus_code = "fun((1, true))(#{ x: 5 });";
    let bogus_block = F32Grammar::parse_statements(bogus_code).unwrap();
    let err = type_env
        .process_statements(&bogus_block)
        .unwrap_err()
        .single();

    assert_eq!(
        err.kind().to_string(),
        "Type `Num` is not assignable to type `(Num, Bool)`"
    );
}

#[test]
fn tuples_as_object_fields() {
    let code = r#"
        test = |obj| { obj.xs == obj.ys.map(|y| (y, y * 2)) };
        test(#{ xs: ((1, 2), (3, 4)), ys: (3, 4) });
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env
        .insert("map", Prelude::Map)
        .process_statements(&block)
        .unwrap();

    assert_eq!(
        type_env["test"].to_string(),
        "for<'T: { xs: [(Num, Num); N], ys: [Num; N] }> ('T) -> Bool"
    );
}

#[test]
fn tuples_with_dyn_length_as_object_fields() {
    let code = r#"
        test = |obj| { obj.xs == obj.ys.filter(|y| y > 1) };
        test(#{ xs: (2, 3), ys: (1, 2, 3) });
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env
        .insert("filter", Prelude::Filter)
        .process_with_arithmetic(&NumArithmetic::with_comparisons(), &block)
        .unwrap();

    assert_eq!(
        type_env["test"].to_string(),
        "for<'T: { xs: [Num], ys: [Num; N] }> ('T) -> Bool"
    );
}

#[test]
fn object_destructure_basics() {
    let code = r#"
        { x } = #{ x: 1 };
        { x -> y } = #{ x: 2 };
        obj = #{ xs: (3, 4, 5), flag: 1 == 1 };
        { xs: (head, ...tail), flag } = obj;
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&block).unwrap();

    assert_eq!(type_env["x"], Type::NUM);
    assert_eq!(type_env["y"], Type::NUM);
    assert_eq!(type_env["head"], Type::NUM);
    assert_eq!(type_env["tail"], Type::from((Type::NUM, Type::NUM)));
    assert_eq!(type_env["flag"], Type::BOOL);
}

#[test]
fn object_destructure_in_fn_args() {
    let code = r#"
        test = |{ x, y }| x + y;
        test(#{ x: 1, y: 2 })
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let output = type_env.process_statements(&block).unwrap();

    assert_eq!(output, Type::NUM);
    assert_eq!(
        type_env["test"].to_string(),
        "for<'T: { x: 'U, y: 'U }, 'U: Ops> ('T) -> 'U"
    );
}

#[test]
fn object_destructure_with_complex_bindings() {
    let code = r#"
        test = |{ xs: (head, ...xs), ys }| xs == ys && head == 1;
        test(#{ xs: (1, 2, 3), ys: (2, 4) });
        test(#{ xs: (1, true), ys: (true,) });
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env
        .insert("true", Type::BOOL)
        .process_statements(&block)
        .unwrap();

    assert_eq!(
        type_env["test"].to_string(),
        "for<'T: { xs: (Num, ...['U; N]), ys: ['U; N] }> ('T) -> Bool"
    );
}

#[test]
fn object_destructure_in_map_pipeline() {
    let code = r#"
        test = |xs| xs.map(|{ x, y }| x as Num + y);
        (#{ x: 1, y: 2 }, #{ x: 3, y: 4 }).test() == (3, 7);
        (
            #{ x: 1, y: 2, z: 3 },
            #{ x: 3, y: 4, z: -1 },
        ).test() == (3, 7);
        // Unfortunately, having `z` in *one* of tuple items doesn't work for now.
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();

    let mut type_env = TypeEnvironment::new();
    type_env
        .insert("map", Prelude::Map)
        .process_statements(&block)
        .unwrap();
    assert_eq!(
        type_env["test"].to_string(),
        "for<'T: { x: Num, y: Num }> (['T; N]) -> [Num; N]"
    );
}

#[test]
fn object_destructure_in_fold_pipeline() {
    let code = r#"
        minmax = |xs| xs.fold(#{ min: INF, max: -INF }, |{ min, max }, x| #{
            min: if(x < min, x, min),
            max: if(x > max, x, max),
        });
        assert_eq((5, -4, 6, 9, 1).minmax(), #{ min: -4, max: 9 });
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();

    let mut type_env = TypeEnvironment::new();
    type_env
        .insert("INF", Type::NUM)
        .insert("if", Prelude::If)
        .insert("fold", Prelude::Fold)
        .insert("assert_eq", Assertions::AssertEq)
        .process_with_arithmetic(&NumArithmetic::with_comparisons(), &block)
        .unwrap();

    assert_eq!(
        type_env["minmax"].to_string(),
        "([Num; N]) -> { max: Num, min: Num }"
    );
}

#[test]
fn accessing_std_function_via_object() {
    let code = "{Array.map}((1, 2, 3), |x| x + 1)";
    let block = F32Grammar::parse_statements(code).unwrap();

    let mut type_env = TypeEnvironment::new();
    type_env.extend(Prelude::iter());
    let output = type_env.process_statements(&block).unwrap();
    assert_eq!(output.to_string(), "(Num, Num, Num)");
}
