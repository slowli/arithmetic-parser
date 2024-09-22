//! Tests length equations.

use arithmetic_parser::grammars::{NumGrammar, Parse};
use arithmetic_typing::{
    defs::Prelude,
    error::{ErrorKind, TupleContext},
    Annotated, Function, TupleLen, Type, TypeEnvironment, UnknownLen,
};
use assert_matches::assert_matches;

use crate::ErrorsExt;

type F32Grammar = Annotated<NumGrammar<f32>>;

#[test]
fn push_fn_basics() {
    let code = "(1, 2).push(3).push(4)";
    let block = F32Grammar::parse_statements(code).unwrap();
    let tuple = TypeEnvironment::new()
        .insert("push", Prelude::Push)
        .process_statements(&block)
        .unwrap();

    assert_eq!(tuple, Type::slice(Type::NUM, TupleLen::from(4)));
}

#[test]
fn push_fn_in_other_fn_definition() {
    let code = "
        push_fork = |...xs, item| (xs, xs.push(item));
        (xs, ys) = push_fork(1, 2, 3, 4);
        (_, (_, z)) = push_fork(4);
    ";
    let block = F32Grammar::parse_statements(code).unwrap();

    let mut type_env = TypeEnvironment::new();
    type_env.insert("push", Prelude::Push);
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(err.main_location().span(code), "(_, z)");
    assert_eq!(err.root_location().span(code), "(_, (_, z))");
    assert_matches!(
        err.kind(),
        ErrorKind::TupleLenMismatch {
            lhs,
            rhs,
            context: TupleContext::Generic,
        } if *lhs == TupleLen::from(2) && *rhs == TupleLen::from(1)
    );

    assert_eq!(
        type_env["push_fork"].to_string(),
        "(...['T; N], 'T) -> (['T; N], ['T; N + 1])"
    );
    assert_eq!(type_env["xs"], Type::slice(Type::NUM, TupleLen::from(3)));
    assert_eq!(type_env["ys"], Type::slice(Type::NUM, TupleLen::from(4)));
}

#[test]
fn several_push_applications() {
    let code = "
        push2 = |xs, x, y| xs.push(x).push(y);
        (head, ...tail) = (1, 2).push2(3, 4);
    ";
    let block = F32Grammar::parse_statements(code).unwrap();

    let mut type_env = TypeEnvironment::new();
    type_env.insert("push", Prelude::Push);
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["push2"].to_string(),
        "(['T; N], 'T, 'T) -> ['T; N + 2]"
    );
    assert_eq!(type_env["head"], Type::NUM);
    assert_eq!(type_env["tail"], Type::slice(Type::NUM, TupleLen::from(3)));
}

#[test]
fn comparing_lengths_after_push() {
    let code = r#"
        simple = |xs, ys| xs.push(0) == ys.push(1);
        _asymmetric = |xs, ys| xs.push(0) == ys;
        asymmetric = |xs, ys| xs == ys.push(0);
        complex = |xs, ys| xs.push(0) + ys.push(1).push(0);
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();

    let mut type_env = TypeEnvironment::new();
    type_env.insert("push", Prelude::Push);
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["simple"].to_string(),
        "([Num; N], [Num; N]) -> Bool"
    );
    assert_eq!(
        type_env["asymmetric"].to_string(),
        "([Num; N + 1], [Num; N]) -> Bool"
    );
    assert_eq!(
        type_env["complex"].to_string(),
        "for<len! N> ([Num; N + 1], [Num; N]) -> [Num; N + 2]"
    );
}

#[test]
fn requirements_on_len_via_destructuring() {
    let code = r#"
        len_at_least2 = |xs| { (_, _, ...) = xs; xs };

        // Check propagation to other fns.
        test_fn = |xs: [_; _]| xs.len_at_least2().fold(0, |acc, x| acc + x);

        other_test_fn = |xs: [_; _]| {
           (head, ...tail) = xs.len_at_least2();
           head == (1, 1 == 1);
           tail.map(|(x, _)| x)
        };

        (1, 2).len_at_least2();
        (..., x) = (1, 2, 3, 4).len_at_least2();
        (1,).len_at_least2();
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();

    let mut type_env = TypeEnvironment::new();
    type_env
        .insert("fold", Prelude::Fold)
        .insert("map", Prelude::Map);
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(err.main_location().span(code), "(1,)");
    assert_eq!(err.root_location().span(code), "(1,).len_at_least2()");
    assert_matches!(
        err.kind(),
        ErrorKind::TupleLenMismatch {
            lhs,
            rhs,
            context: TupleContext::Generic,
        } if lhs.to_string() == "_ + 2" && *rhs == TupleLen::from(1)
    );

    assert_eq!(
        type_env["len_at_least2"].to_string(),
        "(('T, 'U, ...['V; N])) -> ('T, 'U, ...['V; N])"
    );
    assert_eq!(type_env["test_fn"].to_string(), "([Num; N + 2]) -> Num");
    assert_eq!(
        type_env["other_test_fn"].to_string(),
        "([(Num, Bool); N + 2]) -> [Num; N + 1]"
    );
}

#[test]
fn reversing_a_slice() {
    let code = r#"
        reverse = |xs| {
            empty: [_] = ();
            xs.fold(empty, |acc, x| (x,).merge(acc))
        };
        ys = (2, 3, 4).reverse().map(|x| x == 1);
        (_, ...) = ys;
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();

    let mut type_env = TypeEnvironment::new();
    type_env
        .insert("fold", Prelude::Fold)
        .insert("map", Prelude::Map)
        .insert("merge", Prelude::Merge);
    let err = type_env.process_statements(&block).unwrap_err().single();

    assert_eq!(err.main_location().span(code), "(_, ...)");
    assert_matches!(err.kind(), ErrorKind::TupleLenMismatch { .. });
    assert_eq!(type_env["reverse"].to_string(), "(['T; N]) -> ['T]");
    assert_eq!(type_env["ys"].to_string(), "[Bool]");
}

#[test]
fn errors_when_adding_dynamic_slices() {
    let setup_code = r#"
        slice: [_] = (1, 2, 3);
        other_slice: [_] = (4, 5);
        slice = -slice;
        other_slice * 8; // works: dynamic slices are linear
    "#;
    let setup_block = F32Grammar::parse_statements(setup_code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env.process_statements(&setup_block).unwrap();

    let invalid_code = r#"
        slice + slice;
        slice + other_slice;
        (7,) + slice;
    "#;
    for line in invalid_code.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let line = F32Grammar::parse_statements(line).unwrap();
        let errors = type_env.process_statements(&line).unwrap_err();
        let err = errors.into_iter().next().unwrap();
        assert_matches!(err.kind(), ErrorKind::DynamicLen(_));
    }
}

#[test]
fn square_function() {
    let square = Type::slice(Type::param(0), UnknownLen::param(0)).repeat(UnknownLen::param(0));
    let square_fn = Function::builder()
        .with_arg(square)
        .returning(Type::void())
        .with_static_lengths(&[0]);
    assert_eq!(square_fn.to_string(), "for<len! N> ([['T; N]; N]) -> ()");

    let code = r#"
        ((1, 2), (3, 4)).is_square();
        ((true,),).is_square();
        ((1, 2), (3, 4), (5, 6)).is_square();
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    type_env
        .insert("true", Prelude::True)
        .insert("is_square", square_fn);
    let errors = type_env.process_statements(&block).unwrap_err();
    let err = errors.into_iter().next().unwrap();

    assert_eq!(err.main_location().span(code), "(1, 2)");
    assert_eq!(
        err.root_location().span(code),
        "((1, 2), (3, 4), (5, 6)).is_square()"
    );
    assert_matches!(
        err.kind(),
        ErrorKind::TupleLenMismatch { lhs, rhs, .. }
            if *lhs == TupleLen::from(3) && *rhs == TupleLen::from(2)
    );
}

#[test]
fn column_row_equality_fn() {
    let code = r#"
        first_col = |xs| xs.map(|row| { (x, ...) = row; x });
        // Slice annotations are not required, but result in simpler signatures.
        row_eq_col = |xs: [_; _]| {
            (first_row: [_; _], ...) = xs;
            first_row == xs.first_col()
        };

        col: [Bool] = ((true, 1), (false, 5), (true, 9)).first_col();
        ((1, 2), (3, 4)).row_eq_col();
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env: TypeEnvironment = Prelude::iter().collect();
    type_env.process_statements(&block).unwrap();

    assert_eq!(
        type_env["first_col"].to_string(),
        "([('T, ...['U; M]); N]) -> ['T; N]"
    );
    assert_eq!(
        type_env["row_eq_col"].to_string(),
        "([['T; N + 1]; N + 1]) -> Bool"
    );

    let bogus_lines = &[
        "((1, 2), (3, 4), (5, 6)).row_eq_col()",
        "((1, 2, 3), (4, 5, 6)).row_eq_col()",
        // Doesn't work: we require `N + 1 == *`, which cannot be solved for `N`.
        "zs: [[Num]] = ((1, 2), (3, 4)); zs.row_eq_col()",
    ];
    for &bogus_line in bogus_lines {
        let bogus_line = F32Grammar::parse_statements(bogus_line).unwrap();
        let errors = type_env.process_statements(&bogus_line).unwrap_err();
        let err = errors.into_iter().next().unwrap();
        assert_matches!(err.kind(), ErrorKind::TupleLenMismatch { .. });
    }

    let test_code = r#"
        zs: [[Num; _]] = ((1, 2), (3, 4));
        zs.push((5, 6)).row_eq_col(); // works: `N` can be unified with `*`
        zs.push((3, 4, 5)); // fail: `zs` elements are `(Num, Num)`
    "#;
    let block = F32Grammar::parse_statements(test_code).unwrap();
    let err = type_env.process_statements(&block).unwrap_err().single();
    assert_eq!(err.main_location().span(test_code), "(3, 4, 5)");
    assert_eq!(err.root_location().span(test_code), "zs.push((3, 4, 5))");
    assert_matches!(err.kind(), ErrorKind::TupleLenMismatch { .. });
}

#[test]
fn total_sum() {
    let code = r#"
        total_sum = |xs| xs.fold(0, |acc, row| acc + row.fold(0, |acc, x| acc + x));
        xs: [[_]] = ((1, 2), (3, 4, 5), (6,));
        xs.total_sum()
    "#;
    let block = F32Grammar::parse_statements(code).unwrap();
    let mut type_env = TypeEnvironment::new();
    let output = type_env
        .insert("fold", Prelude::Fold)
        .process_statements(&block)
        .unwrap();

    assert_eq!(output, Type::NUM);
    assert_eq!(type_env["total_sum"].to_string(), "([[Num; M]; N]) -> Num");
}
