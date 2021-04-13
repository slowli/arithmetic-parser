//! Unit tests for `Substitutions`.

use assert_matches::assert_matches;

use super::*;
use crate::{error::TupleLenMismatchContext, Num};

fn extract_errors<Prim: PrimitiveType>(
    action: impl FnOnce(OpTypeErrors<'_, Prim>),
) -> Result<(), TypeErrorKind<Prim>> {
    let mut errors = OpTypeErrors::new();
    action(errors.by_ref());
    let mut errors = errors.into_vec();

    match errors.len() {
        0 => Ok(()),
        1 => Err(errors.pop().unwrap()),
        _ => panic!("Unexpected multiple errors: {:?}", errors),
    }
}

fn unify_lengths<Prim: PrimitiveType>(
    substitutions: &mut Substitutions<Prim>,
    lhs: TupleLen,
    rhs: TupleLen,
) -> Result<(), TypeErrorKind<Prim>> {
    substitutions
        .unify_lengths(lhs, rhs, TupleLenMismatchContext::Assignment)
        .map(drop)
}

fn unify_tuples<Prim: PrimitiveType>(
    substitutions: &mut Substitutions<Prim>,
    lhs: &Tuple<Prim>,
    rhs: &Tuple<Prim>,
) -> Result<(), TypeErrorKind<Prim>> {
    extract_errors(|errors| {
        substitutions.unify_tuples(lhs, rhs, TupleLenMismatchContext::Assignment, errors);
    })
}

fn unify<Prim: PrimitiveType>(
    substitutions: &mut Substitutions<Prim>,
    lhs: &ValueType<Prim>,
    rhs: &ValueType<Prim>,
) -> Result<(), TypeErrorKind<Prim>> {
    extract_errors(|errors| substitutions.unify(lhs, rhs, errors))
}

#[test]
fn unifying_lengths_success_without_side_effects() {
    let samples = &[
        (TupleLen::from(2), TupleLen::from(2)),
        (
            UnknownLen::free_var(1).into(),
            UnknownLen::free_var(1).into(),
        ),
        // `Dynamic` should always be accepted on LHS.
        (UnknownLen::Dynamic.into(), TupleLen::from(2)),
        (UnknownLen::Dynamic.into(), UnknownLen::free_var(3).into()),
        (UnknownLen::Dynamic.into(), UnknownLen::Dynamic.into()),
        (UnknownLen::Dynamic.into(), UnknownLen::Dynamic + 1),
    ];

    let mut substitutions = Substitutions::<Num>::default();
    for &(lhs, rhs) in samples {
        unify_lengths(&mut substitutions, lhs, rhs).unwrap();
        assert!(substitutions.length_eqs.is_empty());
    }
}

#[test]
fn unifying_lengths_error() {
    let samples = &[
        (TupleLen::from(1), TupleLen::from(2)),
        (TupleLen::from(2), UnknownLen::Dynamic.into()),
        (UnknownLen::Dynamic + 1, UnknownLen::Dynamic.into()),
        (UnknownLen::free_var(1).into(), UnknownLen::free_var(1) + 1),
        (UnknownLen::free_var(1) + 2, UnknownLen::free_var(1).into()),
    ];

    let mut substitutions = Substitutions::<Num>::default();
    for &(lhs, rhs) in samples {
        let err = unify_lengths(&mut substitutions, lhs, rhs).unwrap_err();

        assert_matches!(err, TypeErrorKind::TupleLenMismatch { .. });
        assert!(substitutions.length_eqs.is_empty());
    }
}

#[test]
fn unifying_lengths_success_with_new_equation() {
    let len_samples = &[
        TupleLen::from(3),
        UnknownLen::free_var(5).into(),
        UnknownLen::Dynamic.into(),
    ];

    for &rhs in len_samples {
        let mut substitutions = Substitutions::<Num>::default();
        unify_lengths(&mut substitutions, UnknownLen::free_var(2).into(), rhs).unwrap();

        assert_eq!(substitutions.length_eqs.len(), 1);
        assert_eq!(substitutions.length_eqs[&2], rhs);
    }

    let mut substitutions = Substitutions::<Num>::default();
    unify_lengths(
        &mut substitutions,
        TupleLen::from(3),
        UnknownLen::free_var(2).into(),
    )
    .unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&2], TupleLen::from(3));
}

#[test]
fn unifying_compound_length_success() {
    let compound_len = UnknownLen::free_var(0) + 2;
    let mut substitutions = Substitutions::<Num>::default();
    unify_lengths(&mut substitutions, compound_len, TupleLen::from(5)).unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&0], TupleLen::from(3));

    let mut substitutions = Substitutions::<Num>::default();
    unify_lengths(&mut substitutions, TupleLen::from(5), compound_len).unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&0], TupleLen::from(3));

    let other_compound_len = UnknownLen::free_var(1) + 2;
    let mut substitutions = Substitutions::<Num>::default();
    unify_lengths(&mut substitutions, compound_len, other_compound_len).unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&0], UnknownLen::free_var(1).into());
}

#[test]
fn unifying_compound_length_with_dyn_length() {
    let compound_len = UnknownLen::Dynamic + 2;
    let mut substitutions = Substitutions::<Num>::default();

    unify_lengths(&mut substitutions, compound_len, TupleLen::from(5)).unwrap();
    assert!(substitutions.length_eqs.is_empty());

    unify_lengths(&mut substitutions, compound_len, compound_len).unwrap();
    assert!(substitutions.length_eqs.is_empty());

    let err = unify_lengths(&mut substitutions, TupleLen::from(5), compound_len).unwrap_err();
    assert_matches!(err, TypeErrorKind::TupleLenMismatch { .. });

    let other_compound_len = UnknownLen::free_var(1) + 2;
    unify_lengths(&mut substitutions, compound_len, other_compound_len).unwrap();
    assert!(substitutions.length_eqs.is_empty());

    let other_dyn_len = UnknownLen::Dynamic + 3;
    unify_lengths(&mut substitutions, compound_len, other_dyn_len).unwrap();
    assert!(substitutions.length_eqs.is_empty());

    let err = unify_lengths(&mut substitutions, other_dyn_len, compound_len).unwrap_err();
    assert_matches!(err, TypeErrorKind::TupleLenMismatch { .. });

    unify_lengths(&mut substitutions, other_compound_len, compound_len).unwrap();
    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&1], UnknownLen::Dynamic.into());
}

#[test]
fn unifying_compound_length_errors() {
    let compound_len = UnknownLen::free_var(0) + 2;
    let mut substitutions = Substitutions::<Num>::default();
    let err = unify_lengths(&mut substitutions, compound_len, TupleLen::from(1)).unwrap_err();

    assert_matches!(err, TypeErrorKind::TupleLenMismatch { .. });
}

#[test]
fn unresolved_param_error() {
    let mut substitutions = Substitutions::<Num>::default();
    let err = unify_lengths(
        &mut substitutions,
        UnknownLen::param(2).into(),
        TupleLen::from(1),
    )
    .unwrap_err();
    assert_matches!(err, TypeErrorKind::UnresolvedParam);

    let err = unify(&mut substitutions, &ValueType::param(2), &ValueType::NUM).unwrap_err();
    assert_matches!(err, TypeErrorKind::UnresolvedParam);
}

#[test]
fn unifying_complex_tuples() {
    let xs = Tuple::new(
        vec![ValueType::NUM, ValueType::NUM],
        ValueType::NUM.repeat(UnknownLen::free_var(0)),
        vec![],
    );
    let ys = Tuple::from(ValueType::NUM.repeat(UnknownLen::free_var(1) + 2));

    let mut substitutions = Substitutions::<Num>::default();
    unify_tuples(&mut substitutions, &xs, &ys).unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&0], UnknownLen::free_var(1).into());

    let zs = Tuple::from(ValueType::free_var(0).repeat(UnknownLen::free_var(1)));
    let mut substitutions = Substitutions::<Num>::default();
    unify_tuples(&mut substitutions, &xs, &zs).unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&1], UnknownLen::free_var(0) + 2);
    assert_eq!(substitutions.eqs.len(), 1);
    assert_eq!(substitutions.eqs[&0], ValueType::NUM);

    let us = Tuple::new(
        vec![],
        ValueType::NUM.repeat(UnknownLen::free_var(1)),
        vec![
            ValueType::free_var(0),
            ValueType::free_var(1),
            ValueType::free_var(2),
        ],
    );
    let mut substitutions = Substitutions::<Num>::default();
    unify_tuples(&mut substitutions, &xs, &us).unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&0], UnknownLen::free_var(1) + 1);
    assert_eq!(substitutions.eqs.len(), 3);
    for i in 0..3 {
        assert_eq!(substitutions.eqs[&i], ValueType::NUM);
    }
}

#[test]
fn any_can_be_unified_with_anything() {
    let mut substitutions = Substitutions::<Num>::default();
    substitutions.eqs.insert(0, ValueType::any());

    let rhs_samples = &[
        ValueType::NUM,
        ValueType::BOOL,
        (ValueType::BOOL, ValueType::NUM).into(),
        ValueType::NUM.repeat(3).into(),
        ValueType::NUM.repeat(UnknownLen::free_var(0)).into(),
        FnType::new(
            vec![ValueType::BOOL, ValueType::NUM].into(),
            ValueType::void(),
        )
        .into(),
        ValueType::any(),
    ];
    for rhs in rhs_samples {
        unify(&mut substitutions, &ValueType::free_var(0), rhs).unwrap();
    }
}

#[test]
fn static_length_restrictions() {
    let mut substitutions = Substitutions::<Num>::default();

    let positive_samples = &[
        TupleLen::from(0),
        TupleLen::from(3),
        UnknownLen::free_var(0).into(),
        UnknownLen::free_var(1) + 2,
    ];
    for &sample in positive_samples {
        substitutions.apply_static_len(sample).unwrap();
    }
    assert_eq!(
        substitutions.static_lengths,
        vec![0, 1].into_iter().collect::<HashSet<_>>()
    );

    let negative_samples = &[UnknownLen::Dynamic.into(), UnknownLen::Dynamic + 2];
    for &sample in negative_samples {
        let err = substitutions.apply_static_len(sample).unwrap_err();
        assert_matches!(err, TypeErrorKind::DynamicLen(_));
    }
}

#[test]
fn marking_length_as_static_and_then_failing_unification() {
    let mut substitutions = Substitutions::<Num>::default();
    substitutions.static_lengths.insert(0);

    let positive_samples: &[(TupleLen, _)] = &[
        (UnknownLen::Dynamic.into(), UnknownLen::free_var(0).into()),
        (UnknownLen::Dynamic.into(), UnknownLen::free_var(0) + 1),
    ];
    for &(lhs, rhs) in positive_samples {
        unify_lengths(&mut substitutions, lhs, rhs).unwrap();
        assert!(substitutions.length_eqs.is_empty());
    }

    let failing_samples: &[(TupleLen, _)] = &[
        (UnknownLen::free_var(0).into(), UnknownLen::Dynamic + 2),
        (UnknownLen::free_var(0).into(), UnknownLen::Dynamic.into()),
        (UnknownLen::free_var(0) + 1, UnknownLen::Dynamic + 2),
    ];
    for &(lhs, rhs) in failing_samples {
        let err = unify_lengths(&mut substitutions, lhs, rhs).unwrap_err();
        assert_matches!(err, TypeErrorKind::DynamicLen(_));
    }
}

#[test]
fn marking_length_as_static_and_then_propagating() {
    let mut substitutions = Substitutions::<Num>::default();
    substitutions.static_lengths.insert(0);
    unify_lengths(
        &mut substitutions,
        UnknownLen::free_var(1).into(),
        UnknownLen::free_var(0).into(),
    )
    .unwrap();

    assert_eq!(substitutions.length_eqs[&1], UnknownLen::free_var(0).into());
    assert_eq!(substitutions.static_lengths.len(), 1);

    let mut substitutions = Substitutions::<Num>::default();
    substitutions.static_lengths.insert(0);
    unify_lengths(
        &mut substitutions,
        UnknownLen::free_var(0).into(),
        UnknownLen::free_var(1).into(),
    )
    .unwrap();

    assert_eq!(substitutions.length_eqs[&0], UnknownLen::free_var(1).into());
    assert!(substitutions.static_lengths.contains(&1));
}
