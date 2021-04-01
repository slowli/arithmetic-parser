//! Unit tests for `Substitutions`.

use assert_matches::assert_matches;

use super::*;
use crate::{Num, TupleLenMismatchContext::Assignment};

#[test]
fn unifying_lengths_success_without_side_effects() {
    let samples = &[
        (TupleLen::from(2), TupleLen::from(2)),
        (SimpleTupleLen::Var(1).into(), SimpleTupleLen::Var(1).into()),
        // `Dynamic` should always be accepted on LHS.
        (SimpleTupleLen::Dynamic.into(), TupleLen::from(2)),
        (
            SimpleTupleLen::Dynamic.into(),
            SimpleTupleLen::Var(3).into(),
        ),
    ];

    let mut substitutions = Substitutions::<Num>::default();
    for &(mut lhs, mut rhs) in samples {
        substitutions.assign_new_len(&mut lhs);
        substitutions.assign_new_len(&mut rhs);
        substitutions.unify_lengths(lhs, rhs, Assignment).unwrap();

        assert!(substitutions.length_eqs.is_empty());
    }
}

#[test]
fn unifying_lengths_error() {
    let samples = &[
        (TupleLen::from(1), TupleLen::from(2)),
        (
            SimpleTupleLen::Dynamic.into(),
            SimpleTupleLen::Dynamic.into(),
        ),
        (
            SimpleTupleLen::Var(0).into(),
            SimpleTupleLen::Dynamic.into(),
        ),
        (TupleLen::from(2), SimpleTupleLen::Dynamic.into()),
    ];

    let mut substitutions = Substitutions::<Num>::default();
    for &(mut lhs, mut rhs) in samples {
        substitutions.assign_new_len(&mut lhs);
        substitutions.assign_new_len(&mut rhs);
        let err = substitutions
            .unify_lengths(lhs, rhs, Assignment)
            .unwrap_err();

        assert_matches!(err, TypeErrorKind::TupleLenMismatch { .. });
        assert!(substitutions.length_eqs.is_empty());
    }
}

#[test]
fn unifying_lengths_success_with_new_equation() {
    let len_samples = &[
        TupleLen::from(3),
        SimpleTupleLen::Var(5).into(),
        SimpleTupleLen::Dynamic.into(),
    ];

    for rhs in len_samples {
        let mut rhs = *rhs;
        let mut substitutions = Substitutions::<Num>::default();
        substitutions.assign_new_len(&mut rhs);
        substitutions
            .unify_lengths(SimpleTupleLen::Var(2).into(), rhs, Assignment)
            .unwrap();

        assert_eq!(substitutions.length_eqs.len(), 1);
        assert_eq!(substitutions.length_eqs[&2], rhs);
    }

    let mut substitutions = Substitutions::<Num>::default();
    substitutions
        .unify_lengths(TupleLen::from(3), SimpleTupleLen::Var(2).into(), Assignment)
        .unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&2], TupleLen::from(3));
}

#[test]
fn unifying_compound_length_success() {
    let compound_len = SimpleTupleLen::Var(0) + 2;
    let mut substitutions = Substitutions::<Num>::default();
    substitutions
        .unify_lengths(compound_len, TupleLen::from(5), Assignment)
        .unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&0], TupleLen::from(3));

    let mut substitutions = Substitutions::<Num>::default();
    substitutions
        .unify_lengths(TupleLen::from(5), compound_len, Assignment)
        .unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&0], TupleLen::from(3));

    let other_compound_len = SimpleTupleLen::Var(1) + 2;
    let mut substitutions = Substitutions::<Num>::default();
    substitutions
        .unify_lengths(compound_len, other_compound_len, Assignment)
        .unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&0], SimpleTupleLen::Var(1).into());
}

#[test]
fn unifying_compound_length_with_dyn_length() {
    let compound_len = SimpleTupleLen::Var(0) + 2;
    let mut substitutions = Substitutions::<Num>::default();
    substitutions.dyn_lengths.insert(0);

    substitutions
        .unify_lengths(compound_len, TupleLen::from(5), Assignment)
        .unwrap();
    assert!(substitutions.length_eqs.is_empty());

    substitutions
        .unify_lengths(compound_len, compound_len, Assignment)
        .unwrap();
    assert!(substitutions.length_eqs.is_empty());

    let err = substitutions
        .unify_lengths(TupleLen::from(5), compound_len, Assignment)
        .unwrap_err();
    assert_matches!(err, TypeErrorKind::TupleLenMismatch { .. });

    let other_compound_len = SimpleTupleLen::Var(1) + 2;
    substitutions
        .unify_lengths(compound_len, other_compound_len, Assignment)
        .unwrap();
    assert!(substitutions.length_eqs.is_empty());

    substitutions.dyn_lengths.insert(1);
    let err = substitutions
        .unify_lengths(other_compound_len, compound_len, Assignment)
        .unwrap_err();
    assert_matches!(err, TypeErrorKind::TupleLenMismatch { .. });

    substitutions.dyn_lengths.remove(&1);
    substitutions
        .unify_lengths(other_compound_len, compound_len, Assignment)
        .unwrap();
    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&1], SimpleTupleLen::Var(0).into());
}

#[test]
fn unifying_compound_length_errors() {
    let compound_len = SimpleTupleLen::Var(0) + 2;

    let mut substitutions = Substitutions::<Num>::default();
    let err = substitutions
        .unify_lengths(compound_len, TupleLen::from(1), Assignment)
        .unwrap_err();

    assert_matches!(err, TypeErrorKind::TupleLenMismatch { .. });
}

#[test]
fn unresolved_param_error() {
    let mut substitutions = Substitutions::<Num>::default();
    let err = substitutions
        .unify_lengths(
            SimpleTupleLen::Param(2).into(),
            TupleLen::from(1),
            Assignment,
        )
        .unwrap_err();
    assert_matches!(err, TypeErrorKind::UnresolvedParam);

    let err = substitutions
        .unify(&ValueType::Param(2), &ValueType::NUM)
        .unwrap_err();
    assert_matches!(err, TypeErrorKind::UnresolvedParam);
}

#[test]
fn unifying_complex_tuples() {
    let xs = Tuple::new(
        vec![ValueType::NUM, ValueType::NUM],
        ValueType::NUM.repeat(SimpleTupleLen::Var(0)),
        vec![],
    );
    let ys = Tuple::from(ValueType::NUM.repeat(SimpleTupleLen::Var(1) + 2));

    let mut substitutions = Substitutions::<Num>::default();
    substitutions.unify_tuples(&xs, &ys, Assignment).unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&0], SimpleTupleLen::Var(1).into());

    let zs = Tuple::from(ValueType::Var(0).repeat(SimpleTupleLen::Var(1)));
    let mut substitutions = Substitutions::<Num>::default();
    substitutions.unify_tuples(&xs, &zs, Assignment).unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&1], SimpleTupleLen::Var(0) + 2);
    assert_eq!(substitutions.eqs.len(), 1);
    assert_eq!(substitutions.eqs[&0], ValueType::NUM);

    let us = Tuple::new(
        vec![],
        ValueType::NUM.repeat(SimpleTupleLen::Var(1)),
        vec![ValueType::Var(0), ValueType::Var(1), ValueType::Var(2)],
    );
    let mut substitutions = Substitutions::<Num>::default();
    substitutions.unify_tuples(&xs, &us, Assignment).unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&0], SimpleTupleLen::Var(1) + 1);
    assert_eq!(substitutions.eqs.len(), 3);
    for i in 0..3 {
        assert_eq!(substitutions.eqs[&i], ValueType::NUM);
    }
}
