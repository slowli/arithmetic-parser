//! Unit tests for `Substitutions`.

use assert_matches::assert_matches;

use super::*;
use crate::{Num, TupleLenMismatchContext::Assignment};

#[test]
fn unifying_lengths_success_without_side_effects() {
    const SAMPLES: &[(TupleLen, TupleLen)] = &[
        (TupleLen::Exact(2), TupleLen::Exact(2)),
        (TupleLen::Var(1), TupleLen::Var(1)),
        // `Dynamic` should always be accepted on LHS.
        (TupleLen::Dynamic, TupleLen::Exact(2)),
        (TupleLen::Dynamic, TupleLen::Var(3)),
    ];

    let mut substitutions = Substitutions::<Num>::default();
    for (lhs, rhs) in SAMPLES {
        let (mut lhs, mut rhs) = (lhs.to_owned(), rhs.to_owned());
        substitutions.assign_new_len(&mut lhs);
        substitutions.assign_new_len(&mut rhs);
        substitutions.unify_lengths(&lhs, &rhs, Assignment).unwrap();

        assert!(substitutions.length_eqs.is_empty());
    }
}

#[test]
fn unifying_lengths_error() {
    const SAMPLES: &[(TupleLen, TupleLen)] = &[
        (TupleLen::Exact(1), TupleLen::Exact(2)),
        (TupleLen::Dynamic, TupleLen::Dynamic),
        (TupleLen::Var(0), TupleLen::Dynamic),
        (TupleLen::Exact(2), TupleLen::Dynamic),
    ];

    let mut substitutions = Substitutions::<Num>::default();
    for (lhs, rhs) in SAMPLES {
        let (mut lhs, mut rhs) = (lhs.to_owned(), rhs.to_owned());
        substitutions.assign_new_len(&mut lhs);
        substitutions.assign_new_len(&mut rhs);
        let err = substitutions
            .unify_lengths(&lhs, &rhs, Assignment)
            .unwrap_err();

        assert_matches!(err, TypeErrorKind::TupleLenMismatch { .. });
        assert!(substitutions.length_eqs.is_empty());
    }
}

#[test]
fn unifying_lengths_success_with_new_equation() {
    const LEN_SAMPLES: &[TupleLen] = &[TupleLen::Exact(3), TupleLen::Var(5), TupleLen::Dynamic];

    for rhs in LEN_SAMPLES {
        let mut substitutions = Substitutions::<Num>::default();
        let mut rhs = rhs.to_owned();
        substitutions.assign_new_len(&mut rhs);
        substitutions
            .unify_lengths(&TupleLen::Var(2), &rhs, Assignment)
            .unwrap();

        assert_eq!(substitutions.length_eqs.len(), 1);
        assert_eq!(substitutions.length_eqs[&2], rhs);
    }

    let mut substitutions = Substitutions::<Num>::default();
    substitutions
        .unify_lengths(&TupleLen::Exact(3), &TupleLen::Var(2), Assignment)
        .unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&2], TupleLen::Exact(3));
}

#[test]
fn unifying_compound_length_success() {
    let compound_len = TupleLen::Var(0) + 2;
    let mut substitutions = Substitutions::<Num>::default();
    substitutions
        .unify_lengths(&compound_len, &TupleLen::Exact(5), Assignment)
        .unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&0], TupleLen::Exact(3));

    let mut substitutions = Substitutions::<Num>::default();
    substitutions
        .unify_lengths(&TupleLen::Exact(5), &compound_len, Assignment)
        .unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&0], TupleLen::Exact(3));

    let other_compound_len = TupleLen::Var(1) + 2;
    let mut substitutions = Substitutions::<Num>::default();
    substitutions
        .unify_lengths(&compound_len, &other_compound_len, Assignment)
        .unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&0], TupleLen::Var(1));
}

#[test]
fn unifying_compound_length_with_dyn_length() {
    let compound_len = TupleLen::Var(0) + 2;
    let mut substitutions = Substitutions::<Num>::default();
    substitutions.dyn_lengths.insert(0);

    substitutions
        .unify_lengths(&compound_len, &TupleLen::Exact(5), Assignment)
        .unwrap();
    assert!(substitutions.length_eqs.is_empty());

    substitutions
        .unify_lengths(&compound_len, &compound_len, Assignment)
        .unwrap();
    assert!(substitutions.length_eqs.is_empty());

    let err = substitutions
        .unify_lengths(&TupleLen::Exact(5), &compound_len, Assignment)
        .unwrap_err();
    assert_matches!(err, TypeErrorKind::TupleLenMismatch { .. });

    let other_compound_len = TupleLen::Var(1) + 2;
    substitutions
        .unify_lengths(&compound_len, &other_compound_len, Assignment)
        .unwrap();
    assert!(substitutions.length_eqs.is_empty());

    substitutions.dyn_lengths.insert(1);
    let err = substitutions
        .unify_lengths(&other_compound_len, &compound_len, Assignment)
        .unwrap_err();
    assert_matches!(err, TypeErrorKind::TupleLenMismatch { .. });

    substitutions.dyn_lengths.remove(&1);
    substitutions
        .unify_lengths(&other_compound_len, &compound_len, Assignment)
        .unwrap();
    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&1], TupleLen::Var(0));
}

#[test]
fn unifying_compound_length_errors() {
    let compound_len = TupleLen::Var(0) + 2;

    let mut substitutions = Substitutions::<Num>::default();
    let err = substitutions
        .unify_lengths(&compound_len, &TupleLen::Exact(1), Assignment)
        .unwrap_err();

    assert_matches!(err, TypeErrorKind::TupleLenMismatch { .. });
}

#[test]
fn unresolved_param_error() {
    let mut substitutions = Substitutions::<Num>::default();
    let err = substitutions
        .unify_lengths(&TupleLen::Param(2), &TupleLen::Exact(1), Assignment)
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
        ValueType::NUM.repeat(TupleLen::Var(0)),
        vec![],
    );
    let ys = Tuple::from(ValueType::NUM.repeat(TupleLen::Var(1) + 2));

    let mut substitutions = Substitutions::<Num>::default();
    substitutions.unify_tuples(&xs, &ys, Assignment).unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&0], TupleLen::Var(1));

    let zs = Tuple::from(ValueType::Var(0).repeat(TupleLen::Var(1)));
    let mut substitutions = Substitutions::<Num>::default();
    substitutions.unify_tuples(&xs, &zs, Assignment).unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&1], TupleLen::Var(0) + 2);
    assert_eq!(substitutions.eqs.len(), 1);
    assert_eq!(substitutions.eqs[&0], ValueType::NUM);

    let us = Tuple::new(
        vec![],
        ValueType::NUM.repeat(TupleLen::Var(1)),
        vec![ValueType::Var(0), ValueType::Var(1), ValueType::Var(2)],
    );
    let mut substitutions = Substitutions::<Num>::default();
    substitutions.unify_tuples(&xs, &us, Assignment).unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&0], TupleLen::Var(1) + 1);
    assert_eq!(substitutions.eqs.len(), 3);
    for i in 0..3 {
        assert_eq!(substitutions.eqs[&i], ValueType::NUM);
    }
}
