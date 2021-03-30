//! Unit tests for `Substitutions`.

use assert_matches::assert_matches;

use super::*;
use crate::{Num, TupleLenMismatchContext::Assignment};

const DYN: TupleLength = TupleLength::Some { is_dynamic: true };

#[test]
fn unifying_lengths_success_without_side_effects() {
    const SAMPLES: &[(TupleLength, TupleLength)] = &[
        (TupleLength::Exact(2), TupleLength::Exact(2)),
        (TupleLength::Var(1), TupleLength::Var(1)),
        // `Dynamic` should always be accepted on LHS.
        (DYN, TupleLength::Exact(2)),
        (DYN, TupleLength::Var(3)),
    ];

    let mut substitutions = Substitutions::<Num>::default();
    for (lhs, rhs) in SAMPLES {
        let (mut lhs, mut rhs) = (lhs.to_owned(), rhs.to_owned());
        substitutions.assign_new_length(&mut lhs);
        substitutions.assign_new_length(&mut rhs);
        substitutions.unify_lengths(&lhs, &rhs, Assignment).unwrap();

        assert!(substitutions.length_eqs.is_empty());
    }
}

#[test]
fn unifying_lengths_error() {
    const SAMPLES: &[(TupleLength, TupleLength)] = &[
        (TupleLength::Exact(1), TupleLength::Exact(2)),
        (DYN, DYN),
        (TupleLength::Var(0), DYN),
        (TupleLength::Exact(2), DYN),
    ];

    let mut substitutions = Substitutions::<Num>::default();
    for (lhs, rhs) in SAMPLES {
        let (mut lhs, mut rhs) = (lhs.to_owned(), rhs.to_owned());
        substitutions.assign_new_length(&mut lhs);
        substitutions.assign_new_length(&mut rhs);
        let err = substitutions
            .unify_lengths(&lhs, &rhs, Assignment)
            .unwrap_err();

        assert_matches!(err, TypeErrorKind::TupleLenMismatch { .. });
        assert!(substitutions.length_eqs.is_empty());
    }
}

#[test]
fn unifying_lengths_success_with_new_equation() {
    const LEN_SAMPLES: &[TupleLength] = &[TupleLength::Exact(3), TupleLength::Var(5), DYN];

    for rhs in LEN_SAMPLES {
        let mut substitutions = Substitutions::<Num>::default();
        let mut rhs = rhs.to_owned();
        substitutions.assign_new_length(&mut rhs);
        substitutions
            .unify_lengths(&TupleLength::Var(2), &rhs, Assignment)
            .unwrap();

        assert_eq!(substitutions.length_eqs.len(), 1);
        assert_eq!(substitutions.length_eqs[&2], rhs);
    }

    let mut substitutions = Substitutions::<Num>::default();
    substitutions
        .unify_lengths(&TupleLength::Exact(3), &TupleLength::Var(2), Assignment)
        .unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&2], TupleLength::Exact(3));
}

#[test]
fn unifying_compound_length_success() {
    let compound_len = TupleLength::Var(0) + 2;
    let mut substitutions = Substitutions::<Num>::default();
    substitutions
        .unify_lengths(&compound_len, &TupleLength::Exact(5), Assignment)
        .unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&0], TupleLength::Exact(3));

    let mut substitutions = Substitutions::<Num>::default();
    substitutions
        .unify_lengths(&TupleLength::Exact(5), &compound_len, Assignment)
        .unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&0], TupleLength::Exact(3));

    let other_compound_len = TupleLength::Var(1) + 2;
    let mut substitutions = Substitutions::<Num>::default();
    substitutions
        .unify_lengths(&compound_len, &other_compound_len, Assignment)
        .unwrap();

    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&0], TupleLength::Var(1));
}

#[test]
fn unifying_compound_length_with_dyn_length() {
    let compound_len = TupleLength::Var(0) + 2;
    let mut substitutions = Substitutions::<Num>::default();
    substitutions.dyn_lengths.insert(0);

    substitutions
        .unify_lengths(&compound_len, &TupleLength::Exact(5), Assignment)
        .unwrap();
    assert!(substitutions.length_eqs.is_empty());

    substitutions
        .unify_lengths(&compound_len, &compound_len, Assignment)
        .unwrap();
    assert!(substitutions.length_eqs.is_empty());

    let err = substitutions
        .unify_lengths(&TupleLength::Exact(5), &compound_len, Assignment)
        .unwrap_err();
    assert_matches!(err, TypeErrorKind::TupleLenMismatch { .. });

    let other_compound_len = TupleLength::Var(1) + 2;
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
    assert_eq!(substitutions.length_eqs[&1], TupleLength::Var(0));
}

#[test]
fn unifying_compound_length_errors() {
    let compound_len = TupleLength::Var(0) + 2;

    let mut substitutions = Substitutions::<Num>::default();
    let err = substitutions
        .unify_lengths(&compound_len, &TupleLength::Exact(1), Assignment)
        .unwrap_err();

    assert_matches!(err, TypeErrorKind::TupleLenMismatch { .. });

    let mut substitutions = Substitutions::<Num>::default();
    let other_compound_len = TupleLength::Var(1) + 1;
    let err = substitutions
        .unify_lengths(&compound_len, &other_compound_len, Assignment)
        .unwrap_err();

    assert_matches!(err, TypeErrorKind::TupleLenMismatch { .. });
}

#[test]
fn unresolved_param_error() {
    let mut substitutions = Substitutions::<Num>::default();
    let err = substitutions
        .unify_lengths(&TupleLength::Param(2), &TupleLength::Exact(1), Assignment)
        .unwrap_err();
    assert_matches!(err, TypeErrorKind::UnresolvedParam);

    let err = substitutions
        .unify(&ValueType::Param(2), &ValueType::NUM)
        .unwrap_err();
    assert_matches!(err, TypeErrorKind::UnresolvedParam);
}
