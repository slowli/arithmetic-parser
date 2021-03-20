//! Unit tests for `Substitutions`.

use super::*;
use crate::Num;

#[test]
fn unifying_lengths_success_without_side_effects() {
    const SAMPLES: &[(TupleLength, TupleLength)] = &[
        (TupleLength::Exact(2), TupleLength::Exact(2)),
        // `Dynamic` should always be accepted on LHS.
        (TupleLength::Dynamic, TupleLength::Dynamic),
        (TupleLength::Dynamic, TupleLength::Exact(2)),
        (TupleLength::Dynamic, TupleLength::Var(3)),
    ];

    let mut substitutions = Substitutions::<Num>::default();
    for &(lhs, rhs) in SAMPLES {
        substitutions.unify_lengths(lhs, rhs).unwrap();
        assert!(substitutions.length_eqs.is_empty());
    }
}

#[test]
fn unifying_lengths_success_with_new_equation() {
    const LEN_SAMPLES: &[TupleLength] = &[
        TupleLength::Exact(3),
        TupleLength::Var(5),
        TupleLength::Dynamic,
    ];

    for &rhs in LEN_SAMPLES {
        let mut substitutions = Substitutions::<Num>::default();
        substitutions
            .unify_lengths(TupleLength::Var(2), rhs)
            .unwrap();
        assert_eq!(substitutions.length_eqs.len(), 1);
        assert_eq!(substitutions.length_eqs[&2], rhs);
    }

    let mut substitutions = Substitutions::<Num>::default();
    substitutions
        .unify_lengths(TupleLength::Exact(3), TupleLength::Var(2))
        .unwrap();
    assert_eq!(substitutions.length_eqs.len(), 1);
    assert_eq!(substitutions.length_eqs[&2], TupleLength::Exact(3));
}
