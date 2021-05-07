//! `ErrorLocation` and related functionality.

use crate::{
    ast::{SpannedTypeAst, TupleAst, TypeAst},
    TupleIndex,
};
use arithmetic_parser::{
    grammars::Grammar, Destructure, DestructureRest, Expr, Lvalue, Spanned, SpannedExpr,
    SpannedLvalue,
};

impl TupleIndex {
    fn get_from_tuple<'r, 'a>(self, tuple: &'r TupleAst<'a>) -> Option<&'r SpannedTypeAst<'a>> {
        match self {
            Self::Start(i) => tuple.start.get(i),
            Self::Middle => tuple
                .middle
                .as_ref()
                .map(|middle| middle.extra.element.as_ref()),
            Self::End(i) => tuple.end.get(i),
        }
    }

    fn get_from_destructure<'r, 'a>(
        self,
        destructure: &'r Destructure<'a, TypeAst<'a>>,
    ) -> Option<LvalueTree<'r, 'a>> {
        match self {
            Self::Start(i) => destructure.start.get(i).map(LvalueTree::Lvalue),
            Self::Middle => destructure
                .middle
                .as_ref()
                .and_then(|middle| match &middle.extra {
                    DestructureRest::Named { ty: Some(ty), .. } => Some(LvalueTree::Type(ty)),
                    DestructureRest::Named { variable, .. } => Some(LvalueTree::Just(*variable)),
                    _ => None,
                }),
            Self::End(i) => destructure.end.get(i).map(LvalueTree::Lvalue),
        }
    }
}

/// Fragment of an error location.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ErrorLocation {
    /// Function argument with the specified index (0-based; can be `None` if the error cannot
    /// be attributed to a specific index).
    FnArg(Option<TupleIndex>),
    /// Function return type.
    FnReturnType,
    /// Tuple element with the specified index (0-based; can be `None` if the error cannot
    /// be attributed to a specific index).
    TupleElement(Option<TupleIndex>),
    /// Left-hand side of a binary operation.
    Lhs,
    /// Right-hand side of a binary operation.
    Rhs,
}

impl From<TupleIndex> for ErrorLocation {
    fn from(index: TupleIndex) -> Self {
        Self::TupleElement(Some(index))
    }
}

impl ErrorLocation {
    /// Walks the provided `expr` and returns the most exact span found in it.
    pub(super) fn walk_expr<'a, T: Grammar<'a>>(
        location: &[Self],
        expr: &SpannedExpr<'a, T>,
    ) -> Spanned<'a> {
        Self::walk(location, expr, Self::step_into_expr).with_no_extra()
    }

    fn walk<T: Copy>(mut location: &[Self], init: T, refine: impl Fn(Self, T) -> Option<T>) -> T {
        let mut refined = init;
        while !location.is_empty() {
            if let Some(refinement) = refine(location[0], refined) {
                refined = refinement;
                location = &location[1..];
            } else {
                break;
            }
        }
        refined
    }

    fn step_into_expr<'r, 'a, T: Grammar<'a>>(
        self,
        expr: &'r SpannedExpr<'a, T>,
    ) -> Option<&'r SpannedExpr<'a, T>> {
        match self {
            // `TupleIndex::FromEnd` should not occur in this context.
            Self::FnArg(Some(TupleIndex::Start(index))) => match &expr.extra {
                Expr::Function { args, .. } => Some(&args[index]),
                Expr::Method { receiver, args, .. } => Some(if index == 0 {
                    receiver.as_ref()
                } else {
                    &args[index - 1]
                }),
                _ => None,
            },

            Self::Lhs => {
                if let Expr::Binary { lhs, .. } = &expr.extra {
                    Some(lhs.as_ref())
                } else {
                    None
                }
            }
            Self::Rhs => {
                if let Expr::Binary { rhs, .. } = &expr.extra {
                    Some(rhs.as_ref())
                } else {
                    None
                }
            }

            Self::TupleElement(Some(TupleIndex::Start(index))) => {
                if let Expr::Tuple(elements) = &expr.extra {
                    Some(&elements[index])
                } else {
                    None
                }
            }

            _ => None,
        }
    }

    pub(super) fn walk_lvalue<'a>(
        location: &[Self],
        lvalue: &SpannedLvalue<'a, TypeAst<'a>>,
    ) -> Spanned<'a> {
        Self::walk(location, LvalueTree::Lvalue(lvalue), Self::step_into_lvalue)
            .refine_lvalue()
            .with_no_extra()
    }

    pub(super) fn walk_destructure<'a>(
        location: &[Self],
        destructure: &Spanned<'a, Destructure<'a, TypeAst<'a>>>,
    ) -> Spanned<'a> {
        let destructure = LvalueTree::Destructure(destructure);
        Self::walk(location, destructure, Self::step_into_lvalue)
            .refine_lvalue()
            .with_no_extra()
    }

    fn step_into_lvalue<'r, 'a>(self, lvalue: LvalueTree<'r, 'a>) -> Option<LvalueTree<'r, 'a>> {
        match lvalue {
            LvalueTree::Type(ty) => self.step_into_type(&ty.extra),
            LvalueTree::Destructure(destructure) => self.step_into_destructure(&destructure.extra),
            LvalueTree::Lvalue(lvalue) => match &lvalue.extra {
                Lvalue::Tuple(destructure) => self.step_into_destructure(destructure),
                Lvalue::Variable { ty: Some(ty) } => self.step_into_type(&ty.extra),
                _ => None,
            },
            LvalueTree::Just(_) => None,
        }
    }

    fn step_into_type<'r, 'a>(self, ty: &'r TypeAst<'a>) -> Option<LvalueTree<'r, 'a>> {
        match (self, ty) {
            (Self::TupleElement(Some(i)), TypeAst::Tuple(tuple)) => {
                i.get_from_tuple(tuple).map(LvalueTree::Type)
            }
            (Self::TupleElement(Some(TupleIndex::Middle)), TypeAst::Slice(slice)) => {
                Some(LvalueTree::Type(&slice.element))
            }
            _ => None,
        }
    }

    fn step_into_destructure<'r, 'a>(
        self,
        destructure: &'r Destructure<'a, TypeAst<'a>>,
    ) -> Option<LvalueTree<'r, 'a>> {
        match self {
            Self::TupleElement(Some(i)) => i.get_from_destructure(destructure),
            _ => None,
        }
    }
}

/// Enumeration of all types encountered on the lvalue side of assignments.
#[derive(Debug, Clone, Copy)]
enum LvalueTree<'r, 'a> {
    Lvalue(&'r SpannedLvalue<'a, TypeAst<'a>>),
    Destructure(&'r Spanned<'a, Destructure<'a, TypeAst<'a>>>),
    Type(&'r Spanned<'a, TypeAst<'a>>),
    Just(Spanned<'a>),
}

impl<'a> LvalueTree<'_, 'a> {
    fn with_no_extra(self) -> Spanned<'a> {
        match self {
            Self::Lvalue(lvalue) => lvalue.with_no_extra(),
            Self::Destructure(destructure) => destructure.with_no_extra(),
            Self::Type(ty) => ty.with_no_extra(),
            Self::Just(spanned) => spanned,
        }
    }

    /// Refines the `Lvalue` variant if it is a variable with an annotation.
    fn refine_lvalue(self) -> Self {
        if let Self::Lvalue(lvalue) = self {
            if let Lvalue::Variable { ty: Some(ty) } = &lvalue.extra {
                return Self::Type(ty);
            }
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Annotated;

    use arithmetic_parser::{
        grammars::{NumGrammar, Parse, Typed},
        Statement,
    };

    type F32Grammar = Annotated<NumGrammar<f32>>;

    fn parse_expr(code: &str) -> SpannedExpr<'_, F32Grammar> {
        *Typed::<F32Grammar>::parse_statements(code)
            .unwrap()
            .return_value
            .unwrap()
    }

    fn parse_lvalue(code: &str) -> SpannedLvalue<'_, TypeAst<'_>> {
        let statement = Typed::<F32Grammar>::parse_statements(code)
            .unwrap()
            .statements
            .pop()
            .unwrap()
            .extra;
        match statement {
            Statement::Assignment { lhs, .. } => lhs,
            _ => panic!("Unexpected statement type: {:?}", statement),
        }
    }

    #[test]
    fn walking_simple_expr() {
        let expr = parse_expr("1 + (2, x)");
        let location = &[ErrorLocation::Rhs, TupleIndex::Start(1).into()];
        let located = ErrorLocation::walk_expr(location, &expr);

        assert_eq!(*located.fragment(), "x");
    }

    #[test]
    fn walking_expr_with_fn_call() {
        let expr = parse_expr("hash(1, (false, 2), x)");
        let location = &[
            ErrorLocation::FnArg(Some(TupleIndex::Start(1))),
            TupleIndex::Start(0).into(),
        ];
        let located = ErrorLocation::walk_expr(location, &expr);

        assert_eq!(*located.fragment(), "false");
    }

    #[test]
    fn walking_expr_with_method_call() {
        let expr = parse_expr("xs.map(|x| x + 1)");
        let location = &[ErrorLocation::FnArg(Some(TupleIndex::Start(0)))];
        let located = ErrorLocation::walk_expr(location, &expr);

        assert_eq!(*located.fragment(), "xs");

        let other_location = &[ErrorLocation::FnArg(Some(TupleIndex::Start(1)))];
        let other_located = ErrorLocation::walk_expr(other_location, &expr);

        assert_eq!(*other_located.fragment(), "|x| x + 1");
    }

    #[test]
    fn walking_expr_with_partial_match() {
        let expr = parse_expr("hash(1, xs)");
        let location = &[
            ErrorLocation::FnArg(Some(TupleIndex::Start(1))),
            TupleIndex::Start(0).into(),
        ];
        let located = ErrorLocation::walk_expr(location, &expr);

        assert_eq!(*located.fragment(), "xs");
    }

    #[test]
    fn walking_lvalue() {
        let lvalue = parse_lvalue("((u, v), ...ys, _, z) = x;");
        let start_location = &[ErrorLocation::from(TupleIndex::Start(0))];
        let located_start = ErrorLocation::walk_lvalue(start_location, &lvalue);
        assert_eq!(*located_start.fragment(), "(u, v)");

        let embedded_location = &[
            ErrorLocation::from(TupleIndex::Start(0)),
            ErrorLocation::from(TupleIndex::Start(1)),
        ];
        let embedded = ErrorLocation::walk_lvalue(embedded_location, &lvalue);
        assert_eq!(*embedded.fragment(), "v");

        let middle_location = &[ErrorLocation::from(TupleIndex::Middle)];
        let located_middle = ErrorLocation::walk_lvalue(middle_location, &lvalue);
        assert_eq!(*located_middle.fragment(), "ys");

        let end_location = &[ErrorLocation::from(TupleIndex::End(1))];
        let located_end = ErrorLocation::walk_lvalue(end_location, &lvalue);
        assert_eq!(*located_end.fragment(), "z");
    }

    #[test]
    fn walking_lvalue_with_annotations() {
        let lvalue = parse_lvalue("x: (Bool, ...[(Num, Bool); _]) = x;");
        let start_location = &[ErrorLocation::from(TupleIndex::Start(0))];
        let located_start = ErrorLocation::walk_lvalue(start_location, &lvalue);
        assert_eq!(*located_start.fragment(), "Bool");

        let middle_location = &[ErrorLocation::from(TupleIndex::Middle)];
        let located_middle = ErrorLocation::walk_lvalue(middle_location, &lvalue);
        assert_eq!(*located_middle.fragment(), "(Num, Bool)");

        let narrowed_location = &[
            ErrorLocation::from(TupleIndex::Middle),
            TupleIndex::Start(0).into(),
        ];
        let located_ty = ErrorLocation::walk_lvalue(narrowed_location, &lvalue);
        assert_eq!(*located_ty.fragment(), "Num");
    }

    #[test]
    fn walking_lvalue_with_annotation_mix() {
        let lvalue = parse_lvalue("(flag, y: [Num]) = x;");
        let start_location = &[ErrorLocation::from(TupleIndex::Start(0))];
        let located_start = ErrorLocation::walk_lvalue(start_location, &lvalue);
        assert_eq!(*located_start.fragment(), "flag");

        let slice_location = &[ErrorLocation::from(TupleIndex::Start(1))];
        let located_slice = ErrorLocation::walk_lvalue(slice_location, &lvalue);
        assert_eq!(*located_slice.fragment(), "[Num]");

        let slice_elem_location = &[
            ErrorLocation::from(TupleIndex::Start(1)),
            ErrorLocation::from(TupleIndex::Middle),
        ];
        let located_slice_elem = ErrorLocation::walk_lvalue(slice_elem_location, &lvalue);
        assert_eq!(*located_slice_elem.fragment(), "Num");
    }

    #[test]
    fn walking_slice() {
        let lvalue = parse_lvalue("xs: [(Num, Bool); _] = x;");
        let slice_location = &[ErrorLocation::from(TupleIndex::Middle)];
        let located_slice = ErrorLocation::walk_lvalue(slice_location, &lvalue);
        assert_eq!(*located_slice.fragment(), "(Num, Bool)");

        let narrow_location = &[
            ErrorLocation::from(TupleIndex::Middle),
            ErrorLocation::from(TupleIndex::Start(1)),
        ];
        let located_elem = ErrorLocation::walk_lvalue(narrow_location, &lvalue);
        assert_eq!(*located_elem.fragment(), "Bool");
    }
}
