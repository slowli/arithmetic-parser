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
    ) -> Option<SpannedLvalueTree<'r, 'a>> {
        match self {
            Self::Start(i) => destructure.start.get(i).map(LvalueTree::from_lvalue),
            Self::Middle => destructure
                .middle
                .as_ref()
                .and_then(|middle| match &middle.extra {
                    DestructureRest::Named { ty: Some(ty), .. } => Some(LvalueTree::from_span(ty)),
                    DestructureRest::Named { variable, .. } => {
                        Some(LvalueTree::from_span(variable))
                    }
                    DestructureRest::Unnamed => None,
                }),
            Self::End(i) => destructure.end.get(i).map(LvalueTree::from_lvalue),
        }
    }
}

/// Fragment of a path for an [`Error`](crate::error::Error).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorPathFragment {
    /// Function argument with the specified index (0-based; can be `None` if the error cannot
    /// be attributed to a specific index).
    FnArg(Option<TupleIndex>),
    /// Function return type.
    FnReturnType,
    /// Tuple element with the specified index (0-based; can be `None` if the error cannot
    /// be attributed to a specific index).
    TupleElement(Option<TupleIndex>),
    /// Object field with the specified name.
    ObjectField(String),
    /// Left-hand side of a binary operation.
    Lhs,
    /// Right-hand side of a binary operation.
    Rhs,
}

impl From<TupleIndex> for ErrorPathFragment {
    fn from(index: TupleIndex) -> Self {
        Self::TupleElement(Some(index))
    }
}

impl From<&str> for ErrorPathFragment {
    fn from(field_name: &str) -> Self {
        Self::ObjectField(field_name.to_owned())
    }
}

impl ErrorPathFragment {
    /// Walks the provided `expr` and returns the most exact span found in it.
    pub(super) fn walk_expr<'a, T: Grammar>(
        path: &[Self],
        expr: &SpannedExpr<'a, T>,
    ) -> Spanned<'a> {
        let mut refined = Self::walk(path, expr, Self::step_into_expr);
        while let Expr::TypeCast { value, .. } = &refined.extra {
            refined = value.as_ref();
        }
        refined.with_no_extra()
    }

    fn walk<T: Copy>(mut path: &[Self], init: T, refine: impl Fn(&Self, T) -> Option<T>) -> T {
        let mut refined = init;
        while !path.is_empty() {
            if let Some(refinement) = refine(&path[0], refined) {
                refined = refinement;
                path = &path[1..];
            } else {
                break;
            }
        }
        refined
    }

    fn step_into_expr<'r, 'a, T: Grammar>(
        &self,
        mut expr: &'r SpannedExpr<'a, T>,
    ) -> Option<&'r SpannedExpr<'a, T>> {
        while let Expr::TypeCast { value, .. } = &expr.extra {
            expr = value.as_ref();
        }

        match self {
            // `TupleIndex::FromEnd` should not occur in this context.
            Self::FnArg(Some(TupleIndex::Start(index))) => match &expr.extra {
                Expr::Function { args, .. } => Some(&args[*index]),
                Expr::Method { receiver, args, .. } => Some(if *index == 0 {
                    receiver.as_ref()
                } else {
                    &args[*index - 1]
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
                    Some(&elements[*index])
                } else {
                    None
                }
            }

            _ => None,
        }
    }

    pub(super) fn walk_lvalue<'a>(
        path: &[Self],
        lvalue: &SpannedLvalue<'a, TypeAst<'a>>,
    ) -> Spanned<'a> {
        let lvalue = LvalueTree::from_lvalue(lvalue);
        Self::walk(path, lvalue, Self::step_into_lvalue).with_no_extra()
    }

    pub(super) fn walk_destructure<'a>(
        path: &[Self],
        destructure: &Spanned<'a, Destructure<'a, TypeAst<'a>>>,
    ) -> Spanned<'a> {
        let destructure = LvalueTree::from_span(destructure);
        Self::walk(path, destructure, Self::step_into_lvalue).with_no_extra()
    }

    fn step_into_lvalue<'r, 'a>(
        &self,
        lvalue: SpannedLvalueTree<'r, 'a>,
    ) -> Option<SpannedLvalueTree<'r, 'a>> {
        match lvalue.extra {
            LvalueTree::Type(ty) => self.step_into_type(ty),
            LvalueTree::Destructure(destructure) => self.step_into_destructure(destructure),
            LvalueTree::JustSpan => None,
        }
    }

    fn step_into_type<'r, 'a>(&self, ty: &'r TypeAst<'a>) -> Option<SpannedLvalueTree<'r, 'a>> {
        match (self, ty) {
            (Self::TupleElement(Some(i)), TypeAst::Tuple(tuple)) => {
                i.get_from_tuple(tuple).map(LvalueTree::from_span)
            }
            (Self::TupleElement(Some(TupleIndex::Middle)), TypeAst::Slice(slice)) => {
                Some(LvalueTree::from_span(&slice.element))
            }
            _ => None,
        }
    }

    fn step_into_destructure<'r, 'a>(
        &self,
        destructure: &'r Destructure<'a, TypeAst<'a>>,
    ) -> Option<SpannedLvalueTree<'r, 'a>> {
        match self {
            Self::TupleElement(Some(i)) => i.get_from_destructure(destructure),
            _ => None,
        }
    }
}

/// Enumeration of all types encountered on the lvalue side of assignments.
#[derive(Debug, Clone, Copy)]
enum LvalueTree<'r, 'a> {
    Destructure(&'r Destructure<'a, TypeAst<'a>>),
    Type(&'r TypeAst<'a>),
    JustSpan,
}

type SpannedLvalueTree<'r, 'a> = Spanned<'a, LvalueTree<'r, 'a>>;

impl<'r, 'a> From<&'r Destructure<'a, TypeAst<'a>>> for LvalueTree<'r, 'a> {
    fn from(destructure: &'r Destructure<'a, TypeAst<'a>>) -> Self {
        Self::Destructure(destructure)
    }
}

impl<'r, 'a> From<&'r TypeAst<'a>> for LvalueTree<'r, 'a> {
    fn from(ty: &'r TypeAst<'a>) -> Self {
        Self::Type(ty)
    }
}

impl<'r> From<&'r ()> for LvalueTree<'r, '_> {
    fn from(_: &'r ()) -> Self {
        Self::JustSpan
    }
}

impl<'r, 'a> LvalueTree<'r, 'a> {
    fn from_lvalue(lvalue: &'r Spanned<'a, Lvalue<'a, TypeAst<'a>>>) -> SpannedLvalueTree<'r, 'a> {
        match &lvalue.extra {
            Lvalue::Tuple(destructure) => lvalue.copy_with_extra(Self::Destructure(destructure)),
            Lvalue::Variable { ty: Some(ty) } => ty.as_ref().map_extra(Self::Type),
            _ => lvalue.copy_with_extra(Self::JustSpan),
        }
    }

    fn from_span<T>(spanned: &'r Spanned<'a, T>) -> SpannedLvalueTree<'r, 'a>
    where
        &'r T: Into<Self>,
    {
        spanned.as_ref().map_extra(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Annotated;

    use arithmetic_parser::{
        grammars::{NumGrammar, Parse},
        Statement,
    };

    type F32Grammar = Annotated<NumGrammar<f32>>;

    fn parse_expr(code: &str) -> SpannedExpr<'_, F32Grammar> {
        *F32Grammar::parse_statements(code)
            .unwrap()
            .return_value
            .unwrap()
    }

    fn parse_lvalue(code: &str) -> SpannedLvalue<'_, TypeAst<'_>> {
        let statement = F32Grammar::parse_statements(code)
            .unwrap()
            .statements
            .pop()
            .unwrap()
            .extra;
        match statement {
            Statement::Assignment { lhs, .. } => lhs,
            _ => panic!("Unexpected statement type: {statement:?}"),
        }
    }

    #[test]
    fn walking_simple_expr() {
        let expr = parse_expr("1 + (2, x)");
        let path = &[ErrorPathFragment::Rhs, TupleIndex::Start(1).into()];
        let located = ErrorPathFragment::walk_expr(path, &expr);

        assert_eq!(*located.fragment(), "x");
    }

    #[test]
    fn walking_expr_with_fn_call() {
        let expr = parse_expr("hash(1, (false, 2), x)");
        let path = &[
            ErrorPathFragment::FnArg(Some(TupleIndex::Start(1))),
            TupleIndex::Start(0).into(),
        ];
        let located = ErrorPathFragment::walk_expr(path, &expr);

        assert_eq!(*located.fragment(), "false");
    }

    #[test]
    fn walking_expr_with_method_call() {
        let expr = parse_expr("xs.map(|x| x + 1)");
        let path = &[ErrorPathFragment::FnArg(Some(TupleIndex::Start(0)))];
        let located = ErrorPathFragment::walk_expr(path, &expr);

        assert_eq!(*located.fragment(), "xs");

        let other_path = &[ErrorPathFragment::FnArg(Some(TupleIndex::Start(1)))];
        let other_located = ErrorPathFragment::walk_expr(other_path, &expr);

        assert_eq!(*other_located.fragment(), "|x| x + 1");
    }

    #[test]
    fn walking_expr_with_partial_match() {
        let expr = parse_expr("hash(1, xs)");
        let path = &[
            ErrorPathFragment::FnArg(Some(TupleIndex::Start(1))),
            TupleIndex::Start(0).into(),
        ];
        let located = ErrorPathFragment::walk_expr(path, &expr);

        assert_eq!(*located.fragment(), "xs");
    }

    #[test]
    fn walking_expr_with_intermediate_type_cast() {
        let expr = parse_expr("hash(1, (xs, ys) as Pair)");
        let path = &[
            ErrorPathFragment::FnArg(Some(TupleIndex::Start(1))),
            TupleIndex::Start(0).into(),
        ];
        let located = ErrorPathFragment::walk_expr(path, &expr);

        assert_eq!(*located.fragment(), "xs");
    }

    #[test]
    fn walking_expr_with_final_type_cast() {
        let expr = parse_expr("hash(1, (xs as [_] as Slice, ys))");
        let path = &[
            ErrorPathFragment::FnArg(Some(TupleIndex::Start(1))),
            TupleIndex::Start(0).into(),
        ];
        let located = ErrorPathFragment::walk_expr(path, &expr);

        assert_eq!(*located.fragment(), "xs");
    }

    #[test]
    fn walking_lvalue() {
        let lvalue = parse_lvalue("((u, v), ...ys, _, z) = x;");
        let start_path = &[ErrorPathFragment::from(TupleIndex::Start(0))];
        let located_start = ErrorPathFragment::walk_lvalue(start_path, &lvalue);
        assert_eq!(*located_start.fragment(), "(u, v)");

        let embedded_path = &[
            ErrorPathFragment::from(TupleIndex::Start(0)),
            ErrorPathFragment::from(TupleIndex::Start(1)),
        ];
        let embedded = ErrorPathFragment::walk_lvalue(embedded_path, &lvalue);
        assert_eq!(*embedded.fragment(), "v");

        let middle_path = &[ErrorPathFragment::from(TupleIndex::Middle)];
        let located_middle = ErrorPathFragment::walk_lvalue(middle_path, &lvalue);
        assert_eq!(*located_middle.fragment(), "ys");

        let end_path = &[ErrorPathFragment::from(TupleIndex::End(1))];
        let located_end = ErrorPathFragment::walk_lvalue(end_path, &lvalue);
        assert_eq!(*located_end.fragment(), "z");
    }

    #[test]
    fn walking_lvalue_with_annotations() {
        let lvalue = parse_lvalue("x: (Bool, ...[(Num, Bool); _]) = x;");
        let start_path = &[ErrorPathFragment::from(TupleIndex::Start(0))];
        let located_start = ErrorPathFragment::walk_lvalue(start_path, &lvalue);
        assert_eq!(*located_start.fragment(), "Bool");

        let middle_path = &[ErrorPathFragment::from(TupleIndex::Middle)];
        let located_middle = ErrorPathFragment::walk_lvalue(middle_path, &lvalue);
        assert_eq!(*located_middle.fragment(), "(Num, Bool)");

        let narrowed_path = &[
            ErrorPathFragment::from(TupleIndex::Middle),
            TupleIndex::Start(0).into(),
        ];
        let located_ty = ErrorPathFragment::walk_lvalue(narrowed_path, &lvalue);
        assert_eq!(*located_ty.fragment(), "Num");
    }

    #[test]
    fn walking_lvalue_with_annotation_mix() {
        let lvalue = parse_lvalue("(flag, y: [Num]) = x;");
        let start_path = &[ErrorPathFragment::from(TupleIndex::Start(0))];
        let located_start = ErrorPathFragment::walk_lvalue(start_path, &lvalue);
        assert_eq!(*located_start.fragment(), "flag");

        let slice_path = &[ErrorPathFragment::from(TupleIndex::Start(1))];
        let located_slice = ErrorPathFragment::walk_lvalue(slice_path, &lvalue);
        assert_eq!(*located_slice.fragment(), "[Num]");

        let slice_elem_path = &[
            ErrorPathFragment::from(TupleIndex::Start(1)),
            ErrorPathFragment::from(TupleIndex::Middle),
        ];
        let located_slice_elem = ErrorPathFragment::walk_lvalue(slice_elem_path, &lvalue);
        assert_eq!(*located_slice_elem.fragment(), "Num");
    }

    #[test]
    fn walking_slice() {
        let lvalue = parse_lvalue("xs: [(Num, Bool); _] = x;");
        let slice_path = &[ErrorPathFragment::from(TupleIndex::Middle)];
        let located_slice = ErrorPathFragment::walk_lvalue(slice_path, &lvalue);
        assert_eq!(*located_slice.fragment(), "(Num, Bool)");

        let narrow_path = &[
            ErrorPathFragment::from(TupleIndex::Middle),
            ErrorPathFragment::from(TupleIndex::Start(1)),
        ];
        let located_elem = ErrorPathFragment::walk_lvalue(narrow_path, &lvalue);
        assert_eq!(*located_elem.fragment(), "Bool");
    }
}
