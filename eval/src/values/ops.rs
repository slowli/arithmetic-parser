//! Operations on `Value`s.

use hashbrown::HashMap;

use core::cmp::Ordering;

use crate::{
    alloc::{String, Vec},
    arith::OrdArithmetic,
    error::{AuxErrorInfo, TupleLenMismatchContext},
    Error, ErrorKind, ModuleId, Value,
};
use arithmetic_parser::{BinaryOp, MaybeSpanned, Op, UnaryOp};

#[derive(Debug, Clone, Copy)]
enum OpSide {
    Lhs,
    Rhs,
}

#[derive(Debug)]
struct BinaryOpError {
    inner: ErrorKind,
    side: Option<OpSide>,
}

impl BinaryOpError {
    fn new(op: BinaryOp) -> Self {
        Self {
            inner: ErrorKind::UnexpectedOperand { op: Op::Binary(op) },
            side: None,
        }
    }

    fn tuple(op: BinaryOp, lhs: usize, rhs: usize) -> Self {
        Self {
            inner: ErrorKind::TupleLenMismatch {
                lhs: lhs.into(),
                rhs,
                context: TupleLenMismatchContext::BinaryOp(op),
            },
            side: Some(OpSide::Lhs),
        }
    }

    fn object<T>(op: BinaryOp, lhs: HashMap<String, T>, rhs: HashMap<String, T>) -> Self {
        Self {
            inner: ErrorKind::FieldsMismatch {
                lhs_fields: lhs.into_iter().map(|(name, _)| name).collect(),
                rhs_fields: rhs.into_iter().map(|(name, _)| name).collect(),
                op,
            },
            side: Some(OpSide::Lhs),
        }
    }

    fn with_side(mut self, side: OpSide) -> Self {
        self.side = Some(side);
        self
    }

    fn with_error_kind(mut self, error_kind: ErrorKind) -> Self {
        self.inner = error_kind;
        self
    }

    fn span<'a>(
        self,
        module_id: &dyn ModuleId,
        total_span: MaybeSpanned<'a>,
        lhs_span: MaybeSpanned<'a>,
        rhs_span: MaybeSpanned<'a>,
    ) -> Error<'a> {
        let main_span = match self.side {
            Some(OpSide::Lhs) => lhs_span,
            Some(OpSide::Rhs) => rhs_span,
            None => total_span,
        };

        let aux_info = match &self.inner {
            ErrorKind::TupleLenMismatch { rhs, .. } => Some(AuxErrorInfo::UnbalancedRhsTuple(*rhs)),
            ErrorKind::FieldsMismatch { rhs_fields, .. } => {
                Some(AuxErrorInfo::UnbalancedRhsObject(rhs_fields.clone()))
            }
            _ => None,
        };

        let mut err = Error::new(module_id, &main_span, self.inner);
        if let Some(aux_info) = aux_info {
            err = err.with_span(&rhs_span, aux_info);
        }
        err
    }
}

impl<'a, T: Clone> Value<'a, T> {
    fn try_binary_op_inner(
        self,
        rhs: Self,
        op: BinaryOp,
        arithmetic: &dyn OrdArithmetic<T>,
    ) -> Result<Self, BinaryOpError> {
        match (self, rhs) {
            (Self::Prim(this), Self::Prim(other)) => {
                let op_result = match op {
                    BinaryOp::Add => arithmetic.add(this, other),
                    BinaryOp::Sub => arithmetic.sub(this, other),
                    BinaryOp::Mul => arithmetic.mul(this, other),
                    BinaryOp::Div => arithmetic.div(this, other),
                    BinaryOp::Power => arithmetic.pow(this, other),
                    _ => unreachable!(),
                };
                op_result
                    .map(Self::Prim)
                    .map_err(|e| BinaryOpError::new(op).with_error_kind(ErrorKind::Arithmetic(e)))
            }

            (this @ Self::Prim(_), Self::Tuple(other)) => {
                let output: Result<Vec<_>, _> = other
                    .into_iter()
                    .map(|y| this.clone().try_binary_op_inner(y, op, arithmetic))
                    .collect();
                output.map(Self::Tuple)
            }
            (Self::Tuple(this), other @ Self::Prim(_)) => {
                let output: Result<Vec<_>, _> = this
                    .into_iter()
                    .map(|x| x.try_binary_op_inner(other.clone(), op, arithmetic))
                    .collect();
                output.map(Self::Tuple)
            }

            (Self::Tuple(this), Self::Tuple(other)) => {
                if this.len() == other.len() {
                    let output: Result<Vec<_>, _> = this
                        .into_iter()
                        .zip(other)
                        .map(|(x, y)| x.try_binary_op_inner(y, op, arithmetic))
                        .collect();
                    output.map(Self::Tuple)
                } else {
                    Err(BinaryOpError::tuple(op, this.len(), other.len()))
                }
            }

            (this @ Self::Prim(_), Self::Object(other)) => {
                let output: Result<HashMap<_, _>, _> = other
                    .into_iter()
                    .map(|(name, y)| {
                        this.clone()
                            .try_binary_op_inner(y, op, arithmetic)
                            .map(|res| (name, res))
                    })
                    .collect();
                output.map(Self::Object)
            }
            (Self::Object(this), other @ Self::Prim(_)) => {
                let output: Result<HashMap<_, _>, _> = this
                    .into_iter()
                    .map(|(name, x)| {
                        x.try_binary_op_inner(other.clone(), op, arithmetic)
                            .map(|res| (name, res))
                    })
                    .collect();
                output.map(Self::Object)
            }

            (Self::Object(this), Self::Object(mut other)) => {
                let same_keys =
                    this.len() == other.len() && this.keys().all(|key| other.contains_key(key));
                if same_keys {
                    let output: Result<HashMap<_, _>, _> = this
                        .into_iter()
                        .map(|(name, x)| {
                            let y = other.remove(&name).unwrap();
                            // ^ `unwrap` safety was checked previously
                            x.try_binary_op_inner(y, op, arithmetic)
                                .map(|res| (name, res))
                        })
                        .collect();
                    output.map(Self::Object)
                } else {
                    Err(BinaryOpError::object(op, this, other))
                }
            }

            (Self::Prim(_), _) | (Self::Tuple(_), _) => {
                Err(BinaryOpError::new(op).with_side(OpSide::Rhs))
            }
            _ => Err(BinaryOpError::new(op).with_side(OpSide::Lhs)),
        }
    }

    #[inline]
    pub(crate) fn try_binary_op(
        module_id: &dyn ModuleId,
        total_span: MaybeSpanned<'a>,
        lhs: MaybeSpanned<'a, Self>,
        rhs: MaybeSpanned<'a, Self>,
        op: BinaryOp,
        arithmetic: &dyn OrdArithmetic<T>,
    ) -> Result<Self, Error<'a>> {
        let lhs_span = lhs.with_no_extra();
        let rhs_span = rhs.with_no_extra();
        lhs.extra
            .try_binary_op_inner(rhs.extra, op, arithmetic)
            .map_err(|e| e.span(module_id, total_span, lhs_span, rhs_span))
    }
}

impl<'a, T> Value<'a, T> {
    pub(crate) fn try_neg(self, arithmetic: &dyn OrdArithmetic<T>) -> Result<Self, ErrorKind> {
        match self {
            Self::Prim(val) => arithmetic
                .neg(val)
                .map(Self::Prim)
                .map_err(ErrorKind::Arithmetic),

            Self::Tuple(tuple) => {
                let res: Result<Vec<_>, _> = tuple
                    .into_iter()
                    .map(|elem| Value::try_neg(elem, arithmetic))
                    .collect();
                res.map(Self::Tuple)
            }

            _ => Err(ErrorKind::UnexpectedOperand {
                op: UnaryOp::Neg.into(),
            }),
        }
    }

    pub(crate) fn try_not(self) -> Result<Self, ErrorKind> {
        match self {
            Self::Bool(val) => Ok(Self::Bool(!val)),
            Self::Tuple(tuple) => {
                let res: Result<Vec<_>, _> = tuple.into_iter().map(Value::try_not).collect();
                res.map(Self::Tuple)
            }

            _ => Err(ErrorKind::UnexpectedOperand {
                op: UnaryOp::Not.into(),
            }),
        }
    }

    // **NB.** Must match `PartialEq` impl for `Value`!
    pub(crate) fn eq_by_arithmetic(&self, rhs: &Self, arithmetic: &dyn OrdArithmetic<T>) -> bool {
        match (self, rhs) {
            (Self::Prim(this), Self::Prim(other)) => arithmetic.eq(this, other),
            (Self::Bool(this), Self::Bool(other)) => this == other,
            (Self::Tuple(this), Self::Tuple(other)) => {
                if this.len() == other.len() {
                    this.iter()
                        .zip(other.iter())
                        .all(|(x, y)| x.eq_by_arithmetic(y, arithmetic))
                } else {
                    false
                }
            }
            (Self::Object(this), Self::Object(that)) => {
                if this.len() == that.len() {
                    for (field_name, this_elem) in this {
                        let that_elem = match that.get(field_name) {
                            Some(elem) => elem,
                            None => return false,
                        };
                        if !this_elem.eq_by_arithmetic(that_elem, arithmetic) {
                            return false;
                        }
                    }
                    true
                } else {
                    false
                }
            }
            (Self::Function(this), Self::Function(other)) => this.is_same_function(other),
            (Self::Ref(this), Self::Ref(other)) => this == other,
            _ => false,
        }
    }

    pub(crate) fn compare(
        module_id: &dyn ModuleId,
        lhs: &MaybeSpanned<'a, Self>,
        rhs: &MaybeSpanned<'a, Self>,
        op: BinaryOp,
        arithmetic: &dyn OrdArithmetic<T>,
    ) -> Result<Self, Error<'a>> {
        // We only know how to compare primitive values.
        let lhs_value = match &lhs.extra {
            Value::Prim(value) => value,
            _ => return Err(Error::new(module_id, &lhs, ErrorKind::CannotCompare)),
        };
        let rhs_value = match &rhs.extra {
            Value::Prim(value) => value,
            _ => return Err(Error::new(module_id, &rhs, ErrorKind::CannotCompare)),
        };

        let maybe_ordering = arithmetic.partial_cmp(lhs_value, rhs_value);
        let cmp_result = maybe_ordering.map_or(false, |ordering| match op {
            BinaryOp::Gt => ordering == Ordering::Greater,
            BinaryOp::Lt => ordering == Ordering::Less,
            BinaryOp::Ge => ordering != Ordering::Less,
            BinaryOp::Le => ordering != Ordering::Greater,
            _ => unreachable!(),
        });
        Ok(Value::Bool(cmp_result))
    }

    pub(crate) fn try_and(
        module_id: &dyn ModuleId,
        lhs: &MaybeSpanned<'a, Self>,
        rhs: &MaybeSpanned<'a, Self>,
    ) -> Result<Self, Error<'a>> {
        match (&lhs.extra, &rhs.extra) {
            (Value::Bool(this), Value::Bool(other)) => Ok(Value::Bool(*this && *other)),
            (Value::Bool(_), _) => {
                let err = ErrorKind::UnexpectedOperand {
                    op: BinaryOp::And.into(),
                };
                Err(Error::new(module_id, &rhs, err))
            }
            _ => {
                let err = ErrorKind::UnexpectedOperand {
                    op: BinaryOp::And.into(),
                };
                Err(Error::new(module_id, &lhs, err))
            }
        }
    }

    pub(crate) fn try_or(
        module_id: &dyn ModuleId,
        lhs: &MaybeSpanned<'a, Self>,
        rhs: &MaybeSpanned<'a, Self>,
    ) -> Result<Self, Error<'a>> {
        match (&lhs.extra, &rhs.extra) {
            (Value::Bool(this), Value::Bool(other)) => Ok(Value::Bool(*this || *other)),
            (Value::Bool(_), _) => {
                let err = ErrorKind::UnexpectedOperand {
                    op: BinaryOp::Or.into(),
                };
                Err(Error::new(module_id, &rhs, err))
            }
            _ => {
                let err = ErrorKind::UnexpectedOperand {
                    op: BinaryOp::Or.into(),
                };
                Err(Error::new(module_id, &lhs, err))
            }
        }
    }
}
