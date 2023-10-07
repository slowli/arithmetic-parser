//! Operations on `Value`s.

use core::cmp::Ordering;

use crate::{
    alloc::Arc,
    arith::OrdArithmetic,
    error::{AuxErrorInfo, Error, ErrorKind, TupleLenMismatchContext},
    exec::ModuleId,
    Object, Tuple, Value,
};
use arithmetic_parser::{BinaryOp, Location, Op, UnaryOp};

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

    fn object<T>(op: BinaryOp, lhs: Object<T>, rhs: Object<T>) -> Self {
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

    fn span(
        self,
        module_id: Arc<dyn ModuleId>,
        total_span: Location,
        lhs_span: Location,
        rhs_span: Location,
    ) -> Error {
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
            err = err.with_location(&rhs_span, aux_info);
        }
        err
    }
}

impl<T: Clone> Value<T> {
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
                let output: Result<Tuple<_>, _> = other
                    .into_iter()
                    .map(|y| this.clone().try_binary_op_inner(y, op, arithmetic))
                    .collect();
                output.map(Self::Tuple)
            }
            (Self::Tuple(this), other @ Self::Prim(_)) => {
                let output: Result<Tuple<_>, _> = this
                    .into_iter()
                    .map(|x| x.try_binary_op_inner(other.clone(), op, arithmetic))
                    .collect();
                output.map(Self::Tuple)
            }

            (Self::Tuple(this), Self::Tuple(other)) => {
                if this.len() == other.len() {
                    let output: Result<Tuple<_>, _> = this
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
                let output: Result<Object<_>, _> = other
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
                let output: Result<Object<_>, _> = this
                    .into_iter()
                    .map(|(name, x)| {
                        x.try_binary_op_inner(other.clone(), op, arithmetic)
                            .map(|res| (name, res))
                    })
                    .collect();
                output.map(Self::Object)
            }

            (Self::Object(this), Self::Object(mut other)) => {
                let same_keys = this.len() == other.len()
                    && this.field_names().all(|key| other.contains_field(key));
                if same_keys {
                    let output: Result<Object<_>, _> = this
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

            (Self::Prim(_) | Self::Tuple(_) | Self::Object(_), _) => {
                Err(BinaryOpError::new(op).with_side(OpSide::Rhs))
            }
            _ => Err(BinaryOpError::new(op).with_side(OpSide::Lhs)),
        }
    }

    #[inline]
    pub(crate) fn try_binary_op(
        module_id: &Arc<dyn ModuleId>,
        total_span: Location,
        lhs: Location<Self>,
        rhs: Location<Self>,
        op: BinaryOp,
        arithmetic: &dyn OrdArithmetic<T>,
    ) -> Result<Self, Error> {
        let lhs_span = lhs.with_no_extra();
        let rhs_span = rhs.with_no_extra();
        lhs.extra
            .try_binary_op_inner(rhs.extra, op, arithmetic)
            .map_err(|e| e.span(module_id.clone(), total_span, lhs_span, rhs_span))
    }
}

impl<T> Value<T> {
    pub(crate) fn try_neg(self, arithmetic: &dyn OrdArithmetic<T>) -> Result<Self, ErrorKind> {
        match self {
            Self::Prim(val) => arithmetic
                .neg(val)
                .map(Self::Prim)
                .map_err(ErrorKind::Arithmetic),

            Self::Tuple(tuple) => {
                let res: Result<Tuple<_>, _> = tuple
                    .into_iter()
                    .map(|elem| elem.try_neg(arithmetic))
                    .collect();
                res.map(Self::Tuple)
            }

            Self::Object(object) => {
                let res: Result<Object<_>, _> = object
                    .into_iter()
                    .map(|(name, value)| value.try_neg(arithmetic).map(|mapped| (name, mapped)))
                    .collect();
                res.map(Self::Object)
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
                let res: Result<Tuple<_>, _> = tuple.into_iter().map(Value::try_not).collect();
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
            (Self::Object(this), Self::Object(other)) => this.eq_by_arithmetic(other, arithmetic),
            (Self::Function(this), Self::Function(other)) => this.is_same_function(other),
            (Self::Ref(this), Self::Ref(other)) => this == other,
            _ => false,
        }
    }

    pub(crate) fn compare(
        module_id: &Arc<dyn ModuleId>,
        lhs: &Location<Self>,
        rhs: &Location<Self>,
        op: BinaryOp,
        arithmetic: &dyn OrdArithmetic<T>,
    ) -> Result<Self, Error> {
        // We only know how to compare primitive values.
        let Value::Prim(lhs_value) = &lhs.extra else {
            return Err(Error::new(module_id.clone(), lhs, ErrorKind::CannotCompare));
        };
        let Value::Prim(rhs_value) = &rhs.extra else {
            return Err(Error::new(module_id.clone(), rhs, ErrorKind::CannotCompare));
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
        module_id: &Arc<dyn ModuleId>,
        lhs: &Location<Self>,
        rhs: &Location<Self>,
    ) -> Result<Self, Error> {
        match (&lhs.extra, &rhs.extra) {
            (Value::Bool(this), Value::Bool(other)) => Ok(Value::Bool(*this && *other)),
            (Value::Bool(_), _) => {
                let err = ErrorKind::UnexpectedOperand {
                    op: BinaryOp::And.into(),
                };
                Err(Error::new(module_id.clone(), rhs, err))
            }
            _ => {
                let err = ErrorKind::UnexpectedOperand {
                    op: BinaryOp::And.into(),
                };
                Err(Error::new(module_id.clone(), lhs, err))
            }
        }
    }

    pub(crate) fn try_or(
        module_id: &Arc<dyn ModuleId>,
        lhs: &Location<Self>,
        rhs: &Location<Self>,
    ) -> Result<Self, Error> {
        match (&lhs.extra, &rhs.extra) {
            (Value::Bool(this), Value::Bool(other)) => Ok(Value::Bool(*this || *other)),
            (Value::Bool(_), _) => {
                let err = ErrorKind::UnexpectedOperand {
                    op: BinaryOp::Or.into(),
                };
                Err(Error::new(module_id.clone(), rhs, err))
            }
            _ => {
                let err = ErrorKind::UnexpectedOperand {
                    op: BinaryOp::Or.into(),
                };
                Err(Error::new(module_id.clone(), lhs, err))
            }
        }
    }
}

impl<T> Object<T> {
    fn eq_by_arithmetic(&self, other: &Self, arithmetic: &dyn OrdArithmetic<T>) -> bool {
        if self.len() == other.len() {
            for (field_name, this_elem) in self {
                let Some(that_elem) = other.get(field_name) else {
                    return false;
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
}
