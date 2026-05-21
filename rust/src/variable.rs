use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::ExpressionType;
use crate::arena::VariableArena;
use crate::constraints::{EqualityConstraints, InequalityConstraints};
use crate::ffi::ffi;

/// An autodiff variable backed by an [`VariableArena`].
///
/// `Variable<'arena>` is `Copy`: the struct is a 16-byte (arena reference +
/// FFI pointer) handle into the arena's owned storage. Arithmetic operators
/// take `self` by value without consuming anything, so
/// `x * x + y * y` compiles with no `&` or `.clone()` chatter.
///
/// The `'arena` lifetime forces every handle to outlive at most its
/// backing [`VariableArena`] — dangling handles are a compile error, never
/// a runtime UAF.
///
/// Not `Send` / `Sync`: Sleipnir's expression graph is single-thread only.
#[derive(Copy, Clone)]
pub struct Variable<'arena> {
    arena: &'arena VariableArena,
    ptr: *const ffi::Variable,
    _marker: PhantomData<&'arena ()>,
}

impl<'arena> Variable<'arena> {
    /// Construct a constant variable inside the given arena.
    #[inline]
    pub fn constant_in(arena: &'arena VariableArena, value: f64) -> Self {
        let ptr = arena.store(ffi::variable_from_f64(value));
        Self {
            arena,
            ptr,
            _marker: PhantomData,
        }
    }

    /// Evaluate the variable's current numeric value.
    #[inline]
    pub fn value(&self) -> f64 {
        ffi::variable_value(self.as_ref())
    }

    /// Overwrite the variable's underlying value in place. The expression
    /// node identity is preserved — any expression built from this
    /// variable still references the same slot and sees the update.
    #[inline]
    pub fn set_value(&self, value: f64) {
        ffi::variable_set_value(self.as_ref(), value);
    }

    /// Returns the expression type (constant, linear, quadratic, or nonlinear).
    #[inline]
    pub fn expression_type(&self) -> ExpressionType {
        ExpressionType::from_raw(ffi::variable_type(self.as_ref()))
    }

    /// Arena this variable lives in. Useful when building new
    /// variables next to an existing one and you don't have the arena
    /// reference at hand.
    #[inline]
    pub fn arena(&self) -> &'arena VariableArena {
        self.arena
    }

    /// Wrap a freshly-minted FFI handle and store it in the given arena.
    #[inline]
    pub(crate) fn from_unique_in(
        arena: &'arena VariableArena,
        unique: cxx::UniquePtr<ffi::Variable>,
    ) -> Self {
        let ptr = arena.store(unique);
        Self {
            arena,
            ptr,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub(crate) fn as_ref(&self) -> &ffi::Variable {
        // SAFETY: `ptr` was produced by the arena from a valid UniquePtr
        // and stays alive for `'arena`.
        unsafe { &*self.ptr }
    }
}

impl fmt::Debug for Variable<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Variable")
            .field("value", &self.value())
            .field("type", &self.expression_type())
            .finish()
    }
}

/// Generic conversion to a [`Variable`] used in DSL and operator positions.
///
/// Implemented for `Variable<'arena>` (no-op since it's `Copy`),
/// `&Variable<'arena>` (dereferenced copy), `f64` and `i32` (materialised
/// as constants in the supplied arena).
pub trait IntoVariable<'arena> {
    fn into_variable(self, arena: &'arena VariableArena) -> Variable<'arena>;
}

impl<'arena> IntoVariable<'arena> for Variable<'arena> {
    #[inline]
    fn into_variable(self, _arena: &'arena VariableArena) -> Variable<'arena> {
        self
    }
}

impl<'arena> IntoVariable<'arena> for &Variable<'arena> {
    #[inline]
    fn into_variable(self, _arena: &'arena VariableArena) -> Variable<'arena> {
        *self
    }
}

impl<'arena> IntoVariable<'arena> for f64 {
    #[inline]
    fn into_variable(self, arena: &'arena VariableArena) -> Variable<'arena> {
        Variable::constant_in(arena, self)
    }
}

impl<'arena> IntoVariable<'arena> for i32 {
    #[inline]
    fn into_variable(self, arena: &'arena VariableArena) -> Variable<'arena> {
        Variable::constant_in(arena, self as f64)
    }
}

// ---- Operators ----
//
// `Variable` is `Copy`, so `self` parameters don't consume the operand.
// Binary operators look up their arena from `self`; the RHS is materialized
// into the same arena via `IntoVariable`.

macro_rules! impl_variable_binop {
    ($trait:ident, $method:ident, $ffi:ident) => {
        impl<'a, R: IntoVariable<'a>> $trait<R> for Variable<'a> {
            type Output = Variable<'a>;
            #[inline]
            fn $method(self, rhs: R) -> Variable<'a> {
                let rhs = rhs.into_variable(self.arena);
                Variable::from_unique_in(self.arena, ffi::$ffi(self.as_ref(), rhs.as_ref()))
            }
        }
        impl<'a, R: IntoVariable<'a>> $trait<R> for &Variable<'a> {
            type Output = Variable<'a>;
            #[inline]
            fn $method(self, rhs: R) -> Variable<'a> {
                let rhs = rhs.into_variable(self.arena);
                Variable::from_unique_in(self.arena, ffi::$ffi(self.as_ref(), rhs.as_ref()))
            }
        }
    };
}

impl_variable_binop!(Add, add, variable_add);
impl_variable_binop!(Sub, sub, variable_sub);
impl_variable_binop!(Mul, mul, variable_mul);
impl_variable_binop!(Div, div, variable_div);

// Scalar-on-the-left impls. Orphan rule forces concrete `f64` here.
macro_rules! impl_scalar_lhs_binop {
    ($trait:ident, $method:ident, $ffi:ident) => {
        impl<'a> $trait<Variable<'a>> for f64 {
            type Output = Variable<'a>;
            #[inline]
            fn $method(self, rhs: Variable<'a>) -> Variable<'a> {
                let lhs = Variable::constant_in(rhs.arena, self);
                Variable::from_unique_in(rhs.arena, ffi::$ffi(lhs.as_ref(), rhs.as_ref()))
            }
        }
        impl<'a> $trait<&Variable<'a>> for f64 {
            type Output = Variable<'a>;
            #[inline]
            fn $method(self, rhs: &Variable<'a>) -> Variable<'a> {
                let lhs = Variable::constant_in(rhs.arena, self);
                Variable::from_unique_in(rhs.arena, ffi::$ffi(lhs.as_ref(), rhs.as_ref()))
            }
        }
    };
}

impl_scalar_lhs_binop!(Add, add, variable_add);
impl_scalar_lhs_binop!(Sub, sub, variable_sub);
impl_scalar_lhs_binop!(Mul, mul, variable_mul);
impl_scalar_lhs_binop!(Div, div, variable_div);

impl<'a> Neg for Variable<'a> {
    type Output = Variable<'a>;
    #[inline]
    fn neg(self) -> Variable<'a> {
        Variable::from_unique_in(self.arena, ffi::variable_neg(self.as_ref()))
    }
}

impl<'a> Neg for &Variable<'a> {
    type Output = Variable<'a>;
    #[inline]
    fn neg(self) -> Variable<'a> {
        Variable::from_unique_in(self.arena, ffi::variable_neg(self.as_ref()))
    }
}

/// Doc-hidden constraint builders used by the DSL macros.
#[doc(hidden)]
pub mod __dsl {
    use super::*;
    use crate::variable_matrix::VariableMatrix;

    #[inline]
    pub fn eq<'a, L, R>(arena: &'a VariableArena, lhs: L, rhs: R) -> EqualityConstraints<'a>
    where
        L: IntoVariable<'a>,
        R: IntoVariable<'a>,
    {
        let lhs = lhs.into_variable(arena);
        let rhs = rhs.into_variable(arena);
        EqualityConstraints::from_unique(ffi::make_equality(lhs.as_ref(), rhs.as_ref()))
    }

    #[inline]
    pub fn ge<'a, L, R>(arena: &'a VariableArena, lhs: L, rhs: R) -> InequalityConstraints<'a>
    where
        L: IntoVariable<'a>,
        R: IntoVariable<'a>,
    {
        let lhs = lhs.into_variable(arena);
        let rhs = rhs.into_variable(arena);
        InequalityConstraints::from_unique(ffi::make_geq(lhs.as_ref(), rhs.as_ref()))
    }

    #[inline]
    pub fn le<'a, L, R>(arena: &'a VariableArena, lhs: L, rhs: R) -> InequalityConstraints<'a>
    where
        L: IntoVariable<'a>,
        R: IntoVariable<'a>,
    {
        let lhs = lhs.into_variable(arena);
        let rhs = rhs.into_variable(arena);
        InequalityConstraints::from_unique(ffi::make_leq(lhs.as_ref(), rhs.as_ref()))
    }

    #[inline]
    pub fn gt<'a, L, R>(arena: &'a VariableArena, lhs: L, rhs: R) -> InequalityConstraints<'a>
    where
        L: IntoVariable<'a>,
        R: IntoVariable<'a>,
    {
        ge(arena, lhs, rhs)
    }

    #[inline]
    pub fn lt<'a, L, R>(arena: &'a VariableArena, lhs: L, rhs: R) -> InequalityConstraints<'a>
    where
        L: IntoVariable<'a>,
        R: IntoVariable<'a>,
    {
        le(arena, lhs, rhs)
    }

    #[inline]
    pub fn eq_mat<'a, L, R>(arena: &'a VariableArena, lhs: L, rhs: R) -> EqualityConstraints<'a>
    where
        L: IntoMatrixOperand<'a>,
        R: IntoMatrixOperand<'a>,
    {
        let lhs = lhs.into_matrix(arena);
        let rhs = rhs.into_matrix(arena);
        EqualityConstraints::from_unique(ffi::make_equality_matrix(lhs.as_ref(), rhs.as_ref()))
    }

    #[inline]
    pub fn ge_mat<'a, L, R>(arena: &'a VariableArena, lhs: L, rhs: R) -> InequalityConstraints<'a>
    where
        L: IntoMatrixOperand<'a>,
        R: IntoMatrixOperand<'a>,
    {
        let lhs = lhs.into_matrix(arena);
        let rhs = rhs.into_matrix(arena);
        InequalityConstraints::from_unique(ffi::make_geq_matrix(lhs.as_ref(), rhs.as_ref()))
    }

    #[inline]
    pub fn le_mat<'a, L, R>(arena: &'a VariableArena, lhs: L, rhs: R) -> InequalityConstraints<'a>
    where
        L: IntoMatrixOperand<'a>,
        R: IntoMatrixOperand<'a>,
    {
        let lhs = lhs.into_matrix(arena);
        let rhs = rhs.into_matrix(arena);
        InequalityConstraints::from_unique(ffi::make_leq_matrix(lhs.as_ref(), rhs.as_ref()))
    }

    /// Conversion trait accepted by matrix DSL functions and operators.
    pub trait IntoMatrixOperand<'arena> {
        fn into_matrix(self, arena: &'arena VariableArena) -> VariableMatrix<'arena>;
    }

    impl<'arena> IntoMatrixOperand<'arena> for VariableMatrix<'arena> {
        fn into_matrix(self, _arena: &'arena VariableArena) -> VariableMatrix<'arena> {
            self
        }
    }
    impl<'arena> IntoMatrixOperand<'arena> for &VariableMatrix<'arena> {
        fn into_matrix(self, _arena: &'arena VariableArena) -> VariableMatrix<'arena> {
            self.clone()
        }
    }
    impl<'arena> IntoMatrixOperand<'arena> for Variable<'arena> {
        fn into_matrix(self, arena: &'arena VariableArena) -> VariableMatrix<'arena> {
            VariableMatrix::from_variable_in(arena, self)
        }
    }
    impl<'arena> IntoMatrixOperand<'arena> for &Variable<'arena> {
        fn into_matrix(self, arena: &'arena VariableArena) -> VariableMatrix<'arena> {
            VariableMatrix::from_variable_in(arena, *self)
        }
    }
    impl<'arena> IntoMatrixOperand<'arena> for f64 {
        fn into_matrix(self, arena: &'arena VariableArena) -> VariableMatrix<'arena> {
            VariableMatrix::from_variable_in(arena, Variable::constant_in(arena, self))
        }
    }
    impl<'arena> IntoMatrixOperand<'arena> for &ndarray::Array2<f64> {
        fn into_matrix(self, arena: &'arena VariableArena) -> VariableMatrix<'arena> {
            VariableMatrix::from_array_in(arena, self)
        }
    }
    impl<'arena> IntoMatrixOperand<'arena> for ndarray::Array2<f64> {
        fn into_matrix(self, arena: &'arena VariableArena) -> VariableMatrix<'arena> {
            VariableMatrix::from_array_in(arena, &self)
        }
    }
}
