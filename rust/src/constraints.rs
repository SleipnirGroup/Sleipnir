use std::marker::PhantomData;

use cxx::UniquePtr;

use crate::ffi::ffi;
use crate::variable::__dsl::IntoMatrixOperand;

/// A vector of equality constraints of the form cₑ(x) = 0.
pub struct EqualityConstraints<'arena> {
    pub(crate) inner: UniquePtr<ffi::EqualityConstraints>,
    _marker: PhantomData<&'arena ()>,
}

impl<'arena> EqualityConstraints<'arena> {
    #[inline]
    pub(crate) fn from_unique(inner: UniquePtr<ffi::EqualityConstraints>) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub(crate) fn as_ref(&self) -> &ffi::EqualityConstraints {
        self.inner
            .as_ref()
            .expect("EqualityConstraints FFI pointer was unexpectedly null")
    }
}

/// A vector of inequality constraints of the form cᵢ(x) ≥ 0.
pub struct InequalityConstraints<'arena> {
    pub(crate) inner: UniquePtr<ffi::InequalityConstraints>,
    _marker: PhantomData<&'arena ()>,
}

impl<'arena> InequalityConstraints<'arena> {
    #[inline]
    pub(crate) fn from_unique(inner: UniquePtr<ffi::InequalityConstraints>) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub(crate) fn as_ref(&self) -> &ffi::InequalityConstraints {
        self.inner
            .as_ref()
            .expect("InequalityConstraints FFI pointer was unexpectedly null")
    }
}

/// Sum type accepted by [`crate::Problem::subject_to`].
pub enum Constraint<'arena> {
    Equality(EqualityConstraints<'arena>),
    Inequality(InequalityConstraints<'arena>),
}

impl<'arena> From<EqualityConstraints<'arena>> for Constraint<'arena> {
    #[inline]
    fn from(value: EqualityConstraints<'arena>) -> Self {
        Self::Equality(value)
    }
}

impl<'arena> From<InequalityConstraints<'arena>> for Constraint<'arena> {
    #[inline]
    fn from(value: InequalityConstraints<'arena>) -> Self {
        Self::Inequality(value)
    }
}

/// Helper mirroring C++'s `slp::bounds(lower, x, upper)`: returns both
/// the `lower <= x` and `x <= upper` inequality constraint groups. Pass
/// each to `Problem::subject_to` separately.
///
/// The arena is extracted from `x` — which must therefore be a
/// `Variable` / `VariableMatrix` (compile error for scalar `x`, since
/// bounding a constant is meaningless).
pub fn bounds<'arena, L, X, U>(
    lower: L,
    x: X,
    upper: U,
) -> Vec<InequalityConstraints<'arena>>
where
    L: IntoMatrixOperand<'arena>,
    X: crate::__marker::HasArena<'arena> + IntoMatrixOperand<'arena> + Clone,
    U: IntoMatrixOperand<'arena>,
{
    let arena = x.arena_ref();
    let x_mat = x.clone().into_matrix(arena);
    let lower_mat = lower.into_matrix(arena);
    let upper_mat = upper.into_matrix(arena);
    vec![
        InequalityConstraints::from_unique(ffi::make_geq_matrix(
            x_mat.as_ref(),
            lower_mat.as_ref(),
        )),
        InequalityConstraints::from_unique(ffi::make_leq_matrix(
            x_mat.as_ref(),
            upper_mat.as_ref(),
        )),
    ]
}
