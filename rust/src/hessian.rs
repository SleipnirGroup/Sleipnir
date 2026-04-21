use std::marker::PhantomData;

use cxx::UniquePtr;
use ndarray::Array2;

use crate::ffi::ffi;
use crate::variable::Variable;
use crate::variable_matrix::VariableMatrix;

/// Which triangle of the Hessian to materialize. Mirrors Eigen's `UpLo`
/// template parameter on `slp::Hessian`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HessianTriangle {
    /// Lower triangle only.
    Lower,
    /// Full symmetric matrix (lower | upper).
    Full,
}

impl HessianTriangle {
    #[inline]
    fn tag(self) -> i32 {
        match self {
            Self::Lower => 0,
            Self::Full => 1,
        }
    }
}

/// Computes the Hessian of a scalar [`Variable`] with respect to a column
/// vector of decision variables.
///
/// Mirrors `slp::Hessian<double, UpLo>` in C++. Linear rows are cached once
/// on construction; only nonlinear rows get recomputed on each
/// [`Hessian::value`] call.
pub struct Hessian<'arena> {
    inner: UniquePtr<ffi::Hessian>,
    arena: &'arena crate::VariableArena,
    _marker: PhantomData<&'arena ()>,
}

impl<'arena> Hessian<'arena> {
    /// Build a new Hessian for `variable` with respect to the column
    /// vector `wrt`.
    pub fn new(
        variable: Variable<'arena>,
        wrt: &VariableMatrix<'arena>,
        tri: HessianTriangle,
    ) -> Self {
        Self {
            inner: ffi::hessian_new(variable.as_ref(), wrt.as_ref(), tri.tag()),
            arena: wrt.arena(),
            _marker: PhantomData,
        }
    }

    pub fn rows(&self) -> i32 {
        ffi::hessian_rows(self.as_ref())
    }

    pub fn cols(&self) -> i32 {
        ffi::hessian_cols(self.as_ref())
    }

    /// Evaluates the Hessian at the current values of `wrt`, returning a
    /// dense `Array2<f64>`.
    pub fn value(&mut self) -> Array2<f64> {
        let rows = self.rows() as usize;
        let cols = self.cols() as usize;
        let pin = self
            .inner
            .as_mut()
            .expect("Hessian FFI pointer was unexpectedly null");
        let flat = ffi::hessian_value(pin);
        Array2::from_shape_vec((rows, cols), flat).expect("Hessian value buffer shape mismatch")
    }

    /// Returns the Hessian as a symbolic `VariableMatrix`.
    pub fn get(&self) -> VariableMatrix<'arena> {
        VariableMatrix::from_unique_in(self.arena, ffi::hessian_get(self.as_ref()))
    }

    #[inline]
    fn as_ref(&self) -> &ffi::Hessian {
        self.inner
            .as_ref()
            .expect("Hessian FFI pointer was unexpectedly null")
    }
}

impl std::fmt::Debug for Hessian<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hessian")
            .field("rows", &self.rows())
            .field("cols", &self.cols())
            .finish()
    }
}
