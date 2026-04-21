use std::marker::PhantomData;

use cxx::UniquePtr;
use ndarray::Array2;

use crate::ffi::ffi;
use crate::variable_matrix::VariableMatrix;

/// Computes the Jacobian of a column vector of variables with respect to
/// a column vector of decision variables.
///
/// Mirrors `slp::Jacobian<double>` in C++.
pub struct Jacobian<'arena> {
    inner: UniquePtr<ffi::Jacobian>,
    arena: &'arena crate::VariableArena,
    _marker: PhantomData<&'arena ()>,
}

impl<'arena> Jacobian<'arena> {
    /// Build a new Jacobian for `variables` (a column vector) with
    /// respect to `wrt` (also a column vector).
    pub fn new(variables: &VariableMatrix<'arena>, wrt: &VariableMatrix<'arena>) -> Self {
        Self {
            inner: ffi::jacobian_new(variables.as_ref(), wrt.as_ref()),
            arena: wrt.arena(),
            _marker: PhantomData,
        }
    }

    pub fn rows(&self) -> i32 {
        ffi::jacobian_rows(self.as_ref())
    }

    pub fn cols(&self) -> i32 {
        ffi::jacobian_cols(self.as_ref())
    }

    /// Evaluate the Jacobian at `wrt`'s current values, returning a dense
    /// `Array2<f64>`.
    pub fn value(&mut self) -> Array2<f64> {
        let rows = self.rows() as usize;
        let cols = self.cols() as usize;
        let pin = self
            .inner
            .as_mut()
            .expect("Jacobian FFI pointer was unexpectedly null");
        let flat = ffi::jacobian_value(pin);
        Array2::from_shape_vec((rows, cols), flat)
            .expect("Jacobian value buffer shape mismatch")
    }

    /// Return the Jacobian as a symbolic `VariableMatrix`.
    pub fn get(&self) -> VariableMatrix<'arena> {
        VariableMatrix::from_unique_in(self.arena, ffi::jacobian_get(self.as_ref()))
    }

    #[inline]
    fn as_ref(&self) -> &ffi::Jacobian {
        self.inner
            .as_ref()
            .expect("Jacobian FFI pointer was unexpectedly null")
    }
}

impl std::fmt::Debug for Jacobian<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Jacobian")
            .field("rows", &self.rows())
            .field("cols", &self.cols())
            .finish()
    }
}
