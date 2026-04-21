use std::marker::PhantomData;

use cxx::UniquePtr;

use crate::ffi::ffi;
use crate::variable::Variable;
use crate::variable_matrix::VariableMatrix;

/// Computes the gradient of a [`Variable`] with respect to a vector of
/// decision variables.
///
/// Mirrors `slp::Gradient<double>` in the C++ API. Lazily recomputes only
/// when the underlying expression is quadratic or higher order.
pub struct Gradient<'arena> {
    inner: UniquePtr<ffi::Gradient>,
    arena: &'arena crate::VariableArena,
    _marker: PhantomData<&'arena ()>,
}

impl<'arena> Gradient<'arena> {
    /// Builds a Gradient of `variable` with respect to `wrt`.
    ///
    /// `wrt` may be either a [`Variable`] (treated as a 1-element
    /// gradient) or a [`VariableMatrix`] column vector. Both forms map
    /// to C++'s `slp::Gradient<double>` overloads.
    ///
    /// ```ignore
    /// let g1 = Gradient::new(f, x);        // wrt single Variable
    /// let g2 = Gradient::new(f, &xs);      // wrt rows×1 VariableMatrix
    /// ```
    pub fn new<W: GradientWrt<'arena>>(variable: Variable<'arena>, wrt: W) -> Self {
        wrt.build(variable)
    }

    /// Evaluates and returns the gradient at the current values of `wrt`,
    /// as a dense `Vec<f64>` of length `wrt.rows() * wrt.cols()`.
    pub fn value(&mut self) -> Vec<f64> {
        let pin = self
            .inner
            .as_mut()
            .expect("Gradient FFI pointer was unexpectedly null");
        ffi::gradient_value(pin)
    }

    /// Returns the gradient as a symbolic `VariableMatrix` — useful for
    /// composing derivatives inside further optimization problems.
    pub fn get(&self) -> VariableMatrix<'arena> {
        let r = self
            .inner
            .as_ref()
            .expect("Gradient FFI pointer was unexpectedly null");
        VariableMatrix::from_unique_in(self.arena, ffi::gradient_get(r))
    }
}

impl std::fmt::Debug for Gradient<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gradient").finish_non_exhaustive()
    }
}

/// Sealed helper that lets `Gradient::new` accept either a single
/// `Variable` or a `VariableMatrix` column vector as the `wrt`
/// argument. Not part of the public API.
#[doc(hidden)]
pub trait GradientWrt<'arena>: crate::__sealed::Sealed {
    fn build(self, variable: Variable<'arena>) -> Gradient<'arena>;
}

impl<'arena> GradientWrt<'arena> for &VariableMatrix<'arena> {
    fn build(self, variable: Variable<'arena>) -> Gradient<'arena> {
        Gradient {
            inner: ffi::gradient_new(variable.as_ref(), self.as_ref()),
            arena: self.arena(),
            _marker: PhantomData,
        }
    }
}

impl<'arena> GradientWrt<'arena> for Variable<'arena> {
    fn build(self, variable: Variable<'arena>) -> Gradient<'arena> {
        let wrt = VariableMatrix::from_variable_in(self.arena(), self);
        Gradient {
            inner: ffi::gradient_new(variable.as_ref(), wrt.as_ref()),
            arena: self.arena(),
            _marker: PhantomData,
        }
    }
}
