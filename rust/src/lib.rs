//! Rust bindings to [Sleipnir](https://github.com/SleipnirGroup/Sleipnir), a
//! sparse nonlinear-programming solver with reverse-mode autodiff.
//!
//! # The arena model
//!
//! Every [`Variable`] the solver builds is stored inside a
//! [`VariableArena`]. The arena owns the FFI wrappers; [`Variable`]
//! itself is a cheap `Copy` handle (one `&VariableArena` + one raw
//! pointer). When the arena drops, every wrapper drops with it — no
//! leak. The borrow checker keeps the handles from outliving their
//! arena, so "dangling autodiff variable" is a compile-time error
//! rather than a runtime UAF.
//!
//! A typical program looks like:
//!
//! ```ignore
//! use hafgufa::{Problem, VariableArena, subject_to};
//!
//! let arena = VariableArena::new();           // 1. create arena
//! let mut problem = Problem::new(&arena);     // 2. bind a Problem to it
//! let x = problem.decision_variable();        // 3. create variables
//! let y = problem.decision_variable();
//! problem.minimize(x * x + y * y);            //    …build cost/constraints
//! subject_to!(problem, x + y == 1.0);
//! problem.solve(Default::default())?;         // 4. solve
//! assert!((x.value() - 0.5).abs() < 1e-6);    // 5. read results
//! # Ok::<(), hafgufa::SleipnirError>(())
//! ```
//!
//! The arena must outlive every `Variable` / `VariableMatrix` /
//! `Problem` / `OCP` that borrows it — drop it last. `Problem` takes
//! `&mut self` for its builder methods, but `Variable` borrows the
//! arena (not the problem), so you can freely hold variables across
//! `problem.subject_to(...)` calls.
//!
//! # DSL
//!
//! Rust's `PartialEq` / `PartialOrd` must return `bool`, so
//! comparison-returning constraint builders are lifted into macros.
//! See [`subject_to!`] and [`cmp!`] — both parse `==`, `<=`, `>=`,
//! `<`, `>` in their input via a tt-muncher and dispatch to the right
//! sparse/dense constraint builder.
//!
//! # Iteration callbacks
//!
//! [`Problem::add_callback`] / [`OCP::add_callback`] accept
//! `FnMut(&IterationInfo<'_>) -> bool`. [`IterationInfo`] is a
//! zero-copy view over the solver's internal Eigen storage — dense
//! slices for `x` / `s` / `y` / `z`, CSC-encoded sparse views for the
//! gradient, Hessian, and Jacobians. Nothing is copied per iteration.
//!
//! # Requirements
//!
//! A C++23 toolchain. The build probes for `<print>` up front and
//! panics with a clear error if the compiler can't satisfy it
//! (GCC 14+ libstdc++, libc++ 19+, MSVC 19.37+).
//!
//! # Features
//!
//! - `multistart` — pulls in rayon and exposes
//!   [`multistart()`](crate::multistart::multistart), a parallel
//!   "solve from N initial guesses, pick the best" driver.

pub mod arena;
pub mod constraints;
pub(crate) mod ffi;
pub mod gradient;
pub mod hessian;
pub mod jacobian;
pub mod math;
#[cfg(feature = "multistart")]
pub mod multistart;
pub mod ocp;
pub mod problem;
pub mod variable;
pub mod variable_matrix;

mod macros;

/// Sealed-supertrait module. Keeps the DSL marker traits
/// (`SleipnirScalar`, `SleipnirMatrix`, `SleipnirOperand`,
/// `SleipnirProblem`) closed to downstream crates.
#[doc(hidden)]
pub(crate) mod __sealed {
    pub trait Sealed {}
    impl<'a> Sealed for crate::Variable<'a> {}
    impl<'a> Sealed for &crate::Variable<'a> {}
    impl Sealed for f64 {}
    impl Sealed for i32 {}
    impl<'a> Sealed for crate::VariableMatrix<'a> {}
    impl<'a> Sealed for &crate::VariableMatrix<'a> {}
    impl Sealed for ndarray::Array2<f64> {}
    impl Sealed for &ndarray::Array2<f64> {}
    impl<'a> Sealed for crate::Problem<'a> {}
    impl<'a> Sealed for crate::OCP<'a> {}
}

/// Doc-hidden marker traits used by the DSL macros to emit clear,
/// operand-type-focused compile errors when a caller passes the wrong
/// type to `eq!`, `subject_to!`, etc. Not part of the public API.
///
/// The traits are sealed via the private [`__sealed::Sealed`] supertrait.
#[doc(hidden)]
pub mod __marker {
    use crate::__sealed::Sealed;
    use crate::arena::VariableArena;

    /// Scalar operand in the DSL.
    pub trait SleipnirScalar: Sealed {}
    impl<'a> SleipnirScalar for crate::Variable<'a> {}
    impl<'a> SleipnirScalar for &crate::Variable<'a> {}
    impl SleipnirScalar for f64 {}
    impl SleipnirScalar for i32 {}

    /// Matrix operand in the DSL.
    pub trait SleipnirMatrix: Sealed {}
    impl<'a> SleipnirMatrix for crate::VariableMatrix<'a> {}
    impl<'a> SleipnirMatrix for &crate::VariableMatrix<'a> {}
    impl SleipnirMatrix for ndarray::Array2<f64> {}
    impl SleipnirMatrix for &ndarray::Array2<f64> {}

    /// Any operand accepted by the DSL (scalar or matrix).
    pub trait SleipnirOperand: Sealed {
        /// Returns the operand's backing arena, if it has one. Scalars
        /// like `f64` / `i32` / `ndarray::Array2<f64>` return `None`.
        fn operand_arena<'a>(&self) -> Option<&'a VariableArena>
        where
            Self: HasArena<'a>,
        {
            Some(self.arena_ref())
        }
    }
    impl<'a> SleipnirOperand for crate::Variable<'a> {}
    impl<'a> SleipnirOperand for &crate::Variable<'a> {}
    impl SleipnirOperand for f64 {}
    impl SleipnirOperand for i32 {}
    impl<'a> SleipnirOperand for crate::VariableMatrix<'a> {}
    impl<'a> SleipnirOperand for &crate::VariableMatrix<'a> {}
    impl SleipnirOperand for ndarray::Array2<f64> {}
    impl SleipnirOperand for &ndarray::Array2<f64> {}

    /// Operand that carries its own arena reference (a `Variable` or
    /// `VariableMatrix`). Used by the DSL macros to extract the arena
    /// for constraint construction without requiring the user to pass it
    /// explicitly.
    pub trait HasArena<'a>: Sealed {
        fn arena_ref(&self) -> &'a VariableArena;
    }
    impl<'a> HasArena<'a> for crate::Variable<'a> {
        #[inline]
        fn arena_ref(&self) -> &'a VariableArena {
            self.arena()
        }
    }
    impl<'a> HasArena<'a> for &crate::Variable<'a> {
        #[inline]
        fn arena_ref(&self) -> &'a VariableArena {
            self.arena()
        }
    }
    impl<'a> HasArena<'a> for crate::VariableMatrix<'a> {
        #[inline]
        fn arena_ref(&self) -> &'a VariableArena {
            self.arena()
        }
    }
    impl<'a> HasArena<'a> for &crate::VariableMatrix<'a> {
        #[inline]
        fn arena_ref(&self) -> &'a VariableArena {
            self.arena()
        }
    }

    /// Problem container accepted by `subject_to!`.
    pub trait SleipnirProblem: Sealed {}
    impl<'a> SleipnirProblem for crate::Problem<'a> {}
    impl<'a> SleipnirProblem for crate::OCP<'a> {}

    /// Static-assertion helpers. They don't inspect the value — just
    /// force trait resolution so a bad operand fails with a
    /// `SleipnirOperand` / `SleipnirProblem` error.
    #[inline(always)]
    pub fn assert_operand<T: SleipnirOperand>(_: &T) {}

    #[inline(always)]
    pub fn assert_problem<P: SleipnirProblem>(_: &mut P) {}

    /// Extracts the arena from an operand that has one. Used by the
    /// standalone DSL macros (`eq!`, `ge!`, …) where the arena isn't
    /// supplied explicitly.
    #[inline(always)]
    pub fn arena_from<'a, T: HasArena<'a>>(value: &T) -> &'a VariableArena {
        value.arena_ref()
    }
}

pub use arena::VariableArena;
pub use constraints::{Constraint, EqualityConstraints, InequalityConstraints, bounds};
pub use gradient::Gradient;
pub use hessian::{Hessian, HessianTriangle};
pub use jacobian::Jacobian;
#[cfg(feature = "multistart")]
pub use multistart::{MultistartResult, multistart};
pub use ocp::{DynamicsType, OCP, TimestepMethod, TranscriptionMethod};
pub use problem::{Options, Problem};
pub use variable::{__dsl::IntoMatrixOperand, IntoVariable, Variable};
pub use variable_matrix::{VariableMatrix, hstack, solve, vstack};

/// One-shot glob import for building optimization problems.
pub mod prelude {
    pub use crate::{
        Constraint, DynamicsType, EqualityConstraints, ExitStatus, Gradient, Hessian,
        HessianTriangle, InequalityConstraints, IntoMatrixOperand, IntoVariable, IterationInfo,
        Jacobian, OCP, Options, Problem, SleipnirError, SparseMatrixView, SparseVectorView,
        TimestepMethod, TranscriptionMethod, Variable, VariableArena, VariableMatrix, bounds, cmp,
        hstack, math, solve, subject_to, vstack,
    };

    #[cfg(feature = "multistart")]
    pub use crate::{MultistartResult, multistart};
}

/// Expression type, matching C++'s `slp::ExpressionType`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpressionType {
    None,
    Constant,
    Linear,
    Quadratic,
    Nonlinear,
}

impl ExpressionType {
    pub(crate) fn from_raw(raw: u8) -> Self {
        match raw {
            0 => Self::None,
            1 => Self::Constant,
            2 => Self::Linear,
            3 => Self::Quadratic,
            4 => Self::Nonlinear,
            other => panic!("unknown ExpressionType discriminant: {other}"),
        }
    }
}

/// Solver exit status, matching C++'s `slp::ExitStatus`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExitStatus {
    Success,
    CallbackRequestedStop,
    TooFewDofs,
    LocallyInfeasible,
    GloballyInfeasible,
    FactorizationFailed,
    LineSearchFailed,
    FeasibilityRestorationFailed,
    NonfiniteInitialGuess,
    DivergingIterates,
    MaxIterationsExceeded,
    Timeout,
}

impl ExitStatus {
    pub(crate) fn from_raw(raw: i8) -> Self {
        match raw {
            0 => Self::Success,
            1 => Self::CallbackRequestedStop,
            -1 => Self::TooFewDofs,
            -2 => Self::LocallyInfeasible,
            -3 => Self::GloballyInfeasible,
            -4 => Self::FactorizationFailed,
            -5 => Self::LineSearchFailed,
            -6 => Self::FeasibilityRestorationFailed,
            -7 => Self::NonfiniteInitialGuess,
            -8 => Self::DivergingIterates,
            -9 => Self::MaxIterationsExceeded,
            -10 => Self::Timeout,
            other => panic!("unknown ExitStatus discriminant: {other}"),
        }
    }

    /// Converts this status into a `Result`. `Success` → `Ok(())`; every
    /// other variant → `Err(SleipnirError::…)`.
    pub fn into_result(self) -> Result<(), SleipnirError> {
        match self {
            Self::Success => Ok(()),
            Self::CallbackRequestedStop => Err(SleipnirError::CallbackRequestedStop),
            Self::TooFewDofs => Err(SleipnirError::TooFewDofs),
            Self::LocallyInfeasible => Err(SleipnirError::LocallyInfeasible),
            Self::GloballyInfeasible => Err(SleipnirError::GloballyInfeasible),
            Self::FactorizationFailed => Err(SleipnirError::FactorizationFailed),
            Self::LineSearchFailed => Err(SleipnirError::LineSearchFailed),
            Self::FeasibilityRestorationFailed => Err(SleipnirError::FeasibilityRestorationFailed),
            Self::NonfiniteInitialGuess => Err(SleipnirError::NonfiniteInitialGuess),
            Self::DivergingIterates => Err(SleipnirError::DivergingIterates),
            Self::MaxIterationsExceeded => Err(SleipnirError::MaxIterationsExceeded),
            Self::Timeout => Err(SleipnirError::Timeout),
        }
    }
}

/// Error variants for every non-success `ExitStatus` the solver can produce.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum SleipnirError {
    #[error(
        "the solver returned its best-so-far solution after an iteration callback requested a stop"
    )]
    CallbackRequestedStop,
    #[error("the solver determined the problem to be overconstrained (too few degrees of freedom)")]
    TooFewDofs,
    #[error("the solver determined the problem to be locally infeasible")]
    LocallyInfeasible,
    #[error("the problem frontend determined the feasible region is empty (globally infeasible)")]
    GloballyInfeasible,
    #[error("the linear system factorization failed")]
    FactorizationFailed,
    #[error("the backtracking line search failed and the problem isn't locally infeasible")]
    LineSearchFailed,
    #[error(
        "the solver failed to reach the desired tolerance and feasibility restoration failed to converge"
    )]
    FeasibilityRestorationFailed,
    #[error("the solver encountered nonfinite initial cost, constraints, or derivatives")]
    NonfiniteInitialGuess,
    #[error("the solver encountered diverging primal iterates")]
    DivergingIterates,
    #[error("the solver exceeded the configured maximum number of iterations")]
    MaxIterationsExceeded,
    #[error("the solver exceeded the configured wall-clock timeout")]
    Timeout,
}

/// Per-iteration solver state passed to callbacks registered via
/// [`Problem::add_callback`].
///
/// A zero-copy view over the solver's internal Eigen storage. All
/// accessor methods return slices that borrow directly from the
/// underlying vectors/matrices — no allocation or copy per iteration.
/// The handle is only valid during the callback invocation; don't
/// store it beyond the callback body.
pub struct IterationInfo<'a> {
    pub(crate) inner: &'a crate::ffi::ffi::IterationInfo,
}

impl<'a> IterationInfo<'a> {
    /// Solver iteration counter (0-based).
    #[inline]
    pub fn iteration(&self) -> i32 {
        crate::ffi::ffi::iteration_info_iteration(self.inner)
    }

    /// Decision variables `x` at the current iterate.
    #[inline]
    pub fn x(&self) -> &'a [f64] {
        crate::ffi::ffi::iteration_info_x(self.inner)
    }

    /// Inequality-constraint slack variables `s`.
    #[inline]
    pub fn s(&self) -> &'a [f64] {
        crate::ffi::ffi::iteration_info_s(self.inner)
    }

    /// Equality-constraint dual variables `y`.
    #[inline]
    pub fn y(&self) -> &'a [f64] {
        crate::ffi::ffi::iteration_info_y(self.inner)
    }

    /// Inequality-constraint dual variables `z`.
    #[inline]
    pub fn z(&self) -> &'a [f64] {
        crate::ffi::ffi::iteration_info_z(self.inner)
    }

    /// Sparse cost-function gradient `g = ∇f(x)`.
    #[inline]
    pub fn gradient(&self) -> SparseVectorView<'a> {
        SparseVectorView {
            size: crate::ffi::ffi::iteration_info_g_size(self.inner),
            indices: crate::ffi::ffi::iteration_info_g_indices(self.inner),
            values: crate::ffi::ffi::iteration_info_g_values(self.inner),
        }
    }

    /// Sparse Lagrangian Hessian `H = ∇²ₓₓL`.
    #[inline]
    pub fn hessian(&self) -> SparseMatrixView<'a> {
        SparseMatrixView {
            rows: crate::ffi::ffi::iteration_info_hessian_rows(self.inner),
            cols: crate::ffi::ffi::iteration_info_hessian_cols(self.inner),
            outer_indices: crate::ffi::ffi::iteration_info_hessian_outer(self.inner),
            inner_indices: crate::ffi::ffi::iteration_info_hessian_inner(self.inner),
            values: crate::ffi::ffi::iteration_info_hessian_values(self.inner),
        }
    }

    /// Sparse equality-constraint Jacobian `A_e = ∂cₑ/∂x`.
    #[inline]
    pub fn equality_jacobian(&self) -> SparseMatrixView<'a> {
        SparseMatrixView {
            rows: crate::ffi::ffi::iteration_info_eq_jacobian_rows(self.inner),
            cols: crate::ffi::ffi::iteration_info_eq_jacobian_cols(self.inner),
            outer_indices: crate::ffi::ffi::iteration_info_eq_jacobian_outer(self.inner),
            inner_indices: crate::ffi::ffi::iteration_info_eq_jacobian_inner(self.inner),
            values: crate::ffi::ffi::iteration_info_eq_jacobian_values(self.inner),
        }
    }

    /// Sparse inequality-constraint Jacobian `A_i = ∂cᵢ/∂x`.
    #[inline]
    pub fn inequality_jacobian(&self) -> SparseMatrixView<'a> {
        SparseMatrixView {
            rows: crate::ffi::ffi::iteration_info_ineq_jacobian_rows(self.inner),
            cols: crate::ffi::ffi::iteration_info_ineq_jacobian_cols(self.inner),
            outer_indices: crate::ffi::ffi::iteration_info_ineq_jacobian_outer(self.inner),
            inner_indices: crate::ffi::ffi::iteration_info_ineq_jacobian_inner(self.inner),
            values: crate::ffi::ffi::iteration_info_ineq_jacobian_values(self.inner),
        }
    }
}

/// Zero-copy view over an `Eigen::SparseVector<double>`. Encoded as a
/// length plus parallel `(indices, values)` arrays of the non-zero
/// entries.
#[derive(Copy, Clone)]
pub struct SparseVectorView<'a> {
    /// Logical size of the vector (non-zeros may be fewer).
    pub size: i32,
    /// Indices of non-zero entries.
    pub indices: &'a [i32],
    /// Values at those indices, parallel to [`Self::indices`].
    pub values: &'a [f64],
}

impl<'a> SparseVectorView<'a> {
    /// Number of non-zero entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Iterate `(index, value)` pairs of the non-zero entries.
    pub fn iter(&self) -> impl Iterator<Item = (i32, f64)> + '_ {
        self.indices
            .iter()
            .copied()
            .zip(self.values.iter().copied())
    }

    /// Materialize into a dense `Vec<f64>` of length `size`.
    pub fn to_dense(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.size as usize];
        for (i, v) in self.iter() {
            out[i as usize] = v;
        }
        out
    }
}

/// Zero-copy view over an `Eigen::SparseMatrix<double>` in column-major
/// CSC layout. `outer_indices.len() == cols + 1`; `inner_indices` and
/// `values` have length `nnz()`. Column `j`'s non-zero entries live at
/// indices `outer_indices[j] .. outer_indices[j + 1]`.
#[derive(Copy, Clone)]
pub struct SparseMatrixView<'a> {
    pub rows: i32,
    pub cols: i32,
    pub outer_indices: &'a [i32],
    pub inner_indices: &'a [i32],
    pub values: &'a [f64],
}

impl<'a> SparseMatrixView<'a> {
    /// Number of non-zero entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Iterate `(row, col, value)` triplets of the non-zero entries in
    /// column-major order.
    pub fn iter_triplets(&self) -> impl Iterator<Item = (i32, i32, f64)> + '_ {
        (0..self.cols).flat_map(move |c| {
            let start = self.outer_indices[c as usize] as usize;
            let end = self.outer_indices[c as usize + 1] as usize;
            (start..end).map(move |k| (self.inner_indices[k], c, self.values[k]))
        })
    }

    /// Materialize into a dense `ndarray::Array2<f64>`.
    pub fn to_dense(&self) -> ndarray::Array2<f64> {
        let mut m = ndarray::Array2::zeros((self.rows as usize, self.cols as usize));
        for (r, c, v) in self.iter_triplets() {
            m[[r as usize, c as usize]] = v;
        }
        m
    }
}
