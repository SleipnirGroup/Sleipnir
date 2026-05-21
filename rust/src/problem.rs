use std::marker::PhantomData;
use std::pin::Pin;
use std::time::Duration;

use cxx::UniquePtr;

use crate::arena::VariableArena;
use crate::constraints::{Constraint, EqualityConstraints, InequalityConstraints};
use crate::ffi::{RustCallback, SolverOptions, ffi};
use crate::variable::Variable;
use crate::variable_matrix::VariableMatrix;
use crate::{ExitStatus, ExpressionType, IterationInfo, SleipnirError};

/// Solver options, mirroring `slp::Options` in C++.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub tolerance: f64,
    pub max_iterations: i32,
    pub timeout: Option<Duration>,
    pub feasible_ipm: bool,
    pub diagnostics: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_iterations: 5000,
            timeout: None,
            feasible_ipm: false,
            diagnostics: false,
        }
    }
}

impl Options {
    pub(crate) fn into_ffi(self) -> SolverOptions {
        SolverOptions {
            tolerance: self.tolerance,
            max_iterations: self.max_iterations,
            timeout_seconds: self
                .timeout
                .map(|d| d.as_secs_f64())
                .unwrap_or(f64::INFINITY),
            feasible_ipm: self.feasible_ipm,
            diagnostics: self.diagnostics,
        }
    }

    /// Convergence tolerance on the KKT residual. Default `1e-8`.
    #[inline]
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Maximum number of solver iterations. Default `5000`.
    #[inline]
    pub fn max_iterations(mut self, max_iterations: i32) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Wall-clock timeout. Default `None` (no limit).
    #[inline]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Clears the wall-clock timeout.
    #[inline]
    pub fn no_timeout(mut self) -> Self {
        self.timeout = None;
        self
    }

    /// Enables the feasible interior-point method — only step sizes
    /// that keep inequality slacks positive are taken. Useful when the
    /// problem is ill-conditioned in infeasible regions. Default
    /// `false`.
    #[inline]
    pub fn feasible_ipm(mut self, feasible_ipm: bool) -> Self {
        self.feasible_ipm = feasible_ipm;
        self
    }

    /// Enable Sleipnir's per-iteration diagnostic printing. Default
    /// `false`.
    #[inline]
    pub fn diagnostics(mut self, diagnostics: bool) -> Self {
        self.diagnostics = diagnostics;
        self
    }
}

/// A constrained nonlinear optimization problem in natural mathematical
/// notation.
///
/// Mirrors `slp::Problem<double>` in the C++ API. Bound to a
/// [`VariableArena`] via its `'arena` lifetime — every decision variable
/// and intermediate expression the problem produces lives in that arena
/// until the arena drops.
pub struct Problem<'arena> {
    inner: UniquePtr<ffi::Problem>,
    arena: &'arena VariableArena,
    _marker: PhantomData<&'arena ()>,
}

impl<'arena> Problem<'arena> {
    /// Constructs an empty optimization problem anchored to `arena`.
    pub fn new(arena: &'arena VariableArena) -> Self {
        Self {
            inner: ffi::problem_new(),
            arena,
            _marker: PhantomData,
        }
    }

    /// The arena backing this problem.
    #[inline]
    pub fn arena(&self) -> &'arena VariableArena {
        self.arena
    }

    fn pin_inner(&mut self) -> Pin<&mut ffi::Problem> {
        self.inner
            .as_mut()
            .expect("Problem FFI pointer was unexpectedly null")
    }

    fn ref_inner(&self) -> &ffi::Problem {
        self.inner
            .as_ref()
            .expect("Problem FFI pointer was unexpectedly null")
    }

    /// Creates a scalar decision variable, initial value zero.
    pub fn decision_variable(&mut self) -> Variable<'arena> {
        let unique = ffi::problem_decision_variable(self.pin_inner());
        Variable::from_unique_in(self.arena, unique)
    }

    /// Creates `n` scalar decision variables as a `Vec`. Equivalent to
    /// `(0..n).map(|_| self.decision_variable()).collect()` but avoids
    /// the boilerplate at call sites.
    pub fn decision_variables(&mut self, n: usize) -> Vec<Variable<'arena>> {
        (0..n).map(|_| self.decision_variable()).collect()
    }

    /// Creates a `rows × cols` matrix of fresh decision variables.
    pub fn decision_variable_matrix(&mut self, rows: i32, cols: i32) -> VariableMatrix<'arena> {
        let unique = ffi::problem_decision_variable_matrix(self.pin_inner(), rows, cols);
        VariableMatrix::from_unique_in(self.arena, unique)
    }

    /// Creates a symmetric `rows × rows` matrix of decision variables.
    /// Variables are reused across the diagonal to reduce problem size.
    pub fn symmetric_decision_variable(&mut self, rows: i32) -> VariableMatrix<'arena> {
        let unique = ffi::problem_symmetric_decision_variable(self.pin_inner(), rows);
        VariableMatrix::from_unique_in(self.arena, unique)
    }

    /// Tells the solver to minimize the given cost expression.
    pub fn minimize(&mut self, cost: Variable<'arena>) {
        ffi::problem_minimize(self.pin_inner(), cost.as_ref());
    }

    /// Tells the solver to minimize a 1×1 [`VariableMatrix`] cost.
    pub fn minimize_matrix(&mut self, cost: &VariableMatrix<'arena>) {
        ffi::problem_minimize_matrix(self.pin_inner(), cost.as_ref());
    }

    /// Tells the solver to maximize the given objective expression.
    pub fn maximize(&mut self, objective: Variable<'arena>) {
        ffi::problem_maximize(self.pin_inner(), objective.as_ref());
    }

    /// Tells the solver to maximize a 1×1 [`VariableMatrix`] objective.
    pub fn maximize_matrix(&mut self, objective: &VariableMatrix<'arena>) {
        ffi::problem_maximize_matrix(self.pin_inner(), objective.as_ref());
    }

    /// Adds an equality or inequality constraint to the problem.
    ///
    /// Prefer the [`crate::subject_to`] macro for the `lhs OP rhs` form.
    pub fn subject_to(&mut self, constraint: impl Into<Constraint<'arena>>) {
        match constraint.into() {
            Constraint::Equality(c) => self.subject_to_equality(c),
            Constraint::Inequality(c) => self.subject_to_inequality(c),
        }
    }

    pub fn subject_to_equality(&mut self, constraint: EqualityConstraints<'arena>) {
        ffi::problem_subject_to_eq(self.pin_inner(), constraint.as_ref());
    }

    pub fn subject_to_inequality(&mut self, constraint: InequalityConstraints<'arena>) {
        ffi::problem_subject_to_ineq(self.pin_inner(), constraint.as_ref());
    }

    /// Adds `lower <= x` and `x <= upper` as two inequality constraints.
    /// Shorthand for calling `bounds(lower, x, upper)` followed by two
    /// `subject_to` invocations.
    pub fn bound<L, X, U>(&mut self, lower: L, x: X, upper: U)
    where
        L: crate::variable::__dsl::IntoMatrixOperand<'arena>,
        X: crate::__marker::HasArena<'arena>
            + crate::variable::__dsl::IntoMatrixOperand<'arena>
            + Clone,
        U: crate::variable::__dsl::IntoMatrixOperand<'arena>,
    {
        for c in crate::bounds(lower, x, upper) {
            self.subject_to_inequality(c);
        }
    }

    /// Returns the cost function's expression type.
    pub fn cost_function_type(&self) -> ExpressionType {
        ExpressionType::from_raw(ffi::problem_cost_function_type(self.ref_inner()))
    }

    /// Returns the highest-order equality constraint's expression type.
    pub fn equality_constraint_type(&self) -> ExpressionType {
        ExpressionType::from_raw(ffi::problem_equality_constraint_type(self.ref_inner()))
    }

    /// Returns the highest-order inequality constraint's expression type.
    pub fn inequality_constraint_type(&self) -> ExpressionType {
        ExpressionType::from_raw(ffi::problem_inequality_constraint_type(self.ref_inner()))
    }

    /// Solves the problem. The solution is written back into the decision
    /// variables handed out by [`Problem::decision_variable`].
    pub fn solve(&mut self, options: Options) -> Result<(), SleipnirError> {
        self.solve_status(options).into_result()
    }

    /// Raw variant of [`Problem::solve`] returning the full [`ExitStatus`].
    pub fn solve_status(&mut self, options: Options) -> ExitStatus {
        Self::solve_via_pin(self.pin_inner(), options)
    }

    pub(crate) fn solve_via_pin(
        pin: std::pin::Pin<&mut ffi::Problem>,
        options: Options,
    ) -> ExitStatus {
        ExitStatus::from_raw(ffi::problem_solve(pin, options.into_ffi()))
    }

    /// Registers a callback invoked at the start of each solver iteration.
    /// Return `true` to ask the solver to stop, `false` to continue.
    pub fn add_callback<F>(&mut self, callback: F)
    where
        F: for<'a> FnMut(&IterationInfo<'a>) -> bool + Send + 'static,
    {
        let boxed = Box::new(RustCallback {
            inner: Box::new(callback),
        });
        ffi::problem_add_callback(self.pin_inner(), boxed);
    }

    /// Registers a persistent callback that survives
    /// [`Problem::clear_callbacks`]. Persistent callbacks run after
    /// non-persistent ones.
    pub fn add_persistent_callback<F>(&mut self, callback: F)
    where
        F: for<'a> FnMut(&IterationInfo<'a>) -> bool + Send + 'static,
    {
        let boxed = Box::new(RustCallback {
            inner: Box::new(callback),
        });
        ffi::problem_add_persistent_callback(self.pin_inner(), boxed);
    }

    /// Clears all registered iteration callbacks.
    pub fn clear_callbacks(&mut self) {
        ffi::problem_clear_callbacks(self.pin_inner());
    }
}

impl std::fmt::Debug for Problem<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Problem")
            .field("cost_function_type", &self.cost_function_type())
            .field("equality_constraint_type", &self.equality_constraint_type())
            .field(
                "inequality_constraint_type",
                &self.inequality_constraint_type(),
            )
            .field("arena_len", &self.arena.len())
            .finish()
    }
}
