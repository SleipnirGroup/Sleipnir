use std::marker::PhantomData;
use std::time::Duration;

use cxx::UniquePtr;

use crate::arena::VariableArena;
use crate::ffi::{RustDynamics, RustDynamics4, ffi};
use crate::problem::Options;
use crate::variable::Variable;
use crate::variable::__dsl::IntoMatrixOperand;
use crate::variable_matrix::VariableMatrix;
use crate::{ExitStatus, IterationInfo, SleipnirError};

/// Whether the dynamics function describes an explicit ODE or a discrete
/// state-transition function.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DynamicsType {
    ExplicitOde,
    Discrete,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimestepMethod {
    Fixed = 0,
    VariableSingle = 1,
    Variable = 2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TranscriptionMethod {
    DirectTranscription = 0,
    DirectCollocation = 1,
    SingleShooting = 2,
}

/// Constrained optimal control problem builder, mirroring
/// `slp::OCP<double>`.
///
/// Bound to a [`VariableArena`] via `'arena`. The dynamics closure
/// receives matrices whose lifetime is `'arena` and must return a matrix
/// whose lifetime is also `'arena` (typically the same arena, since
/// matrices only come from arena-aware constructors).
pub struct OCP<'arena> {
    inner: UniquePtr<ffi::OCP>,
    arena: &'arena VariableArena,
    _marker: PhantomData<&'arena ()>,
}

impl<'arena> OCP<'arena> {
    /// Builds an OCP for a discrete state-transition `x_{k+1} = f(x_k, u_k)`.
    pub fn new_discrete<F>(
        arena: &'arena VariableArena,
        num_states: i32,
        num_inputs: i32,
        dt: Duration,
        num_steps: i32,
        dynamics: F,
        timestep_method: TimestepMethod,
    ) -> Self
    where
        F: FnMut(&VariableMatrix<'arena>, &VariableMatrix<'arena>) -> VariableMatrix<'arena>
            + Send
            + 'arena,
    {
        let boxed = box_dynamics(arena, dynamics);
        Self {
            inner: ffi::ocp_new_discrete(
                num_states,
                num_inputs,
                dt.as_secs_f64(),
                num_steps,
                boxed,
                timestep_method as u8,
            ),
            arena,
            _marker: PhantomData,
        }
    }

    /// Builds an OCP for a discrete state-transition
    /// `xₖ₊₁ = f(t, xₖ, uₖ, dt)` with explicit time and timestep
    /// arguments. Use this when the dynamics are time-varying.
    pub fn new_discrete_t<F>(
        arena: &'arena VariableArena,
        num_states: i32,
        num_inputs: i32,
        dt: Duration,
        num_steps: i32,
        dynamics: F,
        timestep_method: TimestepMethod,
    ) -> Self
    where
        F: FnMut(
                &Variable<'arena>,
                &VariableMatrix<'arena>,
                &VariableMatrix<'arena>,
                &Variable<'arena>,
            ) -> VariableMatrix<'arena>
            + Send
            + 'arena,
    {
        let boxed = box_dynamics4(arena, dynamics);
        Self {
            inner: ffi::ocp_new_discrete_4arg(
                num_states,
                num_inputs,
                dt.as_secs_f64(),
                num_steps,
                boxed,
                timestep_method as u8,
            ),
            arena,
            _marker: PhantomData,
        }
    }

    /// Builds an OCP for a time-varying explicit ODE
    /// `dx/dt = f(t, x, u, dt)`.
    pub fn new_explicit_ode_t<F>(
        arena: &'arena VariableArena,
        num_states: i32,
        num_inputs: i32,
        dt: Duration,
        num_steps: i32,
        dynamics: F,
        timestep_method: TimestepMethod,
        transcription: TranscriptionMethod,
    ) -> Self
    where
        F: FnMut(
                &Variable<'arena>,
                &VariableMatrix<'arena>,
                &VariableMatrix<'arena>,
                &Variable<'arena>,
            ) -> VariableMatrix<'arena>
            + Send
            + 'arena,
    {
        let boxed = box_dynamics4(arena, dynamics);
        Self {
            inner: ffi::ocp_new_explicit_ode_4arg(
                num_states,
                num_inputs,
                dt.as_secs_f64(),
                num_steps,
                boxed,
                timestep_method as u8,
                transcription as u8,
            ),
            arena,
            _marker: PhantomData,
        }
    }

    /// Builds an OCP for an explicit ODE `dx/dt = f(x, u)`.
    pub fn new_explicit_ode<F>(
        arena: &'arena VariableArena,
        num_states: i32,
        num_inputs: i32,
        dt: Duration,
        num_steps: i32,
        dynamics: F,
        timestep_method: TimestepMethod,
        transcription: TranscriptionMethod,
    ) -> Self
    where
        F: FnMut(&VariableMatrix<'arena>, &VariableMatrix<'arena>) -> VariableMatrix<'arena>
            + Send
            + 'arena,
    {
        let boxed = box_dynamics(arena, dynamics);
        Self {
            inner: ffi::ocp_new_explicit_ode(
                num_states,
                num_inputs,
                dt.as_secs_f64(),
                num_steps,
                boxed,
                timestep_method as u8,
                transcription as u8,
            ),
            arena,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn arena(&self) -> &'arena VariableArena {
        self.arena
    }

    fn pin_inner(&mut self) -> std::pin::Pin<&mut ffi::OCP> {
        self.inner
            .as_mut()
            .expect("OCP FFI pointer was unexpectedly null")
    }

    pub fn constrain_initial_state<T: IntoMatrixOperand<'arena>>(&mut self, initial: T) {
        let m = initial.into_matrix(self.arena);
        ffi::ocp_constrain_initial_state(self.pin_inner(), m.as_ref());
    }

    pub fn constrain_final_state<T: IntoMatrixOperand<'arena>>(&mut self, final_state: T) {
        let m = final_state.into_matrix(self.arena);
        ffi::ocp_constrain_final_state(self.pin_inner(), m.as_ref());
    }

    pub fn set_lower_input_bound<T: IntoMatrixOperand<'arena>>(&mut self, bound: T) {
        let m = bound.into_matrix(self.arena);
        ffi::ocp_set_lower_input_bound_matrix(self.pin_inner(), m.as_ref());
    }

    pub fn set_upper_input_bound<T: IntoMatrixOperand<'arena>>(&mut self, bound: T) {
        let m = bound.into_matrix(self.arena);
        ffi::ocp_set_upper_input_bound_matrix(self.pin_inner(), m.as_ref());
    }

    pub fn set_min_timestep(&mut self, min: Duration) {
        ffi::ocp_set_min_timestep(self.pin_inner(), min.as_secs_f64());
    }

    pub fn set_max_timestep(&mut self, max: Duration) {
        ffi::ocp_set_max_timestep(self.pin_inner(), max.as_secs_f64());
    }

    /// State trajectory: `(num_states) × (num_steps + 1)`.
    pub fn x(&mut self) -> VariableMatrix<'arena> {
        let arena = self.arena;
        VariableMatrix::from_unique_in(arena, ffi::ocp_X(self.pin_inner()))
    }

    /// Input trajectory: `(num_inputs) × (num_steps + 1)`.
    pub fn u(&mut self) -> VariableMatrix<'arena> {
        let arena = self.arena;
        VariableMatrix::from_unique_in(arena, ffi::ocp_U(self.pin_inner()))
    }

    /// Timestep trajectory: `1 × (num_steps + 1)`.
    pub fn dt(&mut self) -> VariableMatrix<'arena> {
        let arena = self.arena;
        VariableMatrix::from_unique_in(arena, ffi::ocp_dt(self.pin_inner()))
    }

    pub fn initial_state(&mut self) -> VariableMatrix<'arena> {
        let arena = self.arena;
        VariableMatrix::from_unique_in(arena, ffi::ocp_initial_state(self.pin_inner()))
    }

    pub fn final_state(&mut self) -> VariableMatrix<'arena> {
        let arena = self.arena;
        VariableMatrix::from_unique_in(arena, ffi::ocp_final_state(self.pin_inner()))
    }

    pub fn minimize(&mut self, cost: Variable<'arena>) {
        ffi::ocp_minimize(self.pin_inner(), cost.as_ref());
    }

    pub fn minimize_matrix(&mut self, cost: &VariableMatrix<'arena>) {
        ffi::ocp_minimize_matrix(self.pin_inner(), cost.as_ref());
    }

    pub fn maximize(&mut self, objective: Variable<'arena>) {
        ffi::ocp_maximize(self.pin_inner(), objective.as_ref());
    }

    pub fn subject_to_equality(&mut self, c: crate::EqualityConstraints<'arena>) {
        ffi::ocp_subject_to_eq(self.pin_inner(), c.as_ref());
    }

    pub fn subject_to_inequality(&mut self, c: crate::InequalityConstraints<'arena>) {
        ffi::ocp_subject_to_ineq(self.pin_inner(), c.as_ref());
    }

    pub fn subject_to<C: Into<crate::Constraint<'arena>>>(&mut self, c: C) {
        match c.into() {
            crate::Constraint::Equality(e) => self.subject_to_equality(e),
            crate::Constraint::Inequality(i) => self.subject_to_inequality(i),
        }
    }

    /// Adds `lower <= x` and `x <= upper` as two inequality constraints.
    pub fn bound<L, X, U>(&mut self, lower: L, x: X, upper: U)
    where
        L: IntoMatrixOperand<'arena>,
        X: crate::__marker::HasArena<'arena> + IntoMatrixOperand<'arena> + Clone,
        U: IntoMatrixOperand<'arena>,
    {
        for c in crate::bounds(lower, x, upper) {
            self.subject_to_inequality(c);
        }
    }

    /// Runs the solver. `Ok(())` on success, `Err` otherwise.
    pub fn solve(&mut self, options: Options) -> Result<(), SleipnirError> {
        self.solve_status(options).into_result()
    }

    pub fn solve_status(&mut self, options: Options) -> ExitStatus {
        let raw = ffi::ocp_solve(self.pin_inner(), options.into_ffi());
        ExitStatus::from_raw(raw)
    }

    pub fn add_callback<F>(&mut self, callback: F)
    where
        F: for<'a> FnMut(&IterationInfo<'a>) -> bool + Send + 'static,
    {
        let boxed = Box::new(crate::ffi::RustCallback {
            inner: Box::new(callback),
        });
        ffi::ocp_add_callback(self.pin_inner(), boxed);
    }

    pub fn add_persistent_callback<F>(&mut self, callback: F)
    where
        F: for<'a> FnMut(&IterationInfo<'a>) -> bool + Send + 'static,
    {
        let boxed = Box::new(crate::ffi::RustCallback {
            inner: Box::new(callback),
        });
        ffi::ocp_add_persistent_callback(self.pin_inner(), boxed);
    }

    pub fn clear_callbacks(&mut self) {
        ffi::ocp_clear_callbacks(self.pin_inner());
    }

    /// Number of control steps the OCP was constructed with (the `U` and
    /// `dt` matrices each have `num_steps + 1` columns).
    pub fn num_steps(&self) -> i32 {
        let r = self
            .inner
            .as_ref()
            .expect("OCP FFI pointer was unexpectedly null");
        ffi::ocp_num_steps(r)
    }

    /// Invoke `callback` with the `(x, u)` slices for each control step
    /// `0..=num_steps`, matching C++'s `OCP::for_each_step(f)`.
    pub fn for_each_step<F>(&mut self, mut callback: F)
    where
        F: FnMut(&VariableMatrix<'arena>, &VariableMatrix<'arena>),
    {
        let n = self.num_steps() + 1;
        let x = self.x();
        let u = self.u();
        for i in 0..n {
            callback(&x.col(i), &u.col(i));
        }
    }

    /// Invoke `callback` with `(t, x, u, dt)` for each control step
    /// `0..=num_steps`. Matches the 4-arg overload of C++'s
    /// `OCP::for_each_step`. Time accumulates across steps as
    /// `t_{i+1} = t_i + dt_i`.
    pub fn for_each_step_t<F>(&mut self, mut callback: F)
    where
        F: FnMut(
            &Variable<'arena>,
            &VariableMatrix<'arena>,
            &VariableMatrix<'arena>,
            &Variable<'arena>,
        ),
    {
        let arena = self.arena;
        let n = self.num_steps() + 1;
        let x = self.x();
        let u = self.u();
        let dt = self.dt();
        let mut time = Variable::constant_in(arena, 0.0);
        for i in 0..n {
            let step_dt = dt.get(0, i);
            callback(&time, &x.col(i), &u.col(i), &step_dt);
            time = time + step_dt;
        }
    }
}

/// Package a user dynamics closure into a `RustDynamics` box the cxx
/// bridge can consume.
///
/// # Safety
///
/// The cxx bridge declares `RustDynamics.inner` as `'static`, but the
/// user's closure borrows the arena with lifetime `'arena`. The closure
/// is only ever invoked *during* the sibling `ocp_new_*` FFI call in
/// the C++ shim — the Sleipnir OCP constructor consumes the dynamics
/// synchronously via `constrain_direct_transcription`/etc. and doesn't
/// retain it past construction. `'arena` covers that window (it
/// outlives the OCP we're building), so the transmute from `'arena` to
/// `'static` is sound.
impl std::fmt::Debug for OCP<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OCP")
            .field("num_steps", &self.num_steps())
            .finish_non_exhaustive()
    }
}

fn box_dynamics4<'arena, F>(
    arena: &'arena VariableArena,
    mut dynamics: F,
) -> Box<RustDynamics4>
where
    F: FnMut(
            &Variable<'arena>,
            &VariableMatrix<'arena>,
            &VariableMatrix<'arena>,
            &Variable<'arena>,
        ) -> VariableMatrix<'arena>
        + Send
        + 'arena,
{
    type Erased = Box<
        dyn FnMut(
                &ffi::Variable,
                &ffi::VariableMatrix,
                &ffi::VariableMatrix,
                &ffi::Variable,
            ) -> cxx::UniquePtr<ffi::VariableMatrix>
            + Send,
    >;

    let inner: Box<
        dyn FnMut(
                &ffi::Variable,
                &ffi::VariableMatrix,
                &ffi::VariableMatrix,
                &ffi::Variable,
            ) -> cxx::UniquePtr<ffi::VariableMatrix>
            + Send
            + 'arena,
    > = Box::new(move |t_ref, x_ref, u_ref, dt_ref| {
        // Wrap raw C++ Variable/Matrix refs as arena-borrowed handles
        // for the user's closure. The underlying FFI pointers are
        // valid for the invocation; we copy them into arena-owned
        // storage when the user produces the result.
        let t = unsafe { variable_from_raw(arena, t_ref) };
        let dt = unsafe { variable_from_raw(arena, dt_ref) };
        let x = VariableMatrix::from_unique_in(arena, ffi::variable_matrix_clone(x_ref));
        let u = VariableMatrix::from_unique_in(arena, ffi::variable_matrix_clone(u_ref));
        let result = dynamics(&t, &x, &u, &dt);
        result.inner
    });

    // SAFETY: see `box_dynamics` — closure invoked only during the
    // sibling `ocp_new_*_4arg` FFI call.
    let inner_static: Erased = unsafe { std::mem::transmute(inner) };
    Box::new(RustDynamics4 { inner: inner_static })
}

/// Wrap a borrowed C++ `slp::Variable<double>` (as exposed through the
/// shim's `ffi::Variable` opaque type) into an arena-backed
/// `Variable<'arena>` by copying the expression node. The copy is the
/// same shared-ptr bump the shim uses for matrices.
///
/// # Safety
///
/// The `raw` reference must point at a valid `slp::Variable<double>`
/// that outlives the returned `Variable<'arena>`. In our use the raw
/// pointer comes from the shim's `invoke_rust_dynamics4` glue, which
/// keeps the source alive for the whole invocation.
unsafe fn variable_from_raw<'arena>(
    arena: &'arena VariableArena,
    raw: &ffi::Variable,
) -> Variable<'arena> {
    Variable::from_unique_in(arena, ffi::variable_clone_ptr(raw))
}

fn box_dynamics<'arena, F>(arena: &'arena VariableArena, mut dynamics: F) -> Box<RustDynamics>
where
    F: FnMut(&VariableMatrix<'arena>, &VariableMatrix<'arena>) -> VariableMatrix<'arena>
        + Send
        + 'arena,
{
    type Erased = Box<
        dyn FnMut(
                &ffi::VariableMatrix,
                &ffi::VariableMatrix,
            ) -> cxx::UniquePtr<ffi::VariableMatrix>
            + Send,
    >;

    let inner: Box<
        dyn FnMut(
                &ffi::VariableMatrix,
                &ffi::VariableMatrix,
            ) -> cxx::UniquePtr<ffi::VariableMatrix>
            + Send
            + 'arena,
    > = Box::new(move |x_ref, u_ref| {
        let x = VariableMatrix::from_unique_in(arena, ffi::variable_matrix_clone(x_ref));
        let u = VariableMatrix::from_unique_in(arena, ffi::variable_matrix_clone(u_ref));
        let result = dynamics(&x, &u);
        result.inner
    });

    // SAFETY: see fn doc — closure is invoked only while OCP::new_*
    // is still on the Rust call stack.
    let inner_static: Erased = unsafe { std::mem::transmute(inner) };
    Box::new(RustDynamics { inner: inner_static })
}
