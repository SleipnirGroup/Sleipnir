//! End-to-end OCP test against the flywheel setup from the Python
//! example: discrete dynamics, fixed timestep, direct transcription.

use std::time::Duration;

use hafgufa::{OCP, TimestepMethod, VariableArena, VariableMatrix};
use ndarray::Array2;

/// Helper that introduces a named `'a` so the closure's three
/// `VariableMatrix<'a>` lifetimes unify. Rust's closure inference picks
/// independent lifetimes by default, which my `Add`/`Mul` impls (all
/// monomorphic on one arena lifetime) don't accept.
fn make_dynamics<'a>(
    a: f64,
    b: f64,
) -> impl FnMut(&VariableMatrix<'a>, &VariableMatrix<'a>) -> VariableMatrix<'a> + Send + 'a {
    move |x, u| a * x + b * u
}

#[test]
fn flywheel_discrete_ocp() {
    let total_time = 5.0_f64; // s
    let dt = 0.005_f64; // s
    let n_steps = (total_time / dt) as i32;

    let a = (-dt).exp();
    let b = 1.0 - a;

    let arena = VariableArena::new();

    let mut ocp = OCP::new_discrete(
        &arena,
        1,
        1,
        Duration::from_secs_f64(dt),
        n_steps,
        make_dynamics(a, b),
        TimestepMethod::Fixed,
    );

    ocp.constrain_initial_state(0.0);
    ocp.set_upper_input_bound(12.0);
    ocp.set_lower_input_bound(-12.0);

    let r = 10.0_f64;
    let r_mat = VariableMatrix::from_array_in(
        &arena,
        &Array2::from_elem((1, (n_steps + 1) as usize), r),
    );
    let diff = &r_mat - &ocp.x();
    let cost_mat = &diff * &diff.t();
    ocp.minimize_matrix(&cost_mat);

    ocp.solve(Default::default()).unwrap();

    // Initial state starts at 0 by the constraint.
    let mut x0 = ocp.x().block(0, 0, 1, 1);
    assert!(x0.value_at(0, 0).abs() < 1e-6);
}
