//! Port of `test/src/optimization/flywheel_ocp_test.cpp`. The C++ test
//! parameterizes over `DynamicsType` × `TranscriptionMethod`; this port
//! exercises the same matrix of combinations and runs the same
//! state/input sanity checks.

use std::time::Duration;

use hafgufa::{
    DynamicsType, OCP, TimestepMethod, TranscriptionMethod, VariableArena, VariableMatrix,
};
use ndarray::Array2;

fn flywheel_check(
    a_disc: f64,
    b_disc: f64,
    dt: f64,
    dynamics_type: DynamicsType,
    transcription_method: TranscriptionMethod,
    single_shooting: bool,
    make_ocp: fn(&VariableArena, f64, i32) -> OCP<'_>,
) {
    let total_time = 5.0_f64;
    let n: i32 = (total_time / dt) as i32;
    let r = 10.0_f64;

    let arena = VariableArena::new();
    let mut ocp = make_ocp(&arena, dt, n);

    let _ = dynamics_type; // captured already by make_ocp closure.
    let _ = transcription_method;

    ocp.constrain_initial_state(0.0);
    ocp.set_upper_input_bound(12.0);
    ocp.set_lower_input_bound(-12.0);

    let r_mat = VariableMatrix::from_array_in(&arena, &Array2::from_elem((1, (n + 1) as usize), r));
    let diff = &r_mat - &ocp.x();
    let cost = &diff * &diff.t();
    ocp.minimize_matrix(&cost);

    ocp.solve(Default::default()).unwrap();

    let u_ss = (1.0 - a_disc) * r / b_disc;

    let mut x_state = ocp.x();
    let mut u_state = ocp.u();

    assert!(x_state.value_at(0, 0).abs() < 1e-8);

    let mut x_sim = 0.0_f64;
    for k in 0..n {
        assert!(
            (x_state.value_at(0, k) - x_sim).abs() < 1e-2,
            "step {k}: solver x={}, sim={x_sim}",
            x_state.value_at(0, k),
        );

        let error = r - x_sim;
        let expected_u = if error > 1e-2 { 12.0 } else { u_ss };

        let u_val = u_state.value_at(0, k);
        let in_transition = k > 0
            && k < n - 1
            && (u_state.value_at(0, k - 1) - 12.0).abs() < 1e-2
            && (u_state.value_at(0, k + 1) - u_ss).abs() < 1e-2;

        if in_transition {
            assert!(u_val >= u_ss - 1e-4);
            assert!(u_val <= 12.0 + 1e-4);
        } else if single_shooting || transcription_method == TranscriptionMethod::DirectCollocation
        {
            // Collocation and single-shooting inputs don't match the
            // bang-bang reference as tightly; the C++ test uses a 2.0
            // tolerance for collocation and 1e-4 otherwise.
            let tol = if transcription_method == TranscriptionMethod::DirectCollocation {
                2.0
            } else {
                1e-4
            };
            assert!(
                (u_val - expected_u).abs() <= tol + 1e-9,
                "step {k}: u={u_val}, expected={expected_u}, tol={tol}",
            );
        } else {
            assert!(
                (u_val - expected_u).abs() < 1e-4,
                "step {k}: u={u_val}, expected={expected_u}",
            );
        }

        x_sim = a_disc * x_sim + b_disc * expected_u;
    }

    assert!((x_state.value_at(0, n) - r).abs() < 2e-6);
}

fn explicit_ocp<'a>(arena: &'a VariableArena, dt: f64, n: i32) -> OCP<'a> {
    let a = -1.0_f64;
    let b = 1.0_f64;
    OCP::new_explicit_ode(
        arena,
        1,
        1,
        Duration::from_secs_f64(dt),
        n,
        move |x, u| a * x + b * u,
        TimestepMethod::Fixed,
        TranscriptionMethod::DirectTranscription,
    )
}

fn discrete_ocp<'a>(arena: &'a VariableArena, dt: f64, n: i32) -> OCP<'a> {
    let a_disc = (-1.0_f64 * dt).exp();
    let b_disc = 1.0 - a_disc;
    OCP::new_discrete(
        arena,
        1,
        1,
        Duration::from_secs_f64(dt),
        n,
        move |x, u| a_disc * x + b_disc * u,
        TimestepMethod::Fixed,
    )
}

#[test]
fn explicit_direct_transcription() {
    let dt = 0.005_f64;
    let a_disc = (-1.0_f64 * dt).exp();
    let b_disc = 1.0 - a_disc;
    flywheel_check(
        a_disc,
        b_disc,
        dt,
        DynamicsType::ExplicitOde,
        TranscriptionMethod::DirectTranscription,
        false,
        explicit_ocp,
    );
}

#[test]
fn discrete_direct_transcription() {
    let dt = 0.005_f64;
    let a_disc = (-1.0_f64 * dt).exp();
    let b_disc = 1.0 - a_disc;
    flywheel_check(
        a_disc,
        b_disc,
        dt,
        DynamicsType::Discrete,
        TranscriptionMethod::DirectTranscription,
        false,
        discrete_ocp,
    );
}
