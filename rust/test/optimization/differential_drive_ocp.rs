//! Port of `test/src/optimization/differential_drive_ocp_test.cpp`.
//! The C++ solution-replay check is `#if 0`'d out; Rust drops it too.

use std::time::Duration;

use hafgufa::{OCP, TimestepMethod, TranscriptionMethod, VariableArena, VariableMatrix};
use ndarray::{Array2, arr2};

use super::common::differential_drive;

#[test]
fn differential_drive_variable_single_timestep() {
    let n: i32 = 50;
    let min_timestep = 0.05_f64;

    let x_initial = [0.0_f64, 0.0, 0.0, 0.0, 0.0];
    let x_final = [1.0_f64, 1.0, 0.0, 0.0, 0.0];

    let arena = VariableArena::new();
    let arena_ref: &VariableArena = &arena;

    let mut ocp = OCP::new_explicit_ode(
        arena_ref,
        5,
        2,
        Duration::from_secs_f64(min_timestep),
        n,
        move |x, u| differential_drive::dynamics_variable(arena_ref, x, u),
        TimestepMethod::VariableSingle,
        TranscriptionMethod::DirectTranscription,
    );

    // Initial guess: lerp x and y between start and end.
    let mut x_mat = ocp.x();
    for i in 0..n + 1 {
        let t = i as f64 / (n + 1) as f64;
        x_mat.set_value_at(0, i, t);
        x_mat.set_value_at(1, i, t);
    }

    let x_initial_col = VariableMatrix::from_array_in(arena_ref, &arr2_col(&x_initial));
    let x_final_col = VariableMatrix::from_array_in(arena_ref, &arr2_col(&x_final));
    ocp.constrain_initial_state(&x_initial_col);
    ocp.constrain_final_state(&x_final_col);

    let u_min = VariableMatrix::from_array_in(arena_ref, &arr2(&[[-12.0_f64], [-12.0]]));
    let u_max = VariableMatrix::from_array_in(arena_ref, &arr2(&[[12.0_f64], [12.0]]));
    ocp.set_lower_input_bound(&u_min);
    ocp.set_upper_input_bound(&u_max);

    ocp.set_min_timestep(Duration::from_secs_f64(min_timestep));
    ocp.set_max_timestep(Duration::from_secs_f64(3.0));

    // Minimize total time: dt · [1; 1; …; 1]
    let ones =
        VariableMatrix::from_array_in(arena_ref, &Array2::from_elem(((n + 1) as usize, 1), 1.0));
    let cost = &ocp.dt() * &ones;
    ocp.minimize_matrix(&cost);

    ocp.solve(Default::default()).unwrap();

    let mut x_result = ocp.x();
    for row in 0..5 {
        assert!(
            (x_result.value_at(row as i32, 0) - x_initial[row]).abs() < 1e-8,
            "initial row {row}",
        );
    }
    for row in 0..5 {
        assert!(
            (x_result.value_at(row as i32, n) - x_final[row]).abs() < 1e-8,
            "final row {row}",
        );
    }
}

fn arr2_col(v: &[f64]) -> Array2<f64> {
    let n = v.len();
    let mut m = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        m[[i, 0]] = v[i];
    }
    m
}
