//! Port of `test/src/optimization/cart_pole_ocp_test.cpp`. The C++
//! solution-replay check is `#if 0`'d out; Rust drops it too.

use std::time::Duration;

use hafgufa::{OCP, TimestepMethod, TranscriptionMethod, Variable, VariableArena, VariableMatrix};
use ndarray::{Array2, arr2};

use super::common::{cart_pole, lerp::lerp};

#[test]
fn cart_pole_direct_collocation() {
    let dt = 0.05_f64;
    let total_time = 5.0_f64;
    let n: i32 = (total_time / dt) as i32;

    let u_max = 20.0_f64;
    let d_max = 2.0_f64;

    let x_initial = [0.0_f64, 0.0, 0.0, 0.0];
    let x_final = [1.0_f64, std::f64::consts::PI, 0.0, 0.0];

    let arena = VariableArena::new();
    let arena_ref: &VariableArena = &arena;

    let mut ocp = OCP::new_explicit_ode(
        arena_ref,
        4,
        1,
        Duration::from_secs_f64(dt),
        n,
        move |x, u| cart_pole::dynamics_variable(arena_ref, x, u),
        TimestepMethod::VariableSingle,
        TranscriptionMethod::DirectCollocation,
    );

    // Seed initial guess
    let mut x_mat = ocp.x();
    for k in 0..n + 1 {
        let t = k as f64 / n as f64;
        x_mat.set_value_at(0, k, lerp(x_initial[0], x_final[0], t));
        x_mat.set_value_at(1, k, lerp(x_initial[1], x_final[1], t));
    }

    let x_initial_col = VariableMatrix::from_array_in(arena_ref, &arr2_col(&x_initial));
    let x_final_col = VariableMatrix::from_array_in(arena_ref, &arr2_col(&x_final));
    ocp.constrain_initial_state(&x_initial_col);
    ocp.constrain_final_state(&x_final_col);

    // Cart position bounds per step: 0 ≤ x[0, k] ≤ d_max. The C++ test
    // uses `for_each_step` with a callback that mutates the problem;
    // Rust's borrow checker disallows that shape, so we iterate over
    // the trajectory indices directly.
    let x_trajectory = ocp.x();
    for k in 0..n + 1 {
        let cart = x_trajectory.get(0, k);
        ocp.subject_to(hafgufa::cmp!(cart >= 0.0));
        ocp.subject_to(hafgufa::cmp!(cart <= d_max));
    }

    ocp.set_lower_input_bound(-u_max);
    ocp.set_upper_input_bound(u_max);

    let u = ocp.u();
    let mut j = Variable::constant_in(arena_ref, 0.0);
    for k in 0..n {
        let uk = u.col(k);
        j = j + (uk.t() * uk).get(0, 0);
    }
    ocp.minimize(j);

    ocp.solve(Default::default()).unwrap();

    let mut x_result = ocp.x();
    for row in 0..4 {
        assert!(
            (x_result.value_at(row, 0) - x_initial[row as usize]).abs() < 1e-8,
            "initial row {row}",
        );
    }
    for row in 0..4 {
        assert!(
            (x_result.value_at(row, n) - x_final[row as usize]).abs() < 1e-8,
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
    let _ = arr2::<f64, 1>;
    m
}
