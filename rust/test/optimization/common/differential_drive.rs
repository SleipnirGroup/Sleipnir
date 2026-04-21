//! Differential-drive dynamics. Port of
//! `test/include/differential_drive_util.hpp`.

use hafgufa::math::{cos, sin};
use hafgufa::{Variable, VariableArena, VariableMatrix, vstack};
use ndarray::{Array1, Array2, arr2};

pub const TRACKWIDTH: f64 = 0.699; // m
pub const KV_LINEAR: f64 = 3.02; // V / (m/s)
pub const KA_LINEAR: f64 = 0.642; // V / (m/s²)
pub const KV_ANGULAR: f64 = 1.382; // V / (m/s)
pub const KA_ANGULAR: f64 = 0.08495; // V / (m/s²)

pub fn a_matrix() -> Array2<f64> {
    let a1 = -(KV_LINEAR / KA_LINEAR + KV_ANGULAR / KA_ANGULAR) / 2.0;
    let a2 = -(KV_LINEAR / KA_LINEAR - KV_ANGULAR / KA_ANGULAR) / 2.0;
    arr2(&[[a1, a2], [a2, a1]])
}

pub fn b_matrix() -> Array2<f64> {
    let b1 = 0.5 / KA_LINEAR + 0.5 / KA_ANGULAR;
    let b2 = 0.5 / KA_LINEAR - 0.5 / KA_ANGULAR;
    arr2(&[[b1, b2], [b2, b1]])
}

/// Continuous-time dynamics with state
/// `x = [x, y, heading, v_left, v_right]ᵀ` and input
/// `u = [V_left, V_right]ᵀ`.
pub fn dynamics_scalar(x: &Array1<f64>, u: &Array1<f64>) -> Array1<f64> {
    let v = 0.5 * (x[3] + x[4]);
    let a = a_matrix();
    let b = b_matrix();

    let mut xdot = Array1::<f64>::zeros(5);
    xdot[0] = v * x[2].cos();
    xdot[1] = v * x[2].sin();
    xdot[2] = (x[4] - x[3]) / TRACKWIDTH;

    let vel = arr2(&[[x[3]], [x[4]]]);
    let input = arr2(&[[u[0]], [u[1]]]);
    let accel = a.dot(&vel) + b.dot(&input);
    xdot[3] = accel[[0, 0]];
    xdot[4] = accel[[1, 0]];
    xdot
}

/// Variable version of the dynamics for use inside an OCP.
pub fn dynamics_variable<'a>(
    arena: &'a VariableArena,
    x: &VariableMatrix<'a>,
    u: &VariableMatrix<'a>,
) -> VariableMatrix<'a> {
    let heading = x.get(2, 0);
    let v_left = x.get(3, 0);
    let v_right = x.get(4, 0);

    let v = (v_left + v_right) / Variable::constant_in(arena, 2.0);

    let xdot0 = v * cos(heading);
    let xdot1 = v * sin(heading);
    let xdot2 = (v_right - v_left) / Variable::constant_in(arena, TRACKWIDTH);

    let vel = {
        let mut m = VariableMatrix::zeros_in(arena, 2, 1);
        m.set_variable(0, 0, v_left);
        m.set_variable(1, 0, v_right);
        m
    };

    let a = VariableMatrix::from_array_in(arena, &a_matrix());
    let b_mat = VariableMatrix::from_array_in(arena, &b_matrix());
    let u_col = u.block(0, 0, 2, 1);
    let accel = &(&a * &vel) + &(&b_mat * &u_col);

    let top = {
        let mut m = VariableMatrix::zeros_in(arena, 3, 1);
        m.set_variable(0, 0, xdot0);
        m.set_variable(1, 0, xdot1);
        m.set_variable(2, 0, xdot2);
        m
    };

    vstack(&top, &accel)
}
