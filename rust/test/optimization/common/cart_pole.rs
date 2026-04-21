//! Cart-pole system parameters and dynamics. Port of
//! `test/include/cart_pole_util.hpp`. The C++ class is templated over the
//! scalar type; Rust splits it into a pair of free functions keyed on
//! whether the caller needs f64 or `VariableMatrix` dynamics.

use hafgufa::math::{cos, sin};
use hafgufa::{VariableArena, VariableMatrix, solve, vstack};
use ndarray::{Array1, Array2, arr2};

pub const M_C: f64 = 5.0; // cart mass
pub const M_P: f64 = 0.5; // pole mass
pub const L: f64 = 0.5; // pole length
pub const G: f64 = 9.806; // gravity

/// Continuous-time cart-pole dynamics with scalar state and input.
///
/// `x = [x, θ, ẋ, θ̇]ᵀ`, `u = [fₓ]ᵀ`. Returns `ẋ`.
pub fn dynamics_scalar(x: &Array1<f64>, u: &Array1<f64>) -> Array1<f64> {
    let theta = x[1];
    let xdot = x[2];
    let thetadot = x[3];

    // M(q) q̈ = τ_g(q) − C(q, q̇) q̇ + B u
    let m = arr2(&[
        [M_C + M_P, M_P * L * theta.cos()],
        [M_P * L * theta.cos(), M_P * L * L],
    ]);
    let c = arr2(&[[0.0, -M_P * L * thetadot * theta.sin()], [0.0, 0.0]]);
    let tau_g = arr2(&[[0.0], [-M_P * G * L * theta.sin()]]);
    let b = arr2(&[[1.0], [0.0]]);

    // qdot (first 2 entries of state derivative) = velocity part of state
    let qdot = arr2(&[[xdot], [thetadot]]);

    // Solve M * qddot = tau_g − C * qdot + B * u
    let rhs = tau_g + &(-c.dot(&qdot)) + b * u[0];
    let inv_m = invert_2x2(&m);
    let qddot = inv_m.dot(&rhs);

    let mut out = Array1::<f64>::zeros(4);
    out[0] = xdot;
    out[1] = thetadot;
    out[2] = qddot[[0, 0]];
    out[3] = qddot[[1, 0]];
    out
}

/// Variable version of the same dynamics, taking `VariableMatrix` inputs
/// and returning `VariableMatrix`. Used inside the OCP dynamics closure.
pub fn dynamics_variable<'a>(
    arena: &'a VariableArena,
    x: &VariableMatrix<'a>,
    u: &VariableMatrix<'a>,
) -> VariableMatrix<'a> {
    let theta = x.get(1, 0);
    let xdot = x.get(2, 0);
    let thetadot = x.get(3, 0);

    let m_row0 = hafgufa::hstack(
        VariableMatrix::from_variable_in(arena, hafgufa::Variable::constant_in(arena, M_C + M_P)),
        VariableMatrix::from_variable_in(arena, M_P * L * cos(theta)),
    );
    let m_row1 = hafgufa::hstack(
        VariableMatrix::from_variable_in(arena, M_P * L * cos(theta)),
        VariableMatrix::from_variable_in(arena, hafgufa::Variable::constant_in(arena, M_P * L * L)),
    );
    let m = vstack(&m_row0, &m_row1);

    let c_row0 = hafgufa::hstack(
        VariableMatrix::from_variable_in(arena, hafgufa::Variable::constant_in(arena, 0.0)),
        VariableMatrix::from_variable_in(arena, -M_P * L * thetadot * sin(theta)),
    );
    let c_row1 = VariableMatrix::from_array_in(arena, &arr2(&[[0.0, 0.0]]));
    let c = vstack(&c_row0, &c_row1);

    let tau_g = VariableMatrix::from_array_in(arena, &arr2(&[[0.0], [0.0]]));
    let tau_g_filled = {
        let mut t = tau_g;
        t.set_variable(1, 0, -M_P * G * L * sin(theta));
        t
    };
    let b = VariableMatrix::from_array_in(arena, &arr2(&[[1.0], [0.0]]));

    let qdot = {
        let mut q = VariableMatrix::zeros_in(arena, 2, 1);
        q.set_variable(0, 0, xdot);
        q.set_variable(1, 0, thetadot);
        q
    };

    let u0 = u.get(0, 0);
    let bu = &b * u0;
    let rhs = &tau_g_filled - &(&c * &qdot) + bu;
    let qddot = solve(&m, &rhs);

    // Stack [qdot; qddot]
    vstack(&qdot, &qddot)
}

fn invert_2x2(m: &Array2<f64>) -> Array2<f64> {
    let det = m[[0, 0]] * m[[1, 1]] - m[[0, 1]] * m[[1, 0]];
    let inv_det = 1.0 / det;
    arr2(&[
        [m[[1, 1]] * inv_det, -m[[0, 1]] * inv_det],
        [-m[[1, 0]] * inv_det, m[[0, 0]] * inv_det],
    ])
}
