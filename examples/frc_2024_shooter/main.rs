//! Rust port of `examples/frc_2024_shooter/main.py`.
//!
//! FRC 2024 shooter trajectory optimization.
//!
//! Single-shooting RK4 integration with air resistance, minimizing the
//! z-position sensitivity to initial velocity. Plotting is omitted —
//! matplotlib has no Rust equivalent in the crate graph.

use hafgufa::math::sqrt;
use hafgufa::{Gradient, Problem, VariableArena, VariableMatrix, subject_to};
use ndarray::{Array2, array, s};

const FIELD_WIDTH: f64 = 8.2296; // m (27 ft)
const FIELD_LENGTH: f64 = 16.4592; // m (54 ft)
const TARGET_LOWER_EDGE: f64 = 1.98; // m
const TARGET_UPPER_EDGE: f64 = 2.11; // m
const TARGET_DEPTH: f64 = 0.46; // m

fn target_wrt_field() -> Array2<f64> {
    array![
        [FIELD_LENGTH - TARGET_DEPTH / 2.0],
        [FIELD_WIDTH - 2.6575],
        [(TARGET_UPPER_EDGE + TARGET_LOWER_EDGE) / 2.0],
        [0.0],
        [0.0],
        [0.0],
    ]
}

fn gravity() -> Array2<f64> {
    array![[0.0], [0.0], [9.806]]
}

/// Flight dynamics (continuous-time ẋ). Input is a 6×1 VariableMatrix
/// `[px, py, pz, vx, vy, vz]`. Arena is inferred from `state`.
fn f<'a>(state: &VariableMatrix<'a>) -> VariableMatrix<'a> {
    let arena = state.arena();
    let rho = 1.204_f64; // kg/m³
    let v = state.block(3, 0, 3, 1);
    let v_t = v.t();
    let v2 = (&v_t * &v).get(0, 0);
    let v_norm = sqrt(v2);
    let v_hat = &v / v_norm;

    // omega = [0, 0, 2] rad/s, cross(v, omega) = [2·vy, -2·vx, 0].
    let vx = v.get(0, 0);
    let vy = v.get(1, 0);
    let mut cross = arena.slice(3, 1, &[0.0, 0.0, 0.0]);
    cross.set_variable(0, 0, 2.0 * vy);
    cross.set_variable(1, 0, -2.0 * vx);

    let radius = 0.15_f64; // m
    let area = std::f64::consts::PI * radius * radius; // m²
    let mass = 0.283_f64; // kg

    // F_D = ½ρ|v|² C_D A, C_L = 0.5
    let f_d = 0.5 * rho * 0.5 * area * v2;
    let f_l = 0.5 * rho * 0.5 * area * v_norm;

    let g = arena.array(&gravity());
    let drag = &v_hat * (f_d / mass);
    let lift = &cross * (f_l / mass);
    let accel = (-g - drag) - lift;

    stack_vertical(&v, &accel)
}

/// Vertical-stack two matrices with the same column count. Arena taken
/// from `top`.
fn stack_vertical<'a>(top: &VariableMatrix<'a>, bottom: &VariableMatrix<'a>) -> VariableMatrix<'a> {
    assert_eq!(top.cols(), bottom.cols());
    let rows = top.rows() + bottom.rows();
    let cols = top.cols();
    let mut out = top.arena().zeros(rows, cols);
    let top_rows = top.rows();
    for c in 0..cols {
        for r in 0..top_rows {
            out.set_variable(r, c, top.get(r, c));
        }
        for r in 0..bottom.rows() {
            out.set_variable(top_rows + r, c, bottom.get(r, c));
        }
    }
    out
}

fn main() {
    let target = target_wrt_field();
    let robot_wrt_field: Array2<f64> = array![
        [0.75 * FIELD_LENGTH],
        [FIELD_WIDTH / 3.0],
        [0.0],
        [1.524],
        [-1.524],
        [0.0],
    ];

    let max_initial_velocity = 15.0_f64; // m/s

    let shooter_wrt_robot: Array2<f64> = array![[0.0], [0.0], [0.6096], [0.0], [0.0], [0.0]];
    let shooter_wrt_field = &robot_wrt_field + &shooter_wrt_robot;

    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let n: i32 = 10;
    let t_total = problem.decision_variable();
    subject_to!(problem, t_total >= 0.0);
    t_total.set_value(1.0);
    let dt = t_total / n as f64;

    // 6×1 disc state in field frame.
    let x = problem.decision_variable_matrix(6, 1);

    // Position initial guess: shooter position.
    let shooter_pos = shooter_wrt_field.slice(s![0..3, ..]).to_owned();
    for r in 0..3i32 {
        x.block(0, 0, 3, 1)
            .set_value_at(r, 0, shooter_pos[[r as usize, 0]]);
    }

    // Velocity initial guess: max initial velocity toward target.
    let target_pos = target.slice(s![0..3, ..]).to_owned();
    let mut uvec = &target_pos - &shooter_pos;
    let norm: f64 = uvec.iter().map(|v| v * v).sum::<f64>().sqrt();
    uvec.mapv_inplace(|v| v / norm);
    for r in 0..3i32 {
        let val =
            robot_wrt_field[[(r + 3) as usize, 0]] + max_initial_velocity * uvec[[r as usize, 0]];
        x.block(3, 0, 3, 1).set_value_at(r, 0, val);
    }

    let v0_wrt_shooter =
        x.block(3, 0, 3, 1) - arena.array(&shooter_wrt_field.slice(s![3..6, ..]).to_owned());

    // Shooter initial position constraint.
    let first_pos = x.block(0, 0, 3, 1);
    let shooter_pos_vm = arena.array(&shooter_pos);
    subject_to!(problem, first_pos == shooter_pos_vm);

    // Initial speed cap: (v0 - v_robot)^T (v0 - v_robot) <= vmax².
    let v0_col = x.block(3, 0, 3, 1);
    let v_robot_col = arena.array(&robot_wrt_field.slice(s![3..6, ..]).to_owned());
    let dv = v0_col.clone() - v_robot_col;
    let speed2 = (dv.t() * &dv).get(0, 0);
    subject_to!(
        problem,
        speed2 <= max_initial_velocity * max_initial_velocity
    );

    // Single-shooting RK4 integration.
    let h = dt;
    let half_h = h / 2.0;
    let sixth_h = h / 6.0;
    let mut x_k = x.clone();
    for _ in 0..(n - 1) {
        let k1 = f(&x_k);
        let k2 = f(&(&x_k + &k1 * half_h));
        let k3 = f(&(&x_k + &k2 * half_h));
        let k4 = f(&(&x_k + &k3 * h));
        let sum = k1 + 2.0 * k2 + 2.0 * k3 + k4;
        x_k = x_k + sum * sixth_h;
    }

    // Final position at target center.
    let final_pos = x_k.block(0, 0, 3, 1);
    subject_to!(problem, final_pos == arena.array(&target_pos));

    // Require final velocity is upward.
    let final_vz = x_k.get(5, 0);
    subject_to!(problem, final_vz > 0.0);

    // Minimize sensitivity of final z-position to initial velocity.
    let vel_block = x.block(3, 0, 3, 1);
    let sensitivity = Gradient::new(x_k.get(2, 0), &vel_block).get();
    let cost = (&sensitivity.t() * &sensitivity).get(0, 0);
    problem.minimize(cost);

    match problem.solve(hafgufa::Options::default().diagnostics(true)) {
        Ok(()) => println!("exit status: success"),
        Err(e) => println!("exit status: {e}"),
    }

    // Report shot parameters.
    let v0 = [
        v0_wrt_shooter.get(0, 0).value(),
        v0_wrt_shooter.get(1, 0).value(),
        v0_wrt_shooter.get(2, 0).value(),
    ];
    let velocity = (v0[0].powi(2) + v0[1].powi(2) + v0[2].powi(2)).sqrt();
    println!("Velocity = {velocity:.3} m/s");

    let pitch = v0[2].atan2((v0[0].powi(2) + v0[1].powi(2)).sqrt());
    println!("Pitch = {:.3}\u{b0}", pitch.to_degrees());

    let yaw = v0[1].atan2(v0[0]);
    println!("Yaw = {:.3}\u{b0}", yaw.to_degrees());

    println!("Total time = {:.3} s", t_total.value());
    println!("dt = {:.3} ms", dt.value() * 1e3);
}
