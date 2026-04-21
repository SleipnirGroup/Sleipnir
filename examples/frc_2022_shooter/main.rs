//! Rust port of `examples/frc_2022_shooter/main.py`.
//!
//! FRC 2022 shooter trajectory optimization.
//!
//! Uses the `VariableMatrix` API for the 6×N state trajectory, mirroring the
//! Python source's `VariableMatrix`/numpy usage. Plotting is omitted since
//! matplotlib has no Rust equivalent in the crate graph.

use hafgufa::math::{max as vmax, sqrt};
use hafgufa::{Problem, VariableArena, VariableMatrix, subject_to};
use ndarray::{Array2, array, s};

const FIELD_WIDTH: f64 = 8.2296; // m (27 ft)
const FIELD_LENGTH: f64 = 16.4592; // m (54 ft)
const TARGET_RADIUS: f64 = 0.61; // m
const CONE_ANGLE: f64 = std::f64::consts::FRAC_PI_4; // rad

fn target_wrt_field() -> Array2<f64> {
    array![
        [FIELD_LENGTH / 2.0],
        [FIELD_WIDTH / 2.0],
        [2.64],
        [0.0],
        [0.0],
        [0.0],
    ]
}

fn gravity() -> Array2<f64> {
    array![[0.0], [0.0], [9.806]]
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// Flight dynamics (continuous-time ẋ). Input is a 6×1 VariableMatrix of
/// `[px, py, pz, vx, vy, vz]`. Arena is inferred from `state`.
fn f<'a>(state: &VariableMatrix<'a>) -> VariableMatrix<'a> {
    let arena = state.arena();
    let rho = 1.204_f64; // kg/m³
    let v = state.block(3, 0, 3, 1);
    let v_t = v.t();
    let v2 = (&v_t * &v).get(0, 0);
    let v_norm = sqrt(v2);
    let v_hat = &v / v_norm;

    // omega = [0, -2, 0] rad/s, cross(v, omega) = [2·vz, 0, -2·vx]
    let vx = v.get(0, 0);
    let vz = v.get(2, 0);
    let mut cross = VariableMatrix::from_slice_in(arena, 3, 1, &[0.0, 0.0, 0.0]);
    cross.set_variable(0, 0, 2.0 * vz);
    cross.set_variable(2, 0, -2.0 * vx);

    let radius = 0.15_f64; // m
    let area = std::f64::consts::PI * radius * radius;
    let mass = 0.283_f64;

    let f_d = 0.5 * rho * 0.5 * area * v2;
    let f_l = 0.5 * rho * 0.5 * area * v_norm;

    let g = VariableMatrix::from_array_in(arena, &gravity());
    let drag = &v_hat * (f_d / mass);
    let lift = &cross * (f_l / mass);
    let accel = (-g - drag) - lift;

    stack_vertical(&v, &accel)
}

/// Vertical-stack two matrices with the same column count. Arena taken
/// from `top`.
fn stack_vertical<'a>(
    top: &VariableMatrix<'a>,
    bottom: &VariableMatrix<'a>,
) -> VariableMatrix<'a> {
    assert_eq!(top.cols(), bottom.cols());
    let rows = top.rows() + bottom.rows();
    let cols = top.cols();
    let mut out = VariableMatrix::zeros_in(top.arena(), rows, cols);
    let top_rows = top.rows();
    for c in 0..cols {
        for r in 0..top_rows {
            let v = top.get(r, c);
            out.set_variable(r, c, v);
        }
        for r in 0..bottom.rows() {
            let v = bottom.get(r, c);
            out.set_variable(top_rows + r, c, v);
        }
    }
    out
}

fn main() {
    let target = target_wrt_field();
    let robot_wrt_field: Array2<f64> = array![
        [FIELD_LENGTH / 4.0],
        [FIELD_WIDTH / 4.0],
        [0.0],
        [1.524],
        [-1.524],
        [0.0],
    ];

    let max_initial_velocity = 10.0_f64; // m/s

    let shooter_wrt_robot: Array2<f64> = array![[0.0], [0.0], [1.2], [0.0], [0.0], [0.0]];
    let shooter_wrt_field = &robot_wrt_field + &shooter_wrt_robot;

    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let n: i32 = 50;
    let t_total = problem.decision_variable();
    subject_to!(problem, t_total >= 0.0);
    t_total.set_value(1.0);
    let dt = t_total / n as f64;

    // 6×N state trajectory.
    let x_mat = problem.decision_variable_matrix(6, n);

    // Velocity-initial-guess unit vector toward target.
    let shooter_pos = shooter_wrt_field.slice(s![0..3, ..]).to_owned();
    let target_pos = target.slice(s![0..3, ..]).to_owned();
    let mut uvec = &target_pos - &shooter_pos;
    let norm: f64 = uvec.iter().map(|v| v * v).sum::<f64>().sqrt();
    uvec.mapv_inplace(|v| v / norm);

    // Initial guesses for positions (linear interp) and velocities.
    let mut x_state = x_mat.clone();
    for k in 0..n {
        let t = k as f64 / n as f64;
        for r in 0..3usize {
            let val = lerp(shooter_pos[[r, 0]], target_pos[[r, 0]], t);
            x_state.set_value_at(r as i32, k, val);
        }
        for r in 0..3usize {
            let val = robot_wrt_field[[r + 3, 0]] + max_initial_velocity * uvec[[r, 0]];
            x_state.set_value_at((r + 3) as i32, k, val);
        }
    }

    // Shooter initial position constraint.
    let first_pos = x_mat.block(0, 0, 3, 1);
    let shooter_pos_vm = VariableMatrix::from_array_in(&arena, &shooter_pos);
    subject_to!(problem, first_pos == shooter_pos_vm);

    // Initial speed within the cap: (v0 - v_robot)^T (v0 - v_robot) <= vmax².
    let v0_col = x_mat.block(3, 0, 3, 1);
    let v_robot_col =
        VariableMatrix::from_array_in(&arena, &robot_wrt_field.slice(s![3..6, ..]).to_owned());
    let dv = v0_col.clone() - v_robot_col;
    let speed2 = (dv.t() * &dv).get(0, 0);
    subject_to!(problem, speed2 <= max_initial_velocity * max_initial_velocity);

    // Keep-out region: outside cylinder OR inside cone. Applied per column.
    let x_c = target[[0, 0]];
    let y_c = target[[1, 0]];
    let z_c = target[[2, 0]] - TARGET_RADIUS / CONE_ANGLE.tan();
    let tan2 = CONE_ANGLE.tan().powi(2);

    for k in 0..n {
        let dx = x_mat.get(0, k) - x_c;
        let dy = x_mat.get(1, k) - y_c;
        let dz = x_mat.get(2, k) - z_c;

        // Variable is Copy so dx/dy/dz can be reused directly.
        let dx2 = dx * dx;
        let dy2 = dy * dy;
        let dz2 = dz * dz;

        let cylinder = dx2 + dy2 - TARGET_RADIUS * TARGET_RADIUS;
        let cone = dz2 * tan2 - dx2 - dy2;

        subject_to!(problem, vmax(cylinder, cone) >= 0.0);
    }

    // RK4 dynamics constraint.
    let half_dt = dt / 2.0;
    let sixth_dt = dt / 6.0;
    for k in 0..n - 1 {
        let x_k = x_mat.col(k);
        let x_k1 = x_mat.col(k + 1);

        let k1 = f(&x_k);
        let k2 = f(&(&x_k + &k1 * half_dt));
        let k3 = f(&(&x_k + &k2 * half_dt));
        let k4 = f(&(&x_k + &k3 * dt));

        let sum = k1 + 2.0 * k2 + 2.0 * k3 + k4;
        subject_to!(problem, x_k1 == x_k + sum * sixth_dt);
    }

    // Final position at target center.
    let final_pos = x_mat.block(0, n - 1, 3, 1);
    subject_to!(problem, final_pos == VariableMatrix::from_array_in(&arena, &target_pos));

    // Final velocity points downward.
    let final_vz = x_mat.get(5, n - 1);
    subject_to!(problem, final_vz < 0.0);

    // Cost: minimize shooter-relative initial velocity squared.
    let shooter_vel_vm =
        VariableMatrix::from_array_in(&arena, &shooter_wrt_field.slice(s![3..6, ..]).to_owned());
    let v0_wrt_shooter = &v0_col - &shooter_vel_vm;
    let cost = (&v0_wrt_shooter.t() * &v0_wrt_shooter).get(0, 0);
    problem.minimize(cost);

    #[allow(unused_mut)]
    let mut opts = hafgufa::Options::default();
    #[cfg(feature = "diagnostics")]
    {
        opts = opts.diagnostics(true);
    }
    match problem.solve(opts) {
        Ok(()) => println!("exit status: success"),
        Err(e) => println!("exit status: {e}"),
    }

    // Read back the first-column velocity and report the shot parameters.
    let mut x_mat = x_mat;
    let v0 = [
        x_mat.value_at(3, 0) - shooter_wrt_field[[3, 0]],
        x_mat.value_at(4, 0) - shooter_wrt_field[[4, 0]],
        x_mat.value_at(5, 0) - shooter_wrt_field[[5, 0]],
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
