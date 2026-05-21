//! Port of `test/src/optimization/arm_on_elevator_problem_test.cpp`.
//! The C++ builds the end-effector height vector via
//! `arm.row(0).cwise_transform(sin)` — Rust has no `cwise_transform`, so
//! the height-limit inequality is built per-step as a scalar constraint.

use hafgufa::math::{pow, sin};
use hafgufa::{ExpressionType, Problem, Variable, VariableArena, VariableMatrix, subject_to};
use ndarray::Array2;

#[test]
fn arm_on_elevator_swing_up() {
    let n: i32 = 800;

    let elevator_start_height = 1.0_f64;
    let elevator_end_height = 1.25_f64;
    let elevator_max_velocity = 1.0_f64;
    let elevator_max_acceleration = 2.0_f64;

    let arm_length = 1.0_f64;
    let arm_start_angle = 0.0_f64;
    let arm_end_angle = std::f64::consts::PI;
    let arm_max_velocity = 2.0 * std::f64::consts::PI;
    let arm_max_acceleration = 4.0 * std::f64::consts::PI;

    let end_effector_max_height = 1.8_f64;

    let total_time = 4.0_f64;
    let dt = total_time / n as f64;

    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let elevator = problem.decision_variable_matrix(2, n + 1);
    let elevator_accel = problem.decision_variable_matrix(1, n);

    let arm = problem.decision_variable_matrix(2, n + 1);
    let arm_accel = problem.decision_variable_matrix(1, n);

    for k in 0..n {
        let pos_next = elevator.block(0, k + 1, 1, 1);
        let pos = elevator.block(0, k, 1, 1);
        let vel = elevator.block(1, k, 1, 1);
        let a = elevator_accel.block(0, k, 1, 1);
        subject_to!(problem, pos_next == pos + vel * dt + 0.5 * dt * dt * &a);

        let vel_next = elevator.block(1, k + 1, 1, 1);
        let vel_cur = elevator.block(1, k, 1, 1);
        subject_to!(problem, vel_next == vel_cur + &a * dt);

        let a_pos_next = arm.block(0, k + 1, 1, 1);
        let a_pos = arm.block(0, k, 1, 1);
        let a_vel = arm.block(1, k, 1, 1);
        let aa = arm_accel.block(0, k, 1, 1);
        subject_to!(
            problem,
            a_pos_next == a_pos + a_vel * dt + 0.5 * dt * dt * &aa
        );

        let a_vel_next = arm.block(1, k + 1, 1, 1);
        let a_vel_cur = arm.block(1, k, 1, 1);
        subject_to!(problem, a_vel_next == a_vel_cur + &aa * dt);
    }

    let elevator_init = VariableMatrix::from_array_in(
        &arena,
        &Array2::from_shape_vec((2, 1), vec![elevator_start_height, 0.0]).unwrap(),
    );
    let elevator_final_mat = VariableMatrix::from_array_in(
        &arena,
        &Array2::from_shape_vec((2, 1), vec![elevator_end_height, 0.0]).unwrap(),
    );
    subject_to!(problem, elevator.col(0) == elevator_init);
    subject_to!(problem, elevator.col(n) == elevator_final_mat);

    let arm_init = VariableMatrix::from_array_in(
        &arena,
        &Array2::from_shape_vec((2, 1), vec![arm_start_angle, 0.0]).unwrap(),
    );
    let arm_final_mat = VariableMatrix::from_array_in(
        &arena,
        &Array2::from_shape_vec((2, 1), vec![arm_end_angle, 0.0]).unwrap(),
    );
    subject_to!(problem, arm.col(0) == arm_init);
    subject_to!(problem, arm.col(n) == arm_final_mat);

    // Elevator velocity limits
    problem.bound(
        -elevator_max_velocity,
        elevator.row(1),
        elevator_max_velocity,
    );
    // Elevator acceleration limits
    problem.bound(
        -elevator_max_acceleration,
        &elevator_accel,
        elevator_max_acceleration,
    );
    // Arm velocity limits
    problem.bound(-arm_max_velocity, arm.row(1), arm_max_velocity);
    // Arm acceleration limits
    problem.bound(-arm_max_acceleration, &arm_accel, arm_max_acceleration);

    // Height limit: elevator height + arm_length * sin(arm angle) <= MAX
    for k in 0..n + 1 {
        let elevator_pos = elevator.get(0, k);
        let arm_angle = arm.get(0, k);
        let height = elevator_pos + arm_length * sin(arm_angle);
        subject_to!(problem, height <= end_effector_max_height);
    }

    // Cost
    let mut j = Variable::constant_in(&arena, 0.0);
    for k in 0..n + 1 {
        let e = elevator.get(0, k);
        let a = arm.get(0, k);
        j = j + pow(elevator_end_height - e, 2.0) + pow(arm_end_angle - a, 2.0);
    }
    problem.minimize(j);

    assert_eq!(problem.cost_function_type(), ExpressionType::Quadratic);
    assert_eq!(problem.equality_constraint_type(), ExpressionType::Linear);
    assert_eq!(
        problem.inequality_constraint_type(),
        ExpressionType::Nonlinear
    );

    problem.solve(Default::default()).unwrap();
}
