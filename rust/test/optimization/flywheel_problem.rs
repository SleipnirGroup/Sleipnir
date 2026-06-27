//! Port of `test/src/optimization/flywheel_problem_test.cpp`. Drops the
//! offline CSV-logging tail — the file write isn't what the test is
//! asserting.

use hafgufa::{ExpressionType, Problem, VariableArena, VariableMatrix, subject_to};
use ndarray::Array2;

#[test]
fn flywheel_direct_transcription_converges() {
    let total_time = 5.0_f64;
    let dt = 0.005_f64;
    let n: i32 = (total_time / dt) as i32;

    let a = (-dt).exp();
    let b = 1.0 - a;

    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let x = problem.decision_variable_matrix(1, n + 1);
    let u = problem.decision_variable_matrix(1, n);

    for k in 0..n {
        let x_next = x.block(0, k + 1, 1, 1);
        let x_curr = x.block(0, k, 1, 1);
        let u_curr = u.block(0, k, 1, 1);
        subject_to!(problem, x_next == a * x_curr + b * u_curr);
    }

    subject_to!(problem, x.block(0, 0, 1, 1) == 0.0);
    subject_to!(problem, &u >= -12.0);
    subject_to!(problem, &u <= 12.0);

    let r = 10.0_f64;
    let r_mat = VariableMatrix::from_array_in(&arena, &Array2::from_elem((1, (n + 1) as usize), r));
    let diff = &r_mat - &x;
    let cost = &diff * &diff.t();
    problem.minimize_matrix(&cost);

    assert_eq!(problem.cost_function_type(), ExpressionType::Quadratic);
    assert_eq!(problem.equality_constraint_type(), ExpressionType::Linear);
    assert_eq!(problem.inequality_constraint_type(), ExpressionType::Linear);

    problem.solve(Default::default()).unwrap();

    let mut x = x;
    let mut u = u;

    assert!(x.value_at(0, 0).abs() < 1e-8);

    let u_ss = (1.0 - a) * r / b;

    let mut sim_x = 0.0_f64;
    for k in 0..n {
        assert!(
            (x.value_at(0, k) - sim_x).abs() < 1e-2,
            "step {k}: solver x = {}, simulated = {sim_x}",
            x.value_at(0, k),
        );

        let error = r - sim_x;
        let expected_u = if error > 1e-2 { 12.0 } else { u_ss };

        let u_solver = u.value_at(0, k);
        let in_transition = k > 0
            && k < n - 1
            && (u.value_at(0, k - 1) - 12.0).abs() < 1e-2
            && (u.value_at(0, k + 1) - u_ss).abs() < 1e-2;

        if in_transition {
            assert!(u_solver >= u_ss - 1e-4);
            assert!(u_solver <= 12.0 + 1e-4);
        } else {
            assert!(
                (u_solver - expected_u).abs() < 1e-4,
                "step {k}: solver u = {u_solver}, expected = {expected_u}",
            );
        }

        sim_x = a * sim_x + b * expected_u;
    }

    assert!((x.value_at(0, n) - r).abs() < 2e-7);
}
