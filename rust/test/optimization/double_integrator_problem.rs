//! Port of `test/src/optimization/double_integrator_problem_test.cpp`.

use hafgufa::math::pow;
use hafgufa::{ExpressionType, Problem, Variable, VariableArena, VariableMatrix, subject_to};
use ndarray::Array2;

#[test]
fn double_integrator_bang_coast_bang() {
    let total_time = 3.5_f64;
    let dt = 0.005_f64;
    let n: i32 = (total_time / dt) as i32;

    let r = 2.0_f64;

    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let x = problem.decision_variable_matrix(2, n + 1);
    let u = problem.decision_variable_matrix(1, n);

    for k in 0..n {
        let p_k1 = x.block(0, k + 1, 1, 1);
        let v_k1 = x.block(1, k + 1, 1, 1);
        let p_k = x.block(0, k, 1, 1);
        let v_k = x.block(1, k, 1, 1);
        let a_k = u.block(0, k, 1, 1);

        subject_to!(problem, p_k1 == p_k + &v_k * dt + 0.5 * dt * dt * &a_k);
        subject_to!(problem, v_k1 == &v_k + &a_k * dt);
    }

    let start = VariableMatrix::from_array_in(
        &arena,
        &Array2::from_shape_vec((2, 1), vec![0.0, 0.0]).unwrap(),
    );
    let end = VariableMatrix::from_array_in(
        &arena,
        &Array2::from_shape_vec((2, 1), vec![r, 0.0]).unwrap(),
    );
    subject_to!(problem, x.col(0) == start);
    subject_to!(problem, x.col(n) == end);

    problem.bound(-1.0, x.row(1), 1.0);
    problem.bound(-1.0, &u, 1.0);

    let mut j = Variable::constant_in(&arena, 0.0);
    for k in 0..n + 1 {
        let p = x.get(0, k);
        j = j + pow(r - p, 2.0);
    }
    problem.minimize(j);

    assert_eq!(problem.cost_function_type(), ExpressionType::Quadratic);
    assert_eq!(problem.equality_constraint_type(), ExpressionType::Linear);
    assert_eq!(problem.inequality_constraint_type(), ExpressionType::Linear);

    problem.solve(Default::default()).unwrap();

    let mut x = x;
    let mut u = u;

    // Verify initial and final states
    assert!(x.value_at(0, 0).abs() < 1e-8);
    assert!(x.value_at(1, 0).abs() < 1e-8);
    assert!((x.value_at(0, n) - r).abs() < 1e-8);
    assert!(x.value_at(1, n).abs() < 1e-8);

    // Simulate forward and spot-check
    let a_mat = ndarray::arr2(&[[1.0, dt], [0.0, 1.0]]);
    let b_col = ndarray::arr2(&[[0.5 * dt * dt], [dt]]);

    let mut sim = ndarray::arr2(&[[0.0], [0.0]]);
    for k in 0..n {
        let solver_pos = x.value_at(0, k);
        let solver_vel = x.value_at(1, k);
        assert!(
            (solver_pos - sim[[0, 0]]).abs() < 1e-2,
            "step {k}: pos solver={solver_pos}, sim={}",
            sim[[0, 0]],
        );
        assert!((solver_vel - sim[[1, 0]]).abs() < 1e-2);

        let t = k as f64 * dt;
        let expected_u = if t < 1.0 {
            1.0
        } else if t < 2.05 {
            0.0
        } else if t < 3.275 {
            -1.0
        } else {
            1.0
        };

        let u_solver = u.value_at(0, k);
        let in_transition =
            k > 0 && k < n - 1 && (u.value_at(0, k - 1) - u.value_at(0, k + 1)).abs() >= 1.0 - 1e-2;
        if in_transition {
            assert!(u_solver >= -1.0 - 1e-4);
            assert!(u_solver <= 1.0 + 1e-4);
        } else {
            assert!(
                (u_solver - expected_u).abs() < 1e-4,
                "step {k}: u solver={u_solver}, expected={expected_u}",
            );
        }

        let u_vec = ndarray::arr2(&[[expected_u]]);
        sim = a_mat.dot(&sim) + b_col.dot(&u_vec);
    }
}
