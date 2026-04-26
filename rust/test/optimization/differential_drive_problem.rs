//! Port of `test/src/optimization/differential_drive_problem_test.cpp`.

use hafgufa::{ExpressionType, Problem, Variable, VariableArena, VariableMatrix, subject_to};
use ndarray::{Array1, Array2};

use super::common::{differential_drive, lerp::lerp, rk4};

#[test]
fn differential_drive_swing() {
    let dt = 0.05_f64;
    let total_time = 5.0_f64;
    let n: i32 = (total_time / dt) as i32;

    let u_max = 12.0_f64;

    let x_initial: Array1<f64> = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    let x_final: Array1<f64> = Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0, 0.0]);

    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let mut x_mat = problem.decision_variable_matrix(5, n + 1);
    let mut u = problem.decision_variable_matrix(2, n);

    for k in 0..n {
        let t = k as f64 / n as f64;
        x_mat.set_value_at(0, k, lerp(x_initial[0], x_final[0], t));
        x_mat.set_value_at(1, k, lerp(x_initial[1], x_final[1], t));
    }

    let x_initial_col = VariableMatrix::from_array_in(&arena, &to_col(&x_initial));
    let x_final_col = VariableMatrix::from_array_in(&arena, &to_col(&x_final));
    subject_to!(problem, x_mat.col(0) == x_initial_col);
    subject_to!(problem, x_mat.col(n) == x_final_col);

    problem.bound(-u_max, &u, u_max);

    for k in 0..n {
        let x_k = x_mat.col(k);
        let u_k = u.col(k);
        let x_kp1 = rk4::rk4_variable(
            &arena,
            |x, u| differential_drive::dynamics_variable(&arena, x, u),
            x_k,
            u_k,
            dt,
        );
        subject_to!(problem, x_mat.col(k + 1) == x_kp1);
    }

    let mut j = Variable::constant_in(&arena, 0.0);
    for k in 0..n {
        let xk = x_mat.col(k);
        let uk = u.col(k);
        j = j + (xk.t() * xk).get(0, 0) + (uk.t() * uk).get(0, 0);
    }
    problem.minimize(j);

    assert_eq!(problem.cost_function_type(), ExpressionType::Quadratic);
    assert_eq!(
        problem.equality_constraint_type(),
        ExpressionType::Nonlinear
    );
    assert_eq!(problem.inequality_constraint_type(), ExpressionType::Linear);

    problem.solve(Default::default()).unwrap();

    for row in 0..5 {
        assert!(
            (x_mat.value_at(row as i32, 0) - x_initial[row]).abs() < 1e-8,
            "initial row {row}",
        );
    }

    let mut x: Array1<f64> = Array1::zeros(5);
    for k in 0..n {
        let u_col: Array1<f64> = Array1::from_vec(vec![u.value_at(0, k), u.value_at(1, k)]);

        for row in 0..5 {
            assert!(
                (x_mat.value_at(row as i32, k) - x[row]).abs() < 1e-8,
                "step {k} row {row}: solver={}, sim={}",
                x_mat.value_at(row as i32, k),
                x[row],
            );
        }

        assert!(u_col[0] >= -u_max - 1e-8);
        assert!(u_col[0] <= u_max + 1e-8);
        assert!(u_col[1] >= -u_max - 1e-8);
        assert!(u_col[1] <= u_max + 1e-8);

        x = rk4::rk4_scalar(
            |x, u| differential_drive::dynamics_scalar(x, u),
            x,
            &u_col,
            dt,
        );
    }

    for row in 0..5 {
        assert!(
            (x_mat.value_at(row as i32, n) - x_final[row]).abs() < 1e-8,
            "final row {row}",
        );
    }
}

fn to_col(v: &Array1<f64>) -> Array2<f64> {
    let n = v.len();
    let mut m = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        m[[i, 0]] = v[i];
    }
    m
}
