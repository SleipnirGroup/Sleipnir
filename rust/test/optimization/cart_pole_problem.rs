//! Port of `test/src/optimization/cart_pole_problem_test.cpp`. Drops
//! the CSV-logging tail.

use hafgufa::{ExpressionType, Problem, Variable, VariableArena, subject_to};
use ndarray::{Array1, Array2};

use super::common::{cart_pole, lerp::lerp, rk4};

#[test]
fn cart_pole_swing_up() {
    let dt = 0.05_f64;
    let total_time = 5.0_f64;
    let n: i32 = (total_time / dt) as i32;

    let u_max = 20.0_f64; // N
    let d_max = 2.0_f64; // m

    let x_initial: Array1<f64> = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
    let x_final: Array1<f64> = Array1::from_vec(vec![1.0, std::f64::consts::PI, 0.0, 0.0]);

    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    // x = [x, θ, ẋ, θ̇]ᵀ
    let mut x_mat = problem.decision_variable_matrix(4, n + 1);
    let mut u = problem.decision_variable_matrix(1, n);

    // Linear initial guess between x_initial and x_final.
    for k in 0..n + 1 {
        let t = k as f64 / n as f64;
        x_mat.set_value_at(0, k, lerp(x_initial[0], x_final[0], t));
        x_mat.set_value_at(1, k, lerp(x_initial[1], x_final[1], t));
    }

    // Initial conditions
    let x_initial_mat = hafgufa::VariableMatrix::from_array_in(&arena, &to_col(&x_initial));
    let x_final_mat = hafgufa::VariableMatrix::from_array_in(&arena, &to_col(&x_final));
    subject_to!(problem, x_mat.col(0) == x_initial_mat);
    subject_to!(problem, x_mat.col(n) == x_final_mat);

    // Cart-position bounds: 0 ≤ x[0] ≤ d_max (applies to row 0 across all steps)
    problem.bound(0.0, x_mat.row(0), d_max);
    // Input bounds
    problem.bound(-u_max, &u, u_max);

    // RK4 dynamics constraint for each step
    for k in 0..n {
        let x_k = x_mat.col(k);
        let u_k = u.col(k);
        let x_kp1 = rk4::rk4_variable(
            &arena,
            |x, u| cart_pole::dynamics_variable(&arena, x, u),
            x_k,
            u_k,
            dt,
        );
        subject_to!(problem, x_mat.col(k + 1) == x_kp1);
    }

    // Cost: Σ uᵀ u
    let mut j = Variable::constant_in(&arena, 0.0);
    for k in 0..n {
        let uk = u.col(k);
        j = j + (uk.t() * uk).get(0, 0);
    }
    problem.minimize(j);

    assert_eq!(problem.cost_function_type(), ExpressionType::Quadratic);
    assert_eq!(
        problem.equality_constraint_type(),
        ExpressionType::Nonlinear
    );
    assert_eq!(problem.inequality_constraint_type(), ExpressionType::Linear);

    problem.solve(Default::default()).unwrap();

    // Verify initial state
    for row in 0..4 {
        assert!((x_mat.value_at(row as i32, 0) - x_initial[row]).abs() < 1e-8);
    }

    // Verify RK4 dynamics agree between solver and independent f64 rollout
    for k in 0..n {
        let x_col: Array1<f64> = (0..4_i32).map(|r| x_mat.value_at(r, k)).collect();
        let u_col: Array1<f64> = Array1::from_vec(vec![u.value_at(0, k)]);
        let expected = rk4::rk4_scalar(|x, u| cart_pole::dynamics_scalar(x, u), x_col, &u_col, dt);
        for row in 0..4_usize {
            let actual = x_mat.value_at(row as i32, k + 1);
            assert!(
                (actual - expected[row]).abs() < 1e-8,
                "step {k} row {row}: solver={actual}, expected={}",
                expected[row],
            );
        }

        let uv = u.value_at(0, k);
        assert!(uv >= -u_max - 1e-8);
        assert!(uv <= u_max + 1e-8);

        let cart_pos = x_mat.value_at(0, k);
        assert!(cart_pos >= -1e-8);
        assert!(cart_pos <= d_max + 1e-8);
    }

    // Verify final state
    for row in 0..4 {
        assert!((x_mat.value_at(row as i32, n) - x_final[row]).abs() < 1e-8);
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
