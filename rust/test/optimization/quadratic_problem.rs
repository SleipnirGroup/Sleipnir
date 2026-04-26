//! Port of `test/src/optimization/quadratic_problem_test.cpp`.

use hafgufa::{ExpressionType, Problem, VariableArena, VariableMatrix, subject_to};
use ndarray::Array2;

const EPS: f64 = 1e-5;

#[test]
fn unconstrained_1d() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let x = problem.decision_variable();
    x.set_value(2.0);

    problem.minimize(x * x - 6.0 * x);

    assert_eq!(problem.cost_function_type(), ExpressionType::Quadratic);
    assert_eq!(problem.equality_constraint_type(), ExpressionType::None);
    assert_eq!(problem.inequality_constraint_type(), ExpressionType::None);

    problem.solve(Default::default()).unwrap();

    assert!((x.value() - 3.0).abs() < 1e-6);
}

#[test]
fn unconstrained_2d_scalars() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let x = problem.decision_variable();
    let y = problem.decision_variable();
    x.set_value(1.0);
    y.set_value(2.0);

    problem.minimize(x * x + y * y);

    problem.solve(Default::default()).unwrap();

    assert!(x.value().abs() < 1e-6);
    assert!(y.value().abs() < 1e-6);
}

#[test]
fn unconstrained_2d_matrix() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let mut x = problem.decision_variable_matrix(2, 1);
    x.set_value(&Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap());

    let cost = (&x.t() * &x).get(0, 0);
    problem.minimize(cost);

    problem.solve(Default::default()).unwrap();

    assert!(x.value_at(0, 0).abs() < 1e-6);
    assert!(x.value_at(1, 0).abs() < 1e-6);
}

#[test]
fn equality_constrained_scalars() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let x = problem.decision_variable();
    let y = problem.decision_variable();

    problem.maximize(x * y);
    subject_to!(problem, x + 3.0 * y == 36.0);

    assert_eq!(problem.cost_function_type(), ExpressionType::Quadratic);
    assert_eq!(problem.equality_constraint_type(), ExpressionType::Linear);

    problem.solve(Default::default()).unwrap();

    assert!((x.value() - 18.0).abs() < EPS);
    assert!((y.value() - 6.0).abs() < EPS);
}

#[test]
fn equality_constrained_matrix() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let mut x = problem.decision_variable_matrix(2, 1);
    x.set_value(&Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap());

    let cost = (&x.t() * &x).get(0, 0);
    problem.minimize(cost);

    let target = VariableMatrix::from_array_in(
        &arena,
        &Array2::from_shape_vec((2, 1), vec![3.0, 3.0]).unwrap(),
    );
    subject_to!(problem, &x == target);

    problem.solve(Default::default()).unwrap();

    assert!((x.value_at(0, 0) - 3.0).abs() < EPS);
    assert!((x.value_at(1, 0) - 3.0).abs() < EPS);
}

#[test]
fn inequality_constrained_2d() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let x = problem.decision_variable();
    let y = problem.decision_variable();
    x.set_value(5.0);
    y.set_value(5.0);

    problem.minimize(x * x + y * 2.0 * y);
    subject_to!(problem, y >= -x + 5.0);

    assert_eq!(problem.cost_function_type(), ExpressionType::Quadratic);
    assert_eq!(problem.equality_constraint_type(), ExpressionType::None);
    assert_eq!(problem.inequality_constraint_type(), ExpressionType::Linear);

    problem.solve(Default::default()).unwrap();

    assert!((x.value() - (3.0 + 1.0 / 3.0)).abs() < 1e-6);
    assert!((y.value() - (1.0 + 2.0 / 3.0)).abs() < 1e-6);
}
