//! Port of `test/src/optimization/linear_problem_test.cpp`.

use hafgufa::{ExpressionType, Problem, VariableArena, subject_to};

const EPS: f64 = 1e-6;

#[test]
fn maximize() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let x = problem.decision_variable();
    let y = problem.decision_variable();
    x.set_value(1.0);
    y.set_value(1.0);

    problem.maximize(50.0 * x + 40.0 * y);

    subject_to!(problem, x + 1.5 * y <= 750.0);
    subject_to!(problem, 2.0 * x + 3.0 * y <= 1500.0);
    subject_to!(problem, 2.0 * x + y <= 1000.0);
    subject_to!(problem, x >= 0.0);
    subject_to!(problem, y >= 0.0);

    assert_eq!(problem.cost_function_type(), ExpressionType::Linear);
    assert_eq!(problem.equality_constraint_type(), ExpressionType::None);
    assert_eq!(problem.inequality_constraint_type(), ExpressionType::Linear);

    problem.solve(Default::default()).unwrap();

    assert!((x.value() - 375.0).abs() < EPS);
    assert!((y.value() - 250.0).abs() < EPS);
}

#[test]
fn free_variable() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let vs = problem.decision_variables(2);
    let (x0, x1) = (vs[0], vs[1]);
    x0.set_value(1.0);
    x1.set_value(2.0);

    subject_to!(problem, x0 == 0.0);

    assert_eq!(problem.cost_function_type(), ExpressionType::None);
    assert_eq!(problem.equality_constraint_type(), ExpressionType::Linear);
    assert_eq!(problem.inequality_constraint_type(), ExpressionType::None);

    problem.solve(Default::default()).unwrap();

    assert!(x0.value().abs() < EPS);
    assert!((x1.value() - 2.0).abs() < EPS);
}
