//! Port of `test/src/optimization/nonlinear_problem_test.cpp`.

use hafgufa::math::{hypot, pow, sqrt};
use hafgufa::{ExitStatus, ExpressionType, Problem, SleipnirError, VariableArena, subject_to};

#[test]
fn quartic() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let x = problem.decision_variable();
    x.set_value(20.0);

    problem.minimize(pow(x, 4.0));
    subject_to!(problem, x >= 1.0);

    assert_eq!(problem.cost_function_type(), ExpressionType::Nonlinear);
    assert_eq!(problem.inequality_constraint_type(), ExpressionType::Linear);

    problem.solve(Default::default()).unwrap();
    assert!((x.value() - 1.0).abs() < 1e-6);
}

#[test]
fn rosenbrock_cubic_line_constraint() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let x = problem.decision_variable();
    let y = problem.decision_variable();

    problem.minimize(100.0 * pow(y - pow(x, 2.0), 2.0) + pow(1.0 - x, 2.0));
    subject_to!(problem, y >= pow(x - 1.0, 3.0) + 1.0);
    subject_to!(problem, y <= -x + 2.0);

    assert_eq!(problem.cost_function_type(), ExpressionType::Nonlinear);
    assert_eq!(
        problem.inequality_constraint_type(),
        ExpressionType::Nonlinear
    );

    let mut x0 = -1.5;
    while x0 <= 1.5 + 1e-9 {
        let mut y0 = -0.5;
        while y0 <= 2.5 + 1e-9 {
            x.set_value(x0);
            y.set_value(y0);
            problem.solve(Default::default()).unwrap();

            let hits_x = (x.value() - 0.0).abs() < 1e-2 || (x.value() - 1.0).abs() < 1e-2;
            let hits_y = (y.value() - 0.0).abs() < 1e-2 || (y.value() - 1.0).abs() < 1e-2;
            assert!(hits_x, "x={} from start=({x0},{y0})", x.value());
            assert!(hits_y, "y={} from start=({x0},{y0})", y.value());

            y0 += 0.1;
        }
        x0 += 0.1;
    }
}

#[test]
fn rosenbrock_disk_constraint() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let x = problem.decision_variable();
    let y = problem.decision_variable();

    problem.minimize(pow(1.0 - x, 2.0) + 100.0 * pow(y - pow(x, 2.0), 2.0));
    subject_to!(problem, pow(x, 2.0) + pow(y, 2.0) <= 2.0);

    assert_eq!(problem.cost_function_type(), ExpressionType::Nonlinear);
    assert_eq!(
        problem.inequality_constraint_type(),
        ExpressionType::Quadratic
    );

    let mut x0 = -1.5;
    while x0 <= 1.5 + 1e-9 {
        let mut y0 = -1.5;
        while y0 <= 1.5 + 1e-9 {
            x.set_value(x0);
            y.set_value(y0);
            problem.solve(Default::default()).unwrap();

            assert!((x.value() - 1.0).abs() < 1e-3);
            assert!((y.value() - 1.0).abs() < 1e-3);

            y0 += 0.1;
        }
        x0 += 0.1;
    }
}

#[test]
fn minimum_2d_distance_with_linear_constraint() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let x = problem.decision_variable();
    let y = problem.decision_variable();
    x.set_value(20.0);
    y.set_value(50.0);

    problem.minimize(sqrt(x * x + y * y));
    subject_to!(problem, y == -x + 5.0);

    assert_eq!(problem.cost_function_type(), ExpressionType::Nonlinear);
    assert_eq!(problem.equality_constraint_type(), ExpressionType::Linear);

    problem.solve(Default::default()).unwrap();
    assert!((x.value() - 2.5).abs() < 1e-2);
    assert!((y.value() - 2.5).abs() < 1e-2);
}

#[test]
fn conflicting_bounds() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let x = problem.decision_variable();
    let y = problem.decision_variable();

    problem.minimize(hypot(x, y));
    subject_to!(problem, hypot(x, y) <= 1.0);
    problem.bound(0.5, x, -0.5);

    assert_eq!(problem.cost_function_type(), ExpressionType::Nonlinear);
    assert_eq!(
        problem.inequality_constraint_type(),
        ExpressionType::Nonlinear
    );

    let status = problem.solve_status(Default::default());
    assert_eq!(status, ExitStatus::GloballyInfeasible);
    assert_eq!(
        problem.solve(Default::default()),
        Err(SleipnirError::GloballyInfeasible),
    );
}

#[test]
fn wachter_and_biegler_line_search_failure() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let x = problem.decision_variable();
    let s1 = problem.decision_variable();
    let s2 = problem.decision_variable();
    x.set_value(-2.0);
    s1.set_value(3.0);
    s2.set_value(1.0);

    problem.minimize(x);

    subject_to!(problem, pow(x, 2.0) - s1 - 1.0 == 0.0);
    subject_to!(problem, x - s2 - 0.5 == 0.0);
    subject_to!(problem, s1 >= 0.0);
    subject_to!(problem, s2 >= 0.0);

    assert_eq!(problem.cost_function_type(), ExpressionType::Linear);
    assert_eq!(
        problem.equality_constraint_type(),
        ExpressionType::Quadratic
    );
    assert_eq!(problem.inequality_constraint_type(), ExpressionType::Linear);

    problem.solve(Default::default()).unwrap();
    assert!((x.value() - 1.0).abs() < 1e-6);
    assert!(s1.value().abs() < 1e-6);
    assert!((s2.value() - 0.5).abs() < 1e-6);
}
