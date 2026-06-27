//! End-to-end DSL test: minimize `x² + y²` subject to `x + y == 1` and
//! `x >= 0`. Expected optimum `(0.5, 0.5)`.

use hafgufa::{Problem, VariableArena, subject_to};

#[test]
fn solves_equality_constrained_quadratic() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = problem.decision_variable();
    let y = problem.decision_variable();

    problem.minimize(x * x + y * y);
    subject_to!(problem, x + y == 1.0);
    subject_to!(problem, x >= 0.0);

    problem.solve(Default::default()).unwrap();
    assert!((x.value() - 0.5).abs() < 1e-6);
    assert!((y.value() - 0.5).abs() < 1e-6);
}
