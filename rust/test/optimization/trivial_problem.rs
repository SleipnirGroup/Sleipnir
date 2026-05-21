//! Port of `test/src/optimization/trivial_problem_test.cpp`.

use hafgufa::{ExpressionType, Problem, VariableArena};
use ndarray::Array2;

#[test]
fn empty_problem() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    assert_eq!(problem.cost_function_type(), ExpressionType::None);
    assert_eq!(problem.equality_constraint_type(), ExpressionType::None);
    assert_eq!(problem.inequality_constraint_type(), ExpressionType::None);

    problem.solve(Default::default()).unwrap();
}

#[test]
fn no_cost_unconstrained_default_values() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let x = problem.decision_variable_matrix(2, 3);
    assert_eq!(problem.cost_function_type(), ExpressionType::None);
    problem.solve(Default::default()).unwrap();

    let mut x = x;
    for row in 0..2 {
        for col in 0..3 {
            assert_eq!(x.value_at(row, col), 0.0);
        }
    }
}

#[test]
fn no_cost_unconstrained_custom_values() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let mut x = problem.decision_variable_matrix(2, 3);
    x.set_value(&Array2::from_elem((2, 3), 1.0));

    problem.solve(Default::default()).unwrap();

    for row in 0..2 {
        for col in 0..3 {
            assert_eq!(x.value_at(row, col), 1.0);
        }
    }
}
