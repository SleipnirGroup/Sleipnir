//! Port of `test/src/optimization/solver/exit_status_test.cpp`. The
//! C++ void-returning `add_callback` overload (auto-converted to a
//! no-stop callback) has no Rust equivalent — the Rust callback type is
//! `FnMut(&IterationInfo) -> bool`, so every callback must return a
//! bool. The "returns void" sub-test collapses into the "returns false"
//! sub-test.

use std::time::Duration;

use hafgufa::math::sqrt;
use hafgufa::{ExitStatus, ExpressionType, Options, Problem, VariableArena, subject_to};

#[test]
fn callback_requested_stop() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = problem.decision_variable();
    problem.minimize(x * x);

    // Callback that never stops
    problem.add_callback(|_| false);
    x.set_value(1.0);
    assert_eq!(
        problem.solve_status(Default::default()),
        ExitStatus::Success
    );

    // Add another callback that returns true — expect stop on first iteration.
    problem.add_callback(|_| true);
    x.set_value(1.0);
    assert_eq!(
        problem.solve_status(Default::default()),
        ExitStatus::CallbackRequestedStop,
    );

    // Clearing callbacks removes the stop-requesting one.
    problem.clear_callbacks();
    problem.add_callback(|_| false);
    x.set_value(1.0);
    assert_eq!(
        problem.solve_status(Default::default()),
        ExitStatus::Success
    );

    // Persistent callbacks survive clear_callbacks().
    problem.add_persistent_callback(|_| true);
    problem.clear_callbacks();
    x.set_value(1.0);
    assert_eq!(
        problem.solve_status(Default::default()),
        ExitStatus::CallbackRequestedStop,
    );
}

#[test]
fn too_few_dofs() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let x = problem.decision_variable();
    let y = problem.decision_variable();
    let z = problem.decision_variable();

    subject_to!(problem, x == 1.0);
    subject_to!(problem, x == 2.0);
    subject_to!(problem, y == 1.0);
    subject_to!(problem, z == 1.0);

    assert_eq!(problem.cost_function_type(), ExpressionType::None);
    assert_eq!(problem.equality_constraint_type(), ExpressionType::Linear);
    assert_eq!(problem.inequality_constraint_type(), ExpressionType::None);

    assert_eq!(
        problem.solve_status(Default::default()),
        ExitStatus::TooFewDofs
    );
}

#[test]
fn locally_infeasible_equality() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let x = problem.decision_variable();
    let y = problem.decision_variable();
    let z = problem.decision_variable();

    subject_to!(problem, x == y + 1.0);
    subject_to!(problem, y == z + 1.0);
    subject_to!(problem, z == x + 1.0);

    assert_eq!(problem.equality_constraint_type(), ExpressionType::Linear);
    assert_eq!(
        problem.solve_status(Default::default()),
        ExitStatus::LocallyInfeasible,
    );
}

#[test]
fn locally_infeasible_inequality() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let x = problem.decision_variable();
    let y = problem.decision_variable();
    let z = problem.decision_variable();

    subject_to!(problem, x >= y + 1.0);
    subject_to!(problem, y >= z + 1.0);
    subject_to!(problem, z >= x + 1.0);

    assert_eq!(problem.inequality_constraint_type(), ExpressionType::Linear);
    assert_eq!(
        problem.solve_status(Default::default()),
        ExitStatus::LocallyInfeasible,
    );
}

#[test]
fn nonfinite_initial_guess_from_cost() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = problem.decision_variable();
    problem.minimize(1.0 / x);
    assert_eq!(
        problem.solve_status(Default::default()),
        ExitStatus::NonfiniteInitialGuess,
    );
}

#[test]
fn nonfinite_initial_guess_from_gradient() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = problem.decision_variable();
    problem.minimize(sqrt(x));
    assert_eq!(
        problem.solve_status(Default::default()),
        ExitStatus::NonfiniteInitialGuess,
    );
}

#[test]
fn nonfinite_initial_guess_from_equality_constraint() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = problem.decision_variable();
    subject_to!(problem, 1.0 / x == 1.0);
    assert_eq!(
        problem.solve_status(Default::default()),
        ExitStatus::NonfiniteInitialGuess,
    );
}

#[test]
fn nonfinite_initial_guess_from_equality_jacobian() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = problem.decision_variable();
    subject_to!(problem, sqrt(x) == 1.0);
    assert_eq!(
        problem.solve_status(Default::default()),
        ExitStatus::NonfiniteInitialGuess,
    );
}

#[test]
fn nonfinite_initial_guess_from_inequality_constraint() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = problem.decision_variable();
    subject_to!(problem, 1.0 / x > 1.0);
    assert_eq!(
        problem.solve_status(Default::default()),
        ExitStatus::NonfiniteInitialGuess,
    );
}

#[test]
fn nonfinite_initial_guess_from_inequality_jacobian() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = problem.decision_variable();
    subject_to!(problem, sqrt(x) > 1.0);
    assert_eq!(
        problem.solve_status(Default::default()),
        ExitStatus::NonfiniteInitialGuess,
    );
}

#[test]
fn diverging_iterates() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = problem.decision_variable();
    problem.minimize(x);

    assert_eq!(problem.cost_function_type(), ExpressionType::Linear);
    assert_eq!(
        problem.solve_status(Default::default()),
        ExitStatus::DivergingIterates,
    );
}

#[test]
fn max_iterations_exceeded() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = problem.decision_variable();
    x.set_value(1.0);
    problem.minimize(x * x);

    let status = problem.solve_status(Options::default().max_iterations(0));
    assert_eq!(status, ExitStatus::MaxIterationsExceeded);
}

#[test]
fn timeout() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = problem.decision_variable();
    x.set_value(1.0);
    problem.minimize(x * x);

    let status = problem.solve_status(Options::default().timeout(Duration::from_secs(0)));
    assert_eq!(status, ExitStatus::Timeout);
}
