//! Tests for the rayon-based multistart driver. Each worker thread
//! builds its own `VariableArena` inside the user closure — arenas
//! never cross thread boundaries.

use hafgufa::{
    ExitStatus, MultistartResult, Problem, SleipnirError, VariableArena, multistart,
    subject_to,
};

fn solve(guess: &f64) -> Result<MultistartResult<f64>, SleipnirError> {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = problem.decision_variable();
    x.set_value(*guess);

    // cost = (x - 4.0)²
    let err = x - 4.0;
    let cost = err * err;
    problem.minimize(cost);

    // Add a redundant inequality just to exercise the full pipeline.
    subject_to!(problem, x >= -10.0);

    match problem.solve(Default::default()) {
        Ok(()) => Ok(MultistartResult {
            status: ExitStatus::Success,
            cost: cost.value(),
            variables: x.value(),
        }),
        Err(e) => Err(e),
    }
}

#[test]
fn multistart_returns_best_solution() {
    let guesses = vec![-5.0, 0.0, 3.5, 100.0];
    let best = multistart(&guesses, solve).unwrap();

    assert_eq!(best.status, ExitStatus::Success);
    assert!(best.cost < 1e-6, "cost was {}", best.cost);
    assert!((best.variables - 4.0).abs() < 1e-3);
}

#[test]
fn multistart_with_single_guess_works() {
    let guesses = vec![0.0];
    let best = multistart(&guesses, solve).unwrap();
    assert_eq!(best.status, ExitStatus::Success);
}
