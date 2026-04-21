//! Port of `test/src/optimization/multistart_test.cpp` — Mishra's Bird
//! function. The C++ test uses a decision-variables struct; Rust uses
//! a `(f64, f64)` tuple payload for the same purpose.

use hafgufa::math::{cos, exp, pow, sin};
use hafgufa::{
    ExitStatus, MultistartResult, Problem, SleipnirError, VariableArena, multistart, subject_to,
};

fn solve_mishra(guess: &(f64, f64)) -> Result<MultistartResult<(f64, f64)>, SleipnirError> {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = problem.decision_variable();
    let y = problem.decision_variable();
    x.set_value(guess.0);
    y.set_value(guess.1);

    let j = sin(y) * exp(pow(1.0 - cos(x), 2.0))
        + cos(x) * exp(pow(1.0 - sin(y), 2.0))
        + pow(x - y, 2.0);
    problem.minimize(j);

    subject_to!(problem, pow(x + 5.0, 2.0) + pow(y + 5.0, 2.0) < 25.0);

    match problem.solve(Default::default()) {
        Ok(()) => Ok(MultistartResult {
            status: ExitStatus::Success,
            cost: j.value(),
            variables: (x.value(), y.value()),
        }),
        Err(e) => Err(e),
    }
}

#[test]
fn mishras_bird_function() {
    let guesses = vec![(-3.0_f64, -8.0), (-3.0, -1.5)];
    let best = multistart(&guesses, solve_mishra).unwrap();

    assert_eq!(best.status, ExitStatus::Success);
    let (x_val, y_val) = best.variables;
    assert!((x_val - -3.13024680).abs() < 1e-8);
    assert!((y_val - -1.58214218).abs() < 1e-8);
}
