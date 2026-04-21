//! Iteration-callback tests.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use hafgufa::{Problem, SleipnirError, VariableArena, math, subject_to};

fn build_nonlinear_problem<'a>(arena: &'a VariableArena) -> Problem<'a> {
    let mut problem = Problem::new(arena);
    let x = problem.decision_variable();
    let y = problem.decision_variable();

    let cost = math::pow(x - 2.0, 4.0) + math::pow(y - 3.0, 4.0);
    problem.minimize(cost);
    subject_to!(problem, x + y >= 1.0);

    problem
}

#[test]
fn callback_fires_at_least_once() {
    let arena = VariableArena::new();
    let mut problem = build_nonlinear_problem(&arena);
    let count = Arc::new(AtomicUsize::new(0));
    let count_cb = Arc::clone(&count);

    problem.add_callback(move |_info| {
        count_cb.fetch_add(1, Ordering::SeqCst);
        false
    });

    problem.solve(Default::default()).unwrap();
    assert!(
        count.load(Ordering::SeqCst) >= 1,
        "expected at least one callback invocation"
    );
}

#[test]
fn callback_observes_iteration_state() {
    let arena = VariableArena::new();
    let mut problem = build_nonlinear_problem(&arena);
    let observed = Arc::new(Mutex::new(Vec::<(i32, usize)>::new()));
    let observed_cb = Arc::clone(&observed);

    problem.add_callback(move |info| {
        observed_cb
            .lock()
            .unwrap()
            .push((info.iteration(), info.x().len()));
        false
    });

    let _ = problem.solve(Default::default());

    let log = observed.lock().unwrap();
    assert!(!log.is_empty());
    for (_, x_len) in log.iter() {
        assert_eq!(*x_len, 2);
    }
}

#[test]
fn returning_true_requests_early_exit() {
    let arena = VariableArena::new();
    let mut problem = build_nonlinear_problem(&arena);

    problem.add_callback(|_info| true);

    let err = problem.solve(Default::default()).unwrap_err();
    assert_eq!(err, SleipnirError::CallbackRequestedStop);
}

#[test]
fn clear_callbacks_removes_them() {
    let arena = VariableArena::new();
    let mut problem = build_nonlinear_problem(&arena);
    let count = Arc::new(AtomicUsize::new(0));
    let count_cb = Arc::clone(&count);

    problem.add_callback(move |_info| {
        count_cb.fetch_add(1, Ordering::SeqCst);
        true
    });
    problem.clear_callbacks();

    problem.solve(Default::default()).unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 0);
}
