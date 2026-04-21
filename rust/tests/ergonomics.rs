//! Smoke tests for the ergonomics pass: `Problem::bound`, `Options`
//! builder, `Gradient::new` with a single-Variable `wrt`,
//! `VariableMatrix::identity_in` / `::diag_in`, column/row iterators,
//! `Problem::decision_variables`, and `VariableMatrix::slice`.

use hafgufa::{Gradient, Options, Problem, VariableArena, VariableMatrix};
use ndarray::Array2;
use std::time::Duration;

#[test]
fn problem_bound_replaces_two_subject_to_calls() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = problem.decision_variable();
    problem.minimize(x * x);
    problem.bound(-1.0, x, 3.0);
    problem.solve(Default::default()).unwrap();
    assert!(x.value().abs() < 1e-6);
}

#[test]
fn options_builder_chains() {
    let opts = Options::default()
        .tolerance(1e-6)
        .max_iterations(1000)
        .timeout(Duration::from_millis(500))
        .feasible_ipm(true)
        .diagnostics(false);
    assert!((opts.tolerance - 1e-6).abs() < f64::EPSILON);
    assert_eq!(opts.max_iterations, 1000);
    assert_eq!(opts.timeout, Some(Duration::from_millis(500)));
    assert!(opts.feasible_ipm);
    assert!(!opts.diagnostics);
}

#[test]
fn gradient_accepts_single_variable_wrt() {
    // f(x) = 3x² + 5x → ∂f/∂x = 6x + 5. Use a real decision-variable
    // expression (via VariableMatrix::zeros_in) so the gradient isn't
    // trivially zero from constant folding.
    let arena = VariableArena::new();
    let mut xs = VariableMatrix::zeros_in(&arena, 1, 1);
    let x = xs.get(0, 0);
    xs.set_value(&Array2::from_shape_vec((1, 1), vec![4.0]).unwrap());
    let f = 3.0 * x * x + 5.0 * x;
    let mut g = Gradient::new(f, x);
    let values = g.value();
    assert_eq!(values.len(), 1);
    assert!(
        (values[0] - (6.0 * 4.0 + 5.0)).abs() < 1e-9,
        "got {}",
        values[0]
    );
}

#[test]
fn variable_matrix_identity_and_diag() {
    let arena = VariableArena::new();

    let mut i3 = VariableMatrix::identity_in(&arena, 3);
    let expected_eye = Array2::<f64>::eye(3);
    assert_eq!(i3.value(), expected_eye);

    let mut d = VariableMatrix::diag_in(&arena, &[2.0, -1.0, 0.5]);
    let expected_diag = Array2::from_shape_vec(
        (3, 3),
        vec![2.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.5],
    )
    .unwrap();
    assert_eq!(d.value(), expected_diag);
}

#[test]
fn cols_iter_and_rows_iter() {
    let arena = VariableArena::new();
    let m = VariableMatrix::from_array_in(
        &arena,
        &Array2::from_shape_vec(
            (2, 3),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap(),
    );

    let cols: Vec<_> = m.cols_iter().collect();
    assert_eq!(cols.len(), 3);
    for c in &cols {
        assert_eq!(c.shape(), (2, 1));
    }
    let mut first = cols[0].clone();
    assert_eq!(first.value_at(0, 0), 1.0);
    assert_eq!(first.value_at(1, 0), 4.0);

    let rows: Vec<_> = m.rows_iter().collect();
    assert_eq!(rows.len(), 2);
    for r in &rows {
        assert_eq!(r.shape(), (1, 3));
    }
}

#[test]
fn decision_variables_returns_vec() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let xs = problem.decision_variables(4);
    assert_eq!(xs.len(), 4);

    let cost = xs
        .iter()
        .copied()
        .map(|v| v * v)
        .reduce(|a, b| a + b)
        .unwrap();
    problem.minimize(cost);
    for x in &xs {
        problem.subject_to(hafgufa::cmp!(*x >= 1.0));
    }
    problem.solve(Default::default()).unwrap();
    for x in &xs {
        assert!((x.value() - 1.0).abs() < 1e-6);
    }
}

#[test]
fn variable_matrix_range_slicing() {
    let arena = VariableArena::new();
    let m = VariableMatrix::from_array_in(
        &arena,
        &Array2::from_shape_vec(
            (4, 4),
            (0..16).map(|i| i as f64).collect(),
        )
        .unwrap(),
    );

    let mut middle = m.slice(1..3, 1..3);
    assert_eq!(middle.shape(), (2, 2));
    assert_eq!(middle.value_at(0, 0), 5.0);
    assert_eq!(middle.value_at(1, 1), 10.0);

    let mut full_col = m.slice(.., 2..=2);
    assert_eq!(full_col.shape(), (4, 1));
    assert_eq!(full_col.value_at(0, 0), 2.0);
    assert_eq!(full_col.value_at(3, 0), 14.0);

    let mut tail = m.slice(2.., ..);
    assert_eq!(tail.shape(), (2, 4));
    assert_eq!(tail.value_at(0, 0), 8.0);
}

#[test]
fn debug_impls_compile_and_include_shape_info() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let _x = problem.decision_variable();
    let dbg = format!("{:?}", problem);
    assert!(dbg.contains("Problem"));
    assert!(dbg.contains("arena_len"));

    let m = VariableMatrix::zeros_in(&arena, 4, 7);
    let m_dbg = format!("{:?}", m);
    assert!(m_dbg.contains("rows: 4"));
    assert!(m_dbg.contains("cols: 7"));
}
