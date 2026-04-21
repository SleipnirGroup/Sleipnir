//! VariableMatrix construction, access, and arithmetic tests.

use hafgufa::{Problem, VariableArena, VariableMatrix, subject_to};
use ndarray::Array2;

#[test]
fn zeros_and_shape() {
    let arena = VariableArena::new();
    let m = VariableMatrix::zeros_in(&arena, 3, 4);
    assert_eq!(m.shape(), (3, 4));
    assert_eq!(m.rows(), 3);
    assert_eq!(m.cols(), 4);
}

#[test]
fn round_trip_through_ndarray() {
    let arena = VariableArena::new();
    let mut src = Array2::<f64>::zeros((2, 3));
    for ((i, j), cell) in src.indexed_iter_mut() {
        *cell = (i * 10 + j) as f64;
    }

    let mut m = VariableMatrix::from_array_in(&arena, &src);
    let out = m.value();
    assert_eq!(out, src);
}

#[test]
fn matmul_and_transpose() {
    // A * Aᵀ should be a 2×2 matrix whose (0,0) = 1²+2²+3² = 14.
    let arena = VariableArena::new();
    let a = VariableMatrix::from_array_in(
        &arena,
        &Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    );
    let product = &a * &a.t();
    let mut p = product;
    assert_eq!(p.value_at(0, 0), 14.0);
    assert_eq!(p.value_at(0, 1), 32.0);
    assert_eq!(p.value_at(1, 0), 32.0);
    assert_eq!(p.value_at(1, 1), 77.0);
}

#[test]
fn decision_variable_matrix_solves_quadratic() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = problem.decision_variable_matrix(2, 1);

    // Minimize ‖x‖² subject to x₀ + x₁ == 1.
    let cost = (&x.t() * &x).get(0, 0);
    problem.minimize(cost);

    let sum_row = VariableMatrix::from_array_in(
        &arena,
        &Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap(),
    );
    let expected =
        VariableMatrix::from_array_in(&arena, &Array2::from_elem((1, 1), 1.0));
    subject_to!(problem, &sum_row * &x == expected);

    problem.solve(Default::default()).unwrap();

    let mut x = x;
    assert!((x.value_at(0, 0) - 0.5).abs() < 1e-6);
    assert!((x.value_at(1, 0) - 0.5).abs() < 1e-6);
}

#[test]
fn block_row_col_slicing() {
    let arena = VariableArena::new();
    let m = VariableMatrix::from_array_in(
        &arena,
        &Array2::from_shape_vec(
            (3, 3),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap(),
    );

    let mut row_1 = m.row(1);
    assert_eq!(row_1.shape(), (1, 3));
    assert_eq!(row_1.value_at(0, 0), 4.0);
    assert_eq!(row_1.value_at(0, 2), 6.0);

    let mut col_2 = m.col(2);
    assert_eq!(col_2.shape(), (3, 1));
    assert_eq!(col_2.value_at(1, 0), 6.0);

    let mut block = m.block(1, 1, 2, 2);
    assert_eq!(block.shape(), (2, 2));
    assert_eq!(block.value_at(0, 0), 5.0);
    assert_eq!(block.value_at(1, 1), 9.0);
}
