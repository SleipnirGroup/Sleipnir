//! Port of `test/src/autodiff/variable_matrix_test.cpp`. The C++ Slice
//! DSL (`_`, start:stop:step), iterators, `cwise_transform`,
//! `cwise_reduce`, and the multi-block `block({{A,B},{C}})` stacker don't
//! have equivalents in the Rust bindings (Rust uses `RangeBounds`,
//! `rows_iter`/`cols_iter`, and `hstack`/`vstack`). Those sections are
//! skipped; the construction/block/value/solve behaviour is covered here
//! and the bulk of the matrix arithmetic lives in
//! `test/basic/variable_matrix.rs`.

use hafgufa::{VariableArena, VariableMatrix, hstack, solve, vstack};
use ndarray::{Array2, arr2};

#[test]
fn construct_from_array() {
    let arena = VariableArena::new();
    let src = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let mut mat = VariableMatrix::from_array_in(&arena, &src);
    assert_eq!(mat.value(), src);
}

#[test]
fn construct_from_diagonal_slice() {
    let arena = VariableArena::new();
    let mut mat = VariableMatrix::diag_in(&arena, &[1.0, 2.0, 3.0]);
    let expected = arr2(&[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]);
    assert_eq!(mat.value(), expected);
}

#[test]
fn zeros_fills_with_zero() {
    let arena = VariableArena::new();
    let mut mat = VariableMatrix::zeros_in(&arena, 2, 2);
    assert_eq!(mat.rows(), 2);
    assert_eq!(mat.cols(), 2);
    for r in 0..2 {
        for c in 0..2 {
            assert_eq!(mat.value_at(r, c), 0.0);
        }
    }

    mat.set_scalar(0, 0, 1.0);
    mat.set_scalar(0, 1, 2.0);
    mat.set_scalar(1, 0, 3.0);
    mat.set_scalar(1, 1, 4.0);
    assert_eq!(mat.value_at(0, 0), 1.0);
    assert_eq!(mat.value_at(0, 1), 2.0);
    assert_eq!(mat.value_at(1, 0), 3.0);
    assert_eq!(mat.value_at(1, 1), 4.0);
}

#[test]
fn block_access() {
    let arena = VariableArena::new();
    let mat = VariableMatrix::from_array_in(
        &arena,
        &arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
    );

    let mut blk = mat.block(1, 1, 2, 2);
    assert_eq!(blk.rows(), 2);
    assert_eq!(blk.cols(), 2);
    assert_eq!(blk.value_at(0, 0), 5.0);
    assert_eq!(blk.value_at(0, 1), 6.0);
    assert_eq!(blk.value_at(1, 0), 8.0);
    assert_eq!(blk.value_at(1, 1), 9.0);

    let mut inner = blk.block(1, 1, 1, 1);
    assert_eq!(inner.value_at(0, 0), 9.0);
}

#[test]
fn transpose_round_trip() {
    let arena = VariableArena::new();
    let src = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let mat = VariableMatrix::from_array_in(&arena, &src);
    let mut tt = mat.t().t();
    assert_eq!(tt.value(), src);
}

#[test]
fn vstack_and_hstack_basic() {
    let arena = VariableArena::new();
    let a = VariableMatrix::from_array_in(&arena, &arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
    let b = VariableMatrix::from_array_in(&arena, &arr2(&[[7.0], [8.0]]));

    let mut h = hstack(&a, &b);
    assert_eq!(h.rows(), 2);
    assert_eq!(h.cols(), 4);
    assert_eq!(
        h.value(),
        arr2(&[[1.0, 2.0, 3.0, 7.0], [4.0, 5.0, 6.0, 8.0]]),
    );

    let c = VariableMatrix::from_array_in(&arena, &arr2(&[[9.0, 10.0, 11.0, 12.0]]));
    let mut v = vstack(&h, &c);
    assert_eq!(v.rows(), 3);
    assert_eq!(v.cols(), 4);
    assert_eq!(
        v.value(),
        arr2(&[
            [1.0, 2.0, 3.0, 7.0],
            [4.0, 5.0, 6.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ]),
    );
}

fn check_solve(a_data: &Array2<f64>, b_data: &Array2<f64>) {
    let arena = VariableArena::new();
    let a = VariableMatrix::from_array_in(&arena, a_data);
    let b = VariableMatrix::from_array_in(&arena, b_data);

    let mut x = solve(&a, &b);
    assert_eq!(x.rows() as usize, a_data.ncols());
    assert_eq!(x.cols() as usize, b_data.ncols());

    let x_vals = x.value();
    let residual = a_data.dot(&x_vals) - b_data;
    let norm: f64 = residual.iter().map(|r| r * r).sum::<f64>().sqrt();
    assert!(norm < 1e-12, "‖Ax − b‖ = {norm}");
}

#[test]
fn solve_1x1() {
    check_solve(&arr2(&[[2.0]]), &arr2(&[[5.0]]));
}

#[test]
fn solve_2x2() {
    check_solve(&arr2(&[[1.0, 2.0], [3.0, 4.0]]), &arr2(&[[5.0], [6.0]]));
}

#[test]
fn solve_3x3() {
    check_solve(
        &arr2(&[[1.0, 2.0, 3.0], [-4.0, -5.0, 6.0], [7.0, 8.0, 9.0]]),
        &arr2(&[[10.0], [11.0], [12.0]]),
    );
}

#[test]
fn solve_4x4() {
    check_solve(
        &arr2(&[
            [1.0, 2.0, 3.0, -4.0],
            [-5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]),
        &arr2(&[[17.0], [18.0], [19.0], [20.0]]),
    );
}

#[test]
fn solve_5x5() {
    check_solve(
        &arr2(&[
            [1.0, 2.0, 3.0, -4.0, 5.0],
            [-5.0, 6.0, 7.0, 8.0, 9.0],
            [9.0, 10.0, 11.0, 12.0, 13.0],
            [13.0, 14.0, 15.0, 16.0, 17.0],
            [17.0, 18.0, 19.0, 20.0, 21.0],
        ]),
        &arr2(&[[21.0], [22.0], [23.0], [24.0], [25.0]]),
    );
}
