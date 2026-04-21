//! Tests for `hstack` / `vstack` VariableMatrix concatenation.

use hafgufa::{VariableArena, VariableMatrix, hstack, vstack};
use ndarray::Array2;

#[test]
fn vstack_concatenates_rows() {
    let arena = VariableArena::new();
    let a = VariableMatrix::from_array_in(
        &arena,
        &Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap(),
    );
    let b = VariableMatrix::from_array_in(
        &arena,
        &Array2::from_shape_vec((2, 3), vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap(),
    );

    let mut stacked = vstack(&a, &b);
    assert_eq!(stacked.shape(), (3, 3));
    assert_eq!(
        stacked.value(),
        Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap()
    );
}

#[test]
fn hstack_concatenates_cols() {
    let arena = VariableArena::new();
    let a = VariableMatrix::from_array_in(
        &arena,
        &Array2::from_shape_vec((2, 1), vec![1.0, 4.0]).unwrap(),
    );
    let b = VariableMatrix::from_array_in(
        &arena,
        &Array2::from_shape_vec((2, 2), vec![2.0, 3.0, 5.0, 6.0]).unwrap(),
    );

    let mut stacked = hstack(&a, &b);
    assert_eq!(stacked.shape(), (2, 3));
    assert_eq!(
        stacked.value(),
        Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()
    );
}
