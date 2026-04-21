//! Port of `test/src/autodiff/jacobian_test.cpp`.

use hafgufa::{Jacobian, VariableArena, VariableMatrix};
use ndarray::{Array2, arr2};

fn column_of<'a>(arena: &'a VariableArena, values: &[f64]) -> VariableMatrix<'a> {
    let mut m = VariableMatrix::zeros_in(arena, values.len() as i32, 1);
    m.set_value(&Array2::from_shape_vec((values.len(), 1), values.to_vec()).unwrap());
    m
}

fn assert_matrix_eq(actual: &Array2<f64>, expected: &Array2<f64>) {
    assert_eq!(actual.shape(), expected.shape());
    for i in 0..actual.nrows() {
        for j in 0..actual.ncols() {
            assert_eq!(actual[[i, j]], expected[[i, j]], "mismatch at ({i},{j})");
        }
    }
}

#[test]
fn identity() {
    let arena = VariableArena::new();
    let x = column_of(&arena, &[1.0, 2.0, 3.0]);
    let mut j = Jacobian::new(&x, &x);
    let expected = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    assert_matrix_eq(&j.value(), &expected);
}

#[test]
fn scaled_identity() {
    let arena = VariableArena::new();
    let x = column_of(&arena, &[1.0, 2.0, 3.0]);
    let y = 3.0 * &x;
    let mut j = Jacobian::new(&y, &x);
    let expected = arr2(&[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]);
    assert_matrix_eq(&j.value(), &expected);
}

#[test]
fn products() {
    let arena = VariableArena::new();
    let x = column_of(&arena, &[1.0, 2.0, 3.0]);
    let (x0, x1, x2) = (x.get(0, 0), x.get(1, 0), x.get(2, 0));

    let mut y = VariableMatrix::zeros_in(&arena, 3, 1);
    y.set_variable(0, 0, x0 * x1);
    y.set_variable(1, 0, x1 * x2);
    y.set_variable(2, 0, x0 * x2);

    let mut j = Jacobian::new(&y, &x);
    let expected = arr2(&[[2.0, 1.0, 0.0], [0.0, 3.0, 2.0], [3.0, 0.0, 1.0]]);
    assert_matrix_eq(&j.value(), &expected);
}

#[test]
fn nested_products() {
    let arena = VariableArena::new();
    let x = column_of(&arena, &[3.0]);
    let x0 = x.get(0, 0);

    let mut y = VariableMatrix::zeros_in(&arena, 3, 1);
    y.set_variable(0, 0, 5.0 * x0);
    y.set_variable(1, 0, 7.0 * x0);
    y.set_variable(2, 0, 11.0 * x0);
    assert_eq!(y.value_at(0, 0), 15.0);
    assert_eq!(y.value_at(1, 0), 21.0);
    assert_eq!(y.value_at(2, 0), 33.0);

    let (y0, y1, y2) = (y.get(0, 0), y.get(1, 0), y.get(2, 0));
    let mut z = VariableMatrix::zeros_in(&arena, 3, 1);
    z.set_variable(0, 0, y0 * y1);
    z.set_variable(1, 0, y1 * y2);
    z.set_variable(2, 0, y0 * y2);
    assert_eq!(z.value_at(0, 0), 315.0);
    assert_eq!(z.value_at(1, 0), 693.0);
    assert_eq!(z.value_at(2, 0), 495.0);

    let mut j_yx = Jacobian::new(&y, &x);
    let v = j_yx.value();
    assert_eq!(v[[0, 0]], 5.0);
    assert_eq!(v[[1, 0]], 7.0);
    assert_eq!(v[[2, 0]], 11.0);

    let mut j_zy = Jacobian::new(&z, &y);
    let expected = arr2(&[[21.0, 15.0, 0.0], [0.0, 33.0, 21.0], [33.0, 0.0, 15.0]]);
    assert_matrix_eq(&j_zy.value(), &expected);

    let mut j_zx = Jacobian::new(&z, &x);
    let v = j_zx.value();
    assert_eq!(v[[0, 0]], 210.0);
    assert_eq!(v[[1, 0]], 462.0);
    assert_eq!(v[[2, 0]], 330.0);
}

#[test]
fn non_square() {
    let arena = VariableArena::new();
    let x = column_of(&arena, &[1.0, 2.0, 3.0]);
    let (x0, x1, x2) = (x.get(0, 0), x.get(1, 0), x.get(2, 0));

    let mut y = VariableMatrix::zeros_in(&arena, 1, 1);
    y.set_variable(0, 0, x0 + 3.0 * x1 - 5.0 * x2);

    let mut j = Jacobian::new(&y, &x);
    let v = j.value();
    assert_eq!(v.shape(), &[1, 3]);
    assert_eq!(v[[0, 0]], 1.0);
    assert_eq!(v[[0, 1]], 3.0);
    assert_eq!(v[[0, 2]], -5.0);
}

#[test]
fn variable_reuse() {
    let arena = VariableArena::new();
    let x = column_of(&arena, &[1.0, 2.0]);
    let (x0, x1) = (x.get(0, 0), x.get(1, 0));

    let mut y = VariableMatrix::zeros_in(&arena, 1, 1);
    y.set_variable(0, 0, x0 * x1);

    let mut jacobian = Jacobian::new(&y, &x);
    let j = jacobian.value();
    assert_eq!(j.shape(), &[1, 2]);
    assert_eq!(j[[0, 0]], 2.0);
    assert_eq!(j[[0, 1]], 1.0);

    x0.set_value(2.0);
    x1.set_value(1.0);
    let j = jacobian.value();
    assert_eq!(j[[0, 0]], 1.0);
    assert_eq!(j[[0, 1]], 2.0);
}
