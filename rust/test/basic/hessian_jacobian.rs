//! Tests for Hessian, Jacobian, and the 3-arg hypot math helper.

use hafgufa::math;
use hafgufa::{Hessian, HessianTriangle, Jacobian, Variable, VariableArena, VariableMatrix};
use ndarray::Array2;

#[test]
fn hessian_of_quadratic_form() {
    // f(x) = x₀² + 2·x₁² + x₀·x₁
    // ∇²f = [[2, 1], [1, 4]]
    let arena = VariableArena::new();
    let mut x = VariableMatrix::zeros_in(&arena, 2, 1);
    let x0 = x.get(0, 0);
    let x1 = x.get(1, 0);
    x.set_value(&Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap());

    let f = x0 * x0 + 2.0 * x1 * x1 + x0 * x1;

    let mut h = Hessian::new(f, &x, HessianTriangle::Full);
    let values = h.value();

    assert_eq!(h.rows(), 2);
    assert_eq!(h.cols(), 2);
    assert!((values[[0, 0]] - 2.0).abs() < 1e-9);
    assert!((values[[0, 1]] - 1.0).abs() < 1e-9);
    assert!((values[[1, 0]] - 1.0).abs() < 1e-9);
    assert!((values[[1, 1]] - 4.0).abs() < 1e-9);
}

#[test]
fn jacobian_of_vector_function() {
    // f(x, y) = [x² + y, x·y]
    // Jf     = [[2x, 1], [y, x]]
    let arena = VariableArena::new();
    let mut wrt = VariableMatrix::zeros_in(&arena, 2, 1);
    let x = wrt.get(0, 0);
    let y = wrt.get(1, 0);
    wrt.set_value(&Array2::from_shape_vec((2, 1), vec![3.0, 5.0]).unwrap());

    let mut f = VariableMatrix::zeros_in(&arena, 2, 1);
    f.set_variable(0, 0, x * x + y);
    f.set_variable(1, 0, x * y);

    let mut j = Jacobian::new(&f, &wrt);
    let values = j.value();

    assert_eq!(j.rows(), 2);
    assert_eq!(j.cols(), 2);
    assert!((values[[0, 0]] - 6.0).abs() < 1e-9); // 2*3
    assert!((values[[0, 1]] - 1.0).abs() < 1e-9);
    assert!((values[[1, 0]] - 5.0).abs() < 1e-9); // y
    assert!((values[[1, 1]] - 3.0).abs() < 1e-9); // x
}

#[test]
fn hypot3_three_four_twelve() {
    let arena = VariableArena::new();
    let v = math::hypot3(
        Variable::constant_in(&arena, 3.0),
        Variable::constant_in(&arena, 4.0),
        Variable::constant_in(&arena, 12.0),
    );
    assert!((v.value() - 13.0).abs() < 1e-9);
}
