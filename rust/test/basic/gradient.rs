//! Gradient evaluation test.

use hafgufa::{Gradient, VariableArena, VariableMatrix};

#[test]
fn gradient_of_quadratic() {
    // f(x) = x₀² + 2·x₁² + x₀·x₁
    // ∇f = [2·x₀ + x₁, 4·x₁ + x₀]
    let arena = VariableArena::new();
    let mut x = VariableMatrix::zeros_in(&arena, 2, 1);
    let x0 = x.get(0, 0);
    let x1 = x.get(1, 0);
    x.set_value(&ndarray::Array2::from_shape_vec((2, 1), vec![3.0, -1.0]).unwrap());

    let f = x0 * x0 + 2.0 * x1 * x1 + x0 * x1;

    let mut g = Gradient::new(f, &x);
    let values = g.value();

    assert!((values[0] - (2.0 * 3.0 + -1.0)).abs() < 1e-9);
    assert!((values[1] - (4.0 * -1.0 + 3.0)).abs() < 1e-9);
}
