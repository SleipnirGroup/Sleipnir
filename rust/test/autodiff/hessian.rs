//! Port of `test/src/autodiff/hessian_test.cpp`. Uses
//! `HessianTriangle::Full` throughout — matches the default C++ template
//! argument.

use hafgufa::math::{log, pow, sin};
use hafgufa::{
    Gradient, Hessian, HessianTriangle, Jacobian, Variable, VariableArena, VariableMatrix,
};
use ndarray::{Array2, arr2};

fn column_of<'a>(arena: &'a VariableArena, values: &[f64]) -> VariableMatrix<'a> {
    let mut m = VariableMatrix::zeros_in(arena, values.len() as i32, 1);
    m.set_value(&Array2::from_shape_vec((values.len(), 1), values.to_vec()).unwrap());
    m
}

#[test]
fn linear() {
    let arena = VariableArena::new();
    let x = column_of(&arena, &[3.0]);
    let x0 = x.get(0, 0);
    let y = x0;

    assert_eq!(Gradient::new(y, x0).value()[0], 1.0);

    let mut h = Hessian::new(y, &x, HessianTriangle::Full);
    assert_eq!(h.value()[[0, 0]], 0.0);
}

#[test]
fn quadratic() {
    let arena = VariableArena::new();
    let x = column_of(&arena, &[3.0]);
    let x0 = x.get(0, 0);
    let y = x0 * x0;

    assert_eq!(Gradient::new(y, x0).value()[0], 6.0);

    let mut h = Hessian::new(y, &x, HessianTriangle::Full);
    assert_eq!(h.value()[[0, 0]], 2.0);
}

#[test]
fn cubic() {
    let arena = VariableArena::new();
    let x = column_of(&arena, &[3.0]);
    let x0 = x.get(0, 0);
    let y = x0 * x0 * x0;

    assert_eq!(Gradient::new(y, x0).value()[0], 27.0);

    let mut h = Hessian::new(y, &x, HessianTriangle::Full);
    assert_eq!(h.value()[[0, 0]], 18.0);
}

#[test]
fn quartic() {
    let arena = VariableArena::new();
    let x = column_of(&arena, &[3.0]);
    let x0 = x.get(0, 0);
    let y = x0 * x0 * x0 * x0;

    assert_eq!(Gradient::new(y, x0).value()[0], 108.0);

    let mut h = Hessian::new(y, &x, HessianTriangle::Full);
    assert_eq!(h.value()[[0, 0]], 108.0);
}

#[test]
fn sum() {
    let arena = VariableArena::new();
    let x = column_of(&arena, &[1.0, 2.0, 3.0, 4.0, 5.0]);

    let y = x.get(0, 0) + x.get(1, 0) + x.get(2, 0) + x.get(3, 0) + x.get(4, 0);
    assert_eq!(y.value(), 15.0);

    let mut g = Gradient::new(y, &x);
    assert_eq!(g.value(), vec![1.0; 5]);

    let mut h = Hessian::new(y, &x, HessianTriangle::Full);
    let hv = h.value();
    for i in 0..5 {
        for j in 0..5 {
            assert_eq!(hv[[i, j]], 0.0);
        }
    }
}

#[test]
fn sum_of_products() {
    let arena = VariableArena::new();
    let x = column_of(&arena, &[1.0, 2.0, 3.0, 4.0, 5.0]);

    // y = xᵀx = ‖x‖²
    let y = (x.t() * &x).get(0, 0);
    assert_eq!(y.value(), 1.0 + 4.0 + 9.0 + 16.0 + 25.0);

    let mut g = Gradient::new(y, &x);
    let gv = g.value();
    for i in 0..5 {
        assert_eq!(gv[i], 2.0 * (i as f64 + 1.0));
    }

    let mut h = Hessian::new(y, &x, HessianTriangle::Full);
    let hv = h.value();
    for i in 0..5 {
        for j in 0..5 {
            let expected = if i == j { 2.0 } else { 0.0 };
            assert_eq!(hv[[i, j]], expected);
        }
    }
}

#[test]
fn product_of_sines() {
    const EPS: f64 = 1e-15;
    let arena = VariableArena::new();
    let x = column_of(&arena, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    let xvs: Vec<Variable<'_>> = (0..5).map(|i| x.get(i, 0)).collect();

    let mut y = Variable::constant_in(&arena, 1.0);
    for v in &xvs {
        y = y * sin(*v);
    }
    let expected_y: f64 = (1..=5).map(|i| (i as f64).sin()).product();
    assert!((y.value() - expected_y).abs() < EPS);

    let mut g = Gradient::new(y, &x);
    let gv = g.value();
    for i in 0..5 {
        let expected = y.value() / xvs[i].value().tan();
        assert!((gv[i] - expected).abs() < EPS);
    }

    let mut h = Hessian::new(y, &x, HessianTriangle::Full);
    let hv = h.value();
    for i in 0..5 {
        for j in 0..5 {
            let expected = if i == j {
                -y.value()
            } else {
                y.value() / (xvs[i].value().tan() * xvs[j].value().tan())
            };
            assert!((hv[[i, j]] - expected).abs() < EPS);
        }
    }
}

#[test]
fn sum_of_squared_residuals() {
    let arena = VariableArena::new();
    let x = column_of(&arena, &[1.0; 5]);
    let xvs: Vec<Variable<'_>> = (0..5).map(|i| x.get(i, 0)).collect();

    // y = Σᵢ (xᵢ − xᵢ₊₁)²
    let mut y = Variable::constant_in(&arena, 0.0);
    for i in 0..4 {
        let d = xvs[i] - xvs[i + 1];
        y = y + d * d;
    }
    assert_eq!(y.value(), 0.0);

    let mut g = Gradient::new(y, &x);
    let gv = g.value();
    assert_eq!(gv[0], 2.0 * xvs[0].value() - 2.0 * xvs[1].value());
    assert_eq!(
        gv[1],
        -2.0 * xvs[0].value() + 4.0 * xvs[1].value() - 2.0 * xvs[2].value(),
    );
    assert_eq!(
        gv[2],
        -2.0 * xvs[1].value() + 4.0 * xvs[2].value() - 2.0 * xvs[3].value(),
    );
    assert_eq!(
        gv[3],
        -2.0 * xvs[2].value() + 4.0 * xvs[3].value() - 2.0 * xvs[4].value(),
    );
    assert_eq!(gv[4], -2.0 * xvs[3].value() + 2.0 * xvs[4].value());

    let expected_h: Array2<f64> = arr2(&[
        [2.0, -2.0, 0.0, 0.0, 0.0],
        [-2.0, 4.0, -2.0, 0.0, 0.0],
        [0.0, -2.0, 4.0, -2.0, 0.0],
        [0.0, 0.0, -2.0, 4.0, -2.0],
        [0.0, 0.0, 0.0, -2.0, 2.0],
    ]);

    let mut h = Hessian::new(y, &x, HessianTriangle::Full);
    let hv = h.value();
    for i in 0..5 {
        for j in 0..5 {
            assert_eq!(hv[[i, j]], expected_h[[i, j]]);
        }
    }
}

#[test]
fn sum_of_squares() {
    let arena = VariableArena::new();
    let r = column_of(&arena, &[25.0, 10.0, 5.0, 0.0]);
    let x = column_of(&arena, &[0.0, 0.0, 0.0, 0.0]);

    let mut j = Variable::constant_in(&arena, 0.0);
    for i in 0..4 {
        let d = r.get(i, 0) - x.get(i, 0);
        j = j + d * d;
    }

    let mut h = Hessian::new(j, &x, HessianTriangle::Full);
    let hv = h.value();
    for i in 0..4 {
        for k in 0..4 {
            assert_eq!(hv[[i, k]], if i == k { 2.0 } else { 0.0 });
        }
    }
}

#[test]
fn nested_powers() {
    const EPS: f64 = 1e-12;
    let x0 = 3.0;
    let arena = VariableArena::new();
    let xmat = column_of(&arena, &[x0]);
    let x = xmat.get(0, 0);

    let y = pow(pow(x, 2.0), 2.0);

    let mut yvec = VariableMatrix::zeros_in(&arena, 1, 1);
    yvec.set_variable(0, 0, y);
    let mut j = Jacobian::new(&yvec, &xmat);
    assert!((j.value()[[0, 0]] - 4.0 * x0 * x0 * x0).abs() < EPS);

    let mut h = Hessian::new(y, &xmat, HessianTriangle::Full);
    assert!((h.value()[[0, 0]] - 12.0 * x0 * x0).abs() < EPS);
}

#[test]
fn rosenbrock() {
    const EPS: f64 = 1e-11;
    // z = (1 − x)² + 100(y − x²)²
    let arena = VariableArena::new();
    let input = VariableMatrix::zeros_in(&arena, 2, 1);
    let x = input.get(0, 0);
    let y = input.get(1, 0);
    let z = pow(1.0 - x, 2.0) + 100.0 * pow(y - pow(x, 2.0), 2.0);

    let mut hessian = Hessian::new(z, &input, HessianTriangle::Full);

    let mut x0 = -2.5;
    while x0 <= 2.5 + 1e-12 {
        let mut y0 = -2.5;
        while y0 <= 2.5 + 1e-12 {
            x.set_value(x0);
            y.set_value(y0);

            let h = hessian.value();
            assert!((h[[0, 0]] - (1200.0 * x0 * x0 - 400.0 * y0 + 2.0)).abs() < EPS);
            assert_eq!(h[[0, 1]], -400.0 * x0);
            assert_eq!(h[[1, 0]], -400.0 * x0);
            assert_eq!(h[[1, 1]], 200.0);

            y0 += 0.1;
        }
        x0 += 0.1;
    }
}

#[test]
fn edge_pushing_wang_example1() {
    let arena = VariableArena::new();
    let xvec = column_of(&arena, &[3.0, 4.0]);
    let x0 = xvec.get(0, 0);
    let x1 = xvec.get(1, 0);

    // y = (x₀ sin x₁) x₀
    let y = (x0 * sin(x1)) * x0;

    let mut yvec = VariableMatrix::zeros_in(&arena, 1, 1);
    yvec.set_variable(0, 0, y);
    let mut j = Jacobian::new(&yvec, &xvec);
    let jv = j.value();
    assert_eq!(jv[[0, 0]], 6.0 * 4.0_f64.sin());
    assert_eq!(jv[[0, 1]], 9.0 * 4.0_f64.cos());

    let mut h = Hessian::new(y, &xvec, HessianTriangle::Full);
    let hv = h.value();
    assert_eq!(hv[[0, 0]], 2.0 * 4.0_f64.sin());
    assert_eq!(hv[[0, 1]], 6.0 * 4.0_f64.cos());
    assert_eq!(hv[[1, 0]], 6.0 * 4.0_f64.cos());
    assert_eq!(hv[[1, 1]], -9.0 * 4.0_f64.sin());
}

#[test]
fn edge_pushing_petro_figure1() {
    let arena = VariableArena::new();
    let p1 = Variable::constant_in(&arena, 2.0);
    let xvec = column_of(&arena, &[2.0, 3.0]);
    let x0 = xvec.get(0, 0);
    let x1 = xvec.get(1, 0);
    let y = p1 * log(x0 * x1);

    let mut h = Hessian::new(y, &xvec, HessianTriangle::Full);
    let hv = h.value();
    assert_eq!(hv[[0, 0]], -p1.value() / (x0.value() * x0.value()));
    assert_eq!(hv[[0, 1]], 0.0);
    assert_eq!(hv[[1, 0]], 0.0);
    assert_eq!(hv[[1, 1]], -p1.value() / (x1.value() * x1.value()));
}

#[test]
fn variable_reuse() {
    let arena = VariableArena::new();
    let xvec = column_of(&arena, &[1.0]);
    let x = xvec.get(0, 0);

    let y = x * x * x;

    let mut hessian = Hessian::new(y, &xvec, HessianTriangle::Full);
    let h = hessian.value();
    assert_eq!(h.shape(), &[1, 1]);
    assert_eq!(h[[0, 0]], 6.0);

    x.set_value(2.0);
    let h = hessian.value();
    assert_eq!(h[[0, 0]], 12.0);
}
