//! Port of `test/src/autodiff/gradient_test.cpp`. Drops the C++
//! `scope_exit` pool-leak assertions (Rust drops the arena
//! deterministically) and the f64-comparison-only tests (they add no
//! gradient coverage).

use hafgufa::math::{
    abs, acos, asin, atan, atan2, cbrt, cos, cosh, erf, exp, hypot, hypot3, log, log10, max, min,
    pow, sign, sin, sinh, sqrt, tan, tanh,
};
use hafgufa::{Gradient, Problem, Variable, VariableArena, VariableMatrix};

const EPS: f64 = 1e-15;

fn free_vars<'a>(problem: &mut Problem<'a>, values: &[f64]) -> Vec<Variable<'a>> {
    let vars = problem.decision_variables(values.len());
    for (v, &val) in vars.iter().zip(values) {
        v.set_value(val);
    }
    vars
}

#[test]
fn trivial_case() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let vs = free_vars(&mut problem, &[10.0, 20.0]);
    let (a, b) = (vs[0], vs[1]);
    let c = a;

    assert_eq!(Gradient::new(a, a).value()[0], 1.0);
    assert_eq!(Gradient::new(a, b).value()[0], 0.0);
    assert_eq!(Gradient::new(c, a).value()[0], 1.0);
    assert_eq!(Gradient::new(c, b).value()[0], 0.0);
}

#[test]
fn unary_minus() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let a = free_vars(&mut problem, &[10.0])[0];
    let c = -a;

    assert_eq!(c.value(), -a.value());
    assert_eq!(Gradient::new(c, a).value()[0], -1.0);
}

#[test]
fn identical_variables() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let a = free_vars(&mut problem, &[10.0])[0];
    let x = a;
    let c = a * a + x;

    assert_eq!(c.value(), a.value() * a.value() + x.value());
    assert_eq!(
        Gradient::new(c, a).value()[0],
        2.0 * a.value() + Gradient::new(x, a).value()[0],
    );
    assert_eq!(
        Gradient::new(c, x).value()[0],
        2.0 * a.value() * Gradient::new(a, x).value()[0] + 1.0,
    );
}

#[test]
fn elementary() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let vs = free_vars(&mut problem, &[1.0, 2.0]);
    let (a, b) = (vs[0], vs[1]);

    let c = -2.0 * a;
    assert_eq!(Gradient::new(c, a).value()[0], -2.0);

    let c = a / 3.0;
    assert_eq!(Gradient::new(c, a).value()[0], 1.0 / 3.0);

    a.set_value(100.0);
    b.set_value(200.0);

    let c = a + b;
    assert_eq!(Gradient::new(c, a).value()[0], 1.0);
    assert_eq!(Gradient::new(c, b).value()[0], 1.0);

    let c = a - b;
    assert_eq!(Gradient::new(c, a).value()[0], 1.0);
    assert_eq!(Gradient::new(c, b).value()[0], -1.0);

    let c = -a + b;
    assert_eq!(Gradient::new(c, a).value()[0], -1.0);
    assert_eq!(Gradient::new(c, b).value()[0], 1.0);

    let c = a + 1.0;
    assert_eq!(Gradient::new(c, a).value()[0], 1.0);
}

#[test]
fn trigonometry() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = free_vars(&mut problem, &[0.5])[0];

    assert_eq!(sin(x).value(), x.value().sin());
    assert_eq!(Gradient::new(sin(x), x).value()[0], x.value().cos());

    assert_eq!(cos(x).value(), x.value().cos());
    assert_eq!(Gradient::new(cos(x), x).value()[0], -x.value().sin());

    assert_eq!(tan(x).value(), x.value().tan());
    assert_eq!(
        Gradient::new(tan(x), x).value()[0],
        1.0 / (x.value().cos() * x.value().cos()),
    );

    assert_eq!(asin(x).value(), x.value().asin());
    assert_eq!(
        Gradient::new(asin(x), x).value()[0],
        1.0 / (1.0 - x.value() * x.value()).sqrt(),
    );

    assert_eq!(acos(x).value(), x.value().acos());
    assert_eq!(
        Gradient::new(acos(x), x).value()[0],
        -1.0 / (1.0 - x.value() * x.value()).sqrt(),
    );

    assert_eq!(atan(x).value(), x.value().atan());
    assert_eq!(
        Gradient::new(atan(x), x).value()[0],
        1.0 / (1.0 + x.value() * x.value()),
    );
}

#[test]
fn hyperbolic() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = free_vars(&mut problem, &[1.0])[0];

    assert_eq!(sinh(x).value(), x.value().sinh());
    assert_eq!(Gradient::new(sinh(x), x).value()[0], x.value().cosh());

    assert_eq!(cosh(x).value(), x.value().cosh());
    assert_eq!(Gradient::new(cosh(x), x).value()[0], x.value().sinh());

    assert_eq!(tanh(x).value(), x.value().tanh());
    assert_eq!(
        Gradient::new(tanh(x), x).value()[0],
        1.0 / (x.value().cosh() * x.value().cosh()),
    );
}

#[test]
fn exponential() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = free_vars(&mut problem, &[1.0])[0];

    assert_eq!(log(x).value(), x.value().ln());
    assert_eq!(Gradient::new(log(x), x).value()[0], 1.0 / x.value());

    assert_eq!(log10(x).value(), x.value().log10());
    assert_eq!(
        Gradient::new(log10(x), x).value()[0],
        1.0 / (10.0_f64.ln() * x.value()),
    );

    assert_eq!(exp(x).value(), x.value().exp());
    assert_eq!(Gradient::new(exp(x), x).value()[0], x.value().exp());
}

#[test]
fn power() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let vs = free_vars(&mut problem, &[1.0, 2.0]);
    let (x, a) = (vs[0], vs[1]);
    let y = 2.0 * a;
    let two = Variable::constant_in(&arena, 2.0);

    assert_eq!(sqrt(x).value(), x.value().sqrt());
    assert_eq!(Gradient::new(sqrt(x), x).value()[0], 0.5 / x.value().sqrt(),);

    assert_eq!(sqrt(a).value(), a.value().sqrt());
    assert_eq!(Gradient::new(sqrt(a), a).value()[0], 0.5 / a.value().sqrt(),);

    assert_eq!(cbrt(x).value(), x.value().cbrt());
    assert_eq!(
        Gradient::new(cbrt(x), x).value()[0],
        1.0 / (3.0 * x.value().cbrt() * x.value().cbrt()),
    );

    // x²
    assert_eq!(pow(x, 2.0).value(), x.value().powi(2));
    assert_eq!(Gradient::new(pow(x, 2.0), x).value()[0], 2.0 * x.value());

    // 2ˣ
    assert_eq!(pow(two, x).value(), 2.0_f64.powf(x.value()));
    assert_eq!(
        Gradient::new(pow(two, x), x).value()[0],
        2.0_f64.ln() * 2.0_f64.powf(x.value()),
    );

    // xˣ
    assert_eq!(pow(x, x).value(), x.value().powf(x.value()));
    assert_eq!(
        Gradient::new(pow(x, x), x).value()[0],
        (x.value().ln() + 1.0) * x.value().powf(x.value()),
    );

    // y(a)
    assert_eq!(y.value(), 2.0 * a.value());
    assert_eq!(Gradient::new(y, a).value()[0], 2.0);

    // xʸ(x)
    assert_eq!(pow(x, y).value(), x.value().powf(y.value()));
    assert_eq!(
        Gradient::new(pow(x, y), x).value()[0],
        y.value() / x.value() * x.value().powf(y.value()),
    );

    // xʸ(y)
    assert_eq!(pow(x, y).value(), x.value().powf(y.value()));
    assert_eq!(
        Gradient::new(pow(x, y), y).value()[0],
        x.value().ln() * x.value().powf(y.value()),
    );
}

#[test]
fn abs_test() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = free_vars(&mut problem, &[0.0])[0];
    let mut g = Gradient::new(abs(x), x);

    x.set_value(1.0);
    assert_eq!(abs(x).value(), 1.0);
    assert_eq!(g.value()[0], 1.0);

    x.set_value(-1.0);
    assert_eq!(abs(x).value(), 1.0);
    assert_eq!(g.value()[0], -1.0);

    x.set_value(0.0);
    assert_eq!(abs(x).value(), 0.0);
    assert_eq!(g.value()[0], 0.0);
}

#[test]
fn atan2_test() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let vs = free_vars(&mut problem, &[1.0, 0.9]);
    let (x, y) = (vs[0], vs[1]);
    let two = Variable::constant_in(&arena, 2.0);

    assert_eq!(atan2(two, x).value(), 2.0_f64.atan2(x.value()));
    let g = Gradient::new(atan2(two, x), x).value()[0];
    assert!((g - -2.0 / (4.0 + x.value() * x.value())).abs() <= EPS);

    assert_eq!(atan2(x, 2.0).value(), x.value().atan2(2.0));
    let g = Gradient::new(atan2(x, 2.0), x).value()[0];
    assert!((g - 2.0 / (4.0 + x.value() * x.value())).abs() <= EPS);

    x.set_value(1.1);
    y.set_value(0.9);
    assert_eq!(atan2(y, x).value(), y.value().atan2(x.value()));
    let gy = Gradient::new(atan2(y, x), y).value()[0];
    assert!((gy - x.value() / (x.value() * x.value() + y.value() * y.value())).abs() <= EPS);
    let gx = Gradient::new(atan2(y, x), x).value()[0];
    assert!((gx - -y.value() / (x.value() * x.value() + y.value() * y.value())).abs() <= EPS);
}

#[test]
fn hypot_test() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let vs = free_vars(&mut problem, &[1.8, 1.5]);
    let (x, y) = (vs[0], vs[1]);
    let two = Variable::constant_in(&arena, 2.0);

    assert_eq!(hypot(x, 2.0).value(), x.value().hypot(2.0));
    assert_eq!(
        Gradient::new(hypot(x, 2.0), x).value()[0],
        x.value() / x.value().hypot(2.0),
    );

    assert_eq!(hypot(two, y).value(), 2.0_f64.hypot(y.value()));
    assert_eq!(
        Gradient::new(hypot(two, y), y).value()[0],
        y.value() / 2.0_f64.hypot(y.value()),
    );

    x.set_value(1.3);
    y.set_value(2.3);
    assert_eq!(hypot(x, y).value(), x.value().hypot(y.value()));
    assert_eq!(
        Gradient::new(hypot(x, y), x).value()[0],
        x.value() / x.value().hypot(y.value()),
    );
    assert_eq!(
        Gradient::new(hypot(x, y), y).value()[0],
        y.value() / x.value().hypot(y.value()),
    );

    let z = free_vars(&mut problem, &[3.3])[0];
    x.set_value(1.3);
    y.set_value(2.3);
    let h3 = (x.value() * x.value() + y.value() * y.value() + z.value() * z.value()).sqrt();
    assert_eq!(hypot3(x, y, z).value(), h3);
    assert_eq!(Gradient::new(hypot3(x, y, z), x).value()[0], x.value() / h3);
    assert_eq!(Gradient::new(hypot3(x, y, z), y).value()[0], y.value() / h3);
    assert_eq!(Gradient::new(hypot3(x, y, z), z).value()[0], z.value() / h3);
}

#[test]
fn max_test() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = free_vars(&mut problem, &[2.0])[0];
    let x2 = x * x;
    let x3 = x * x * x;

    assert_eq!(max(x2, x3).value(), x3.value());
    assert_eq!(
        Gradient::new(max(x2, x3), x).value()[0],
        Gradient::new(x3, x).value()[0],
    );

    assert_eq!(max(x3, x2).value(), x3.value());
    assert_eq!(
        Gradient::new(max(x3, x2), x).value()[0],
        Gradient::new(x3, x).value()[0],
    );

    assert_eq!(max(x, x).value(), x.value());
    assert_eq!(Gradient::new(max(x, x), x).value()[0], 1.0);
}

#[test]
fn min_test() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = free_vars(&mut problem, &[2.0])[0];
    let x2 = x * x;
    let x3 = x * x * x;

    assert_eq!(min(x2, x3).value(), x2.value());
    assert_eq!(
        Gradient::new(min(x2, x3), x).value()[0],
        Gradient::new(x2, x).value()[0],
    );

    assert_eq!(min(x3, x2).value(), x2.value());
    assert_eq!(
        Gradient::new(min(x3, x2), x).value()[0],
        Gradient::new(x2, x).value()[0],
    );

    assert_eq!(min(x, x).value(), x.value());
    assert_eq!(Gradient::new(min(x, x), x).value()[0], 1.0);
}

#[test]
fn miscellaneous() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = free_vars(&mut problem, &[3.0])[0];

    assert_eq!(abs(x).value(), 3.0);
    assert_eq!(Gradient::new(x, x).value()[0], 1.0);

    x.set_value(0.5);
    let inv_sqrtpi = 1.0 / std::f64::consts::PI.sqrt();
    let expected_grad = 2.0 * inv_sqrtpi * (-x.value() * x.value()).exp();
    let g = Gradient::new(erf(x), x).value()[0];
    assert!((g - expected_grad).abs() < 1e-12);
}

#[test]
fn variable_reuse() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let vs = free_vars(&mut problem, &[10.0, 20.0]);
    let (a, b) = (vs[0], vs[1]);
    let x = a * b;

    let mut g = Gradient::new(x, a);
    assert_eq!(g.value()[0], 20.0);

    b.set_value(10.0);
    assert_eq!(g.value()[0], 10.0);
}

#[test]
fn sign_test() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = free_vars(&mut problem, &[0.0])[0];

    for &v in &[1.0, -1.0, 0.0] {
        x.set_value(v);
        let expected = if v < 0.0 {
            -1.0
        } else if v == 0.0 {
            0.0
        } else {
            1.0
        };
        assert_eq!(sign(x).value(), expected);
        assert_eq!(Gradient::new(sign(x), x).value()[0], 0.0);
    }
}

#[test]
fn non_scalar() {
    let arena = VariableArena::new();
    let mut x = VariableMatrix::zeros_in(&arena, 3, 1);
    x.set_value(&ndarray::Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap());

    // y = x₀ + 3·x₁ − 5·x₂  →  dy/dx = [1, 3, −5]
    let y = x.get(0, 0) + 3.0 * x.get(1, 0) - 5.0 * x.get(2, 0);
    let mut g = Gradient::new(y, &x);
    let values = g.value();
    assert_eq!(values.len(), 3);
    assert_eq!(values[0], 1.0);
    assert_eq!(values[1], 3.0);
    assert_eq!(values[2], -5.0);
}
