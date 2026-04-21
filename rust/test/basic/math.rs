//! Checks the free math functions against a handful of known values.

use hafgufa::math;
use hafgufa::{Variable, VariableArena};

const TOL: f64 = 1e-12;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < TOL
}

#[test]
fn unary_math() {
    let arena = VariableArena::new();
    let zero = Variable::constant_in(&arena, 0.0);
    let one = Variable::constant_in(&arena, 1.0);
    assert!(close(math::sin(zero).value(), 0.0));
    assert!(close(math::cos(zero).value(), 1.0));
    assert!(close(math::exp(zero).value(), 1.0));
    assert!(close(math::log(one).value(), 0.0));
    assert!(close(
        math::sqrt(Variable::constant_in(&arena, 9.0)).value(),
        3.0
    ));
    assert!(close(
        math::abs(Variable::constant_in(&arena, -2.5)).value(),
        2.5
    ));
    assert!(close(
        math::cbrt(Variable::constant_in(&arena, 27.0)).value(),
        3.0
    ));
    assert!(close(math::tanh(zero).value(), 0.0));
}

#[test]
fn binary_math() {
    let arena = VariableArena::new();
    assert!(close(
        math::pow(Variable::constant_in(&arena, 2.0), 10.0).value(),
        1024.0
    ));
    // hypot with f64 `x` isn't allowed (no arena); pass x as a Variable.
    assert!(close(
        math::hypot(Variable::constant_in(&arena, 3.0), 4.0).value(),
        5.0
    ));
    assert!(close(
        math::max(Variable::constant_in(&arena, 1.0), 3.0).value(),
        3.0
    ));
    assert!(close(
        math::min(Variable::constant_in(&arena, 1.0), 3.0).value(),
        1.0
    ));
    assert!(close(
        math::atan2(Variable::constant_in(&arena, 0.0), 1.0).value(),
        0.0
    ));
}

#[test]
fn arithmetic_with_scalars() {
    let arena = VariableArena::new();
    let x = Variable::constant_in(&arena, 3.0);
    assert!(close((x + 1.0).value(), 4.0));
    assert!(close((1.0 + x).value(), 4.0));
    assert!(close((x - 1.0).value(), 2.0));
    assert!(close((2.0 * x).value(), 6.0));
    assert!(close((x / 2.0).value(), 1.5));
    assert!(close((-x).value(), -3.0));
}
