//! Port of `test/src/autodiff/variable_test.cpp`. The C++ "default
//! constructor" path (a bare `Variable` with linear type) corresponds to
//! the Rust `problem.decision_variable()` entry point, since the Rust
//! bindings only expose `Variable::constant_in` outside a `Problem`.

use hafgufa::{ExpressionType, Problem, Variable, VariableArena};

#[test]
fn default_constructor() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let a = problem.decision_variable();
    assert_eq!(a.value(), 0.0);
    assert_eq!(a.expression_type(), ExpressionType::Linear);
}

#[test]
fn constant_constructor_from_f64() {
    let arena = VariableArena::new();
    let a = Variable::constant_in(&arena, 1.0);
    assert_eq!(a.value(), 1.0);
    assert_eq!(a.expression_type(), ExpressionType::Constant);
}

#[test]
fn constant_constructor_from_i32() {
    let arena = VariableArena::new();
    let b = Variable::constant_in(&arena, 2.0);
    assert_eq!(b.value(), 2.0);
    assert_eq!(b.expression_type(), ExpressionType::Constant);
}

#[test]
fn set_value() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let a = problem.decision_variable();

    a.set_value(1.0);
    assert_eq!(a.value(), 1.0);

    a.set_value(2.0);
    assert_eq!(a.value(), 2.0);
}
