//! Port of `test/src/optimization/decision_variable_test.cpp`. Drops
//! the C++ block/segment writeback sub-tests — the Rust `block()` API
//! returns a fresh matrix handle, not a writeable view, so assignment
//! to sub-blocks isn't part of the Rust surface.

use hafgufa::{Problem, VariableArena};
use ndarray::{Array2, arr2};

#[test]
fn scalar_zero_init_then_assign() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let x = problem.decision_variable();
    assert_eq!(x.value(), 0.0);

    x.set_value(1.0);
    assert_eq!(x.value(), 1.0);
    x.set_value(2.0);
    assert_eq!(x.value(), 2.0);
}

#[test]
fn vector_zero_init_then_assign() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let mut y = problem.decision_variable_matrix(2, 1);
    assert_eq!(y.value_at(0, 0), 0.0);
    assert_eq!(y.value_at(1, 0), 0.0);

    y.set_value_at(0, 0, 1.0);
    y.set_value_at(1, 0, 2.0);
    assert_eq!(y.value_at(0, 0), 1.0);
    assert_eq!(y.value_at(1, 0), 2.0);

    y.set_value_at(0, 0, 3.0);
    y.set_value_at(1, 0, 4.0);
    assert_eq!(y.value_at(0, 0), 3.0);
    assert_eq!(y.value_at(1, 0), 4.0);
}

#[test]
fn matrix_zero_init_then_assign() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let mut z = problem.decision_variable_matrix(3, 2);
    for r in 0..3 {
        for c in 0..2 {
            assert_eq!(z.value_at(r, c), 0.0);
        }
    }

    let assign = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    z.set_value(&assign);
    assert_eq!(z.value_at(0, 0), 1.0);
    assert_eq!(z.value_at(0, 1), 2.0);
    assert_eq!(z.value_at(1, 0), 3.0);
    assert_eq!(z.value_at(1, 1), 4.0);
    assert_eq!(z.value_at(2, 0), 5.0);
    assert_eq!(z.value_at(2, 1), 6.0);

    let assign2: Array2<f64> = arr2(&[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);
    z.set_value(&assign2);
    assert_eq!(z.value(), assign2);
}

#[test]
fn symmetric_decision_variable_is_symmetric() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let mut a = problem.symmetric_decision_variable(2);
    assert_eq!(a.value_at(0, 0), 0.0);
    assert_eq!(a.value_at(0, 1), 0.0);
    assert_eq!(a.value_at(1, 0), 0.0);
    assert_eq!(a.value_at(1, 1), 0.0);

    a.set_value_at(0, 0, 1.0);
    a.set_value_at(1, 0, 2.0);
    a.set_value_at(1, 1, 3.0);

    assert_eq!(a.value_at(0, 0), 1.0);
    assert_eq!(a.value_at(1, 0), 2.0);
    // (0,1) should mirror (1,0) thanks to the symmetric layout.
    assert_eq!(a.value_at(0, 1), 2.0);
    assert_eq!(a.value_at(1, 1), 3.0);
}
