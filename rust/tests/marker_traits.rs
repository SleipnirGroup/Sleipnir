//! Sanity tests for the doc-hidden marker traits that the DSL macros
//! use for static type checks. These verify *valid* types go through —
//! the compile-fail side (verifying that e.g. `String` produces a
//! `SleipnirOperand` diagnostic) would need `trybuild`, which we don't
//! depend on.

use hafgufa::__marker::{SleipnirMatrix, SleipnirOperand, SleipnirProblem, SleipnirScalar};
use hafgufa::{OCP, Problem, TimestepMethod, Variable, VariableArena, VariableMatrix};
use ndarray::Array2;

fn accept_scalar<T: SleipnirScalar>(_: T) {}
fn accept_matrix<T: SleipnirMatrix>(_: T) {}
fn accept_operand<T: SleipnirOperand>(_: T) {}
fn accept_problem<P: SleipnirProblem>(_: &mut P) {}

#[test]
fn scalar_markers() {
    let arena = VariableArena::new();
    let v = Variable::constant_in(&arena, 1.0);
    accept_scalar(v);
    let v2 = Variable::constant_in(&arena, 2.0);
    accept_scalar(&v2);
    accept_scalar(3.0_f64);
    accept_scalar(4_i32);
}

#[test]
fn matrix_markers() {
    let arena = VariableArena::new();
    let m = VariableMatrix::zeros_in(&arena, 2, 2);
    accept_matrix(m);
    let m2 = VariableMatrix::zeros_in(&arena, 2, 2);
    accept_matrix(&m2);
    accept_matrix(Array2::<f64>::zeros((2, 2)));
    let arr = Array2::<f64>::zeros((2, 2));
    accept_matrix(&arr);
}

#[test]
fn operand_covers_scalars_and_matrices() {
    let arena = VariableArena::new();
    accept_operand(Variable::constant_in(&arena, 1.0));
    let v2 = Variable::constant_in(&arena, 2.0);
    accept_operand(&v2);
    accept_operand(3.0_f64);
    accept_operand(4_i32);
    accept_operand(VariableMatrix::zeros_in(&arena, 1, 1));
    let m2 = VariableMatrix::zeros_in(&arena, 1, 1);
    accept_operand(&m2);
    accept_operand(Array2::<f64>::zeros((1, 1)));
    let arr = Array2::<f64>::zeros((1, 1));
    accept_operand(&arr);
}

fn identity_dyn<'a>() -> impl FnMut(&VariableMatrix<'a>, &VariableMatrix<'a>) -> VariableMatrix<'a> + Send + 'a {
    |x, u| x + u
}

#[test]
fn problem_marker_accepts_problem_and_ocp() {
    let arena = VariableArena::new();
    let mut p = Problem::new(&arena);
    accept_problem(&mut p);

    let arena2 = VariableArena::new();
    let mut ocp = OCP::new_discrete(
        &arena2,
        1,
        1,
        std::time::Duration::from_millis(10),
        5,
        identity_dyn(),
        TimestepMethod::Fixed,
    );
    accept_problem(&mut ocp);
}
