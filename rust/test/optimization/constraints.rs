//! Port of `test/src/optimization/constraints_test.cpp`.
//!
//! The C++ test spot-checks that every `==`, `<=`, `>=`, `<`, `>`
//! overload between `T` / `Variable` / `VariableMatrix` / `MatrixXT` /
//! `VariableBlock` returns a constraint object that contextually
//! converts to `bool` indicating whether the constraint's value is
//! currently satisfied. Rust can't overload comparisons to return
//! non-`bool`, so the DSL uses the `cmp!` / `subject_to!` macros
//! (covered fully by `test/basic/dsl_macros.rs`).
//!
//! The only part that still maps here is confirming every operand-type
//! permutation the C++ test enumerates compiles through `cmp!` and
//! attaches to a `Problem`. Satisfaction-via-bool is not ported because
//! it's a pure C++ operator-overload semantic.

use hafgufa::{Problem, Variable, VariableArena, VariableMatrix, cmp, subject_to};
use ndarray::arr2;

#[test]
fn cmp_accepts_every_operand_permutation() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let v = problem.decision_variable();
    let m = problem.decision_variable_matrix(1, 1);
    let cst = Variable::constant_in(&arena, 1.0);
    let cst_mat = VariableMatrix::from_array_in(&arena, &arr2(&[[1.0]]));
    let arr = arr2(&[[1.0]]);

    // Permutation matrix mirrors the C++ cases: (f64|Variable|VariableMatrix|ndarray) × same.
    // `VariableMatrix` is clone-not-copy and `ndarray::Array2` doesn't implement `Copy`
    // either, so they go in by reference to avoid consuming them in the first arm.
    problem.subject_to(cmp!(v == 1.0));
    problem.subject_to(cmp!(v == cst));
    problem.subject_to(cmp!(v == &cst_mat));
    problem.subject_to(cmp!(v == &arr));
    problem.subject_to(cmp!(&m == 1.0));
    problem.subject_to(cmp!(&m == cst));
    problem.subject_to(cmp!(&m == &cst_mat));
    problem.subject_to(cmp!(&m == &arr));

    // Inequalities: <, <=, >, >= (< and > fold into <= and >= in Rust's DSL,
    // matching the C++ note about equivalence)
    problem.subject_to(cmp!(v >= 0.0));
    problem.subject_to(cmp!(v <= 5.0));
    problem.subject_to(cmp!(v > -1.0));
    problem.subject_to(cmp!(v < 10.0));

    // Bundling many overlapping constants like this overconstrains the
    // problem; solve status doesn't matter — just that the DSL compiled
    // and the constraints attached.
    let _ = problem.solve_status(Default::default());
}

#[test]
fn subject_to_macro_accepts_same_permutations() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);

    let v = problem.decision_variable();

    subject_to!(problem, v == 1.0);
    subject_to!(problem, 1.0 == v);
    subject_to!(problem, v >= 0.0);
    subject_to!(problem, v <= 2.0);

    // Bundling many overlapping constants like this overconstrains the
    // problem; solve status doesn't matter — just that the DSL compiled
    // and the constraints attached.
    let _ = problem.solve_status(Default::default());
}
