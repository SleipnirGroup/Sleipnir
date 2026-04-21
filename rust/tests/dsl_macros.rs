//! Exercises the DSL macros: `cmp!` and the `subject_to!` tt-muncher
//! in every supported syntactic form.

use hafgufa::{Problem, VariableArena, cmp, math, subject_to};

#[test]
fn cmp_equality_standalone() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = problem.decision_variable();
    let y = problem.decision_variable();

    problem.minimize(x * x + y * y);
    let c = cmp!(x + y == 1.0);
    problem.subject_to(c);

    problem.solve(Default::default()).unwrap();
    assert!((x.value() + y.value() - 1.0).abs() < 1e-6);
}

#[test]
fn cmp_inequalities_standalone() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = problem.decision_variable();

    problem.minimize(x * x);
    problem.subject_to(cmp!(x >= 0.5));
    problem.subject_to(cmp!(x <= 2.0));

    problem.solve(Default::default()).unwrap();
    assert!((x.value() - 0.5).abs() < 1e-6);
}

#[test]
fn subject_to_equality_via_operator() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = problem.decision_variable();
    let y = problem.decision_variable();

    problem.minimize(x * x + y * y);
    subject_to!(problem, x + y == 1.0);

    problem.solve(Default::default()).unwrap();
    assert!((x.value() + y.value() - 1.0).abs() < 1e-6);
}

#[test]
fn subject_to_all_inequality_operators() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = problem.decision_variable();

    problem.minimize(x * x);
    subject_to!(problem, x >= 0.25);
    subject_to!(problem, x <= 10.0);
    subject_to!(problem, x > -100.0);
    subject_to!(problem, x < 100.0);

    problem.solve(Default::default()).unwrap();
    assert!((x.value() - 0.25).abs() < 1e-6);
}

#[test]
fn cmp_parses_complex_expression() {
    // Exercises the tt-muncher with a deeply-nested expression containing
    // function calls, nested operators, parenthesized subexpressions,
    // scalar broadcast, and f64 literals on both sides of the `==`
    // marker. This is the worst-case shape `cmp!` is expected to handle.
    //
    // Problem: minimize r² = (x−1)² + (y−2)² subject to a nonlinear
    // circle constraint through `cmp!`:
    //     sqrt((x − 1)² + (y − 2)²) + max(x, 0.5) − 2.0 * exp(0.0) == 1.5
    // which simplifies to: distance-from-(1, 2) + max(x, 0.5) − 2 == 1.5.
    // Feasible solutions lie on a curve; the minimum of r² on that
    // curve is an analytic fixed point the solver converges to.
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = problem.decision_variable();
    let y = problem.decision_variable();
    x.set_value(2.0);
    y.set_value(2.0);

    problem.minimize(math::pow(x - 1.0, 2.0) + math::pow(y - 2.0, 2.0));

    subject_to!(
        problem,
        math::sqrt(math::pow(x - 1.0, 2.0) + math::pow(y - 2.0, 2.0))
            + math::max(x, 0.5)
            - 2.0 * math::exp(hafgufa::Variable::constant_in(&arena, 0.0))
            == 1.5
    );

    // Same shape inside a standalone `cmp!`.
    let c = cmp!(
        math::sqrt(math::pow(x - 1.0, 2.0) + math::pow(y - 2.0, 2.0))
            + math::max(x, 0.5)
            - 2.0 * math::exp(hafgufa::Variable::constant_in(&arena, 0.0))
            >= 0.0
    );
    problem.subject_to(c);

    problem.solve(Default::default()).unwrap();

    // Verify the equality constraint is actually satisfied.
    let dist = ((x.value() - 1.0).powi(2) + (y.value() - 2.0).powi(2)).sqrt();
    let lhs = dist + x.value().max(0.5) - 2.0 * 1.0_f64;
    assert!((lhs - 1.5).abs() < 1e-6, "equality residual = {}", lhs - 1.5);
}

#[test]
fn subject_to_accepts_prebuilt_constraint() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = problem.decision_variable();
    problem.minimize(x * x);

    let c: hafgufa::InequalityConstraints<'_> = cmp!(x >= 1.0);
    subject_to!(problem, c);

    problem.solve(Default::default()).unwrap();
    assert!((x.value() - 1.0).abs() < 1e-6);
}
