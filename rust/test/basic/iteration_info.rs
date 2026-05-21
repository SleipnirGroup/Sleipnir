//! Tests for the zero-copy `IterationInfo` views (dense slices + sparse
//! gradient / Hessian / Jacobians).

use std::sync::{Arc, Mutex};

use hafgufa::{Problem, VariableArena, math, subject_to};

#[test]
fn iteration_info_exposes_dense_slices() {
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = problem.decision_variable();
    let y = problem.decision_variable();

    problem.minimize(x * x + y * y);
    subject_to!(problem, x + y == 1.0);

    let captured = Arc::new(Mutex::new(None::<(usize, usize, usize, usize)>));
    let cap = Arc::clone(&captured);
    problem.add_callback(move |info| {
        *cap.lock().unwrap() = Some((
            info.x().len(),
            info.s().len(),
            info.y().len(),
            info.z().len(),
        ));
        false
    });

    problem.solve(Default::default()).unwrap();

    let (x_len, s_len, y_len, z_len) = captured.lock().unwrap().unwrap();
    assert_eq!(x_len, 2, "two decision variables");
    assert_eq!(s_len, 0, "no inequality slacks");
    assert_eq!(y_len, 1, "one equality multiplier");
    assert_eq!(z_len, 0, "no inequality multipliers");
}

#[test]
fn iteration_info_exposes_sparse_matrices() {
    // Nonlinear problem forces the solver to report nonzero gradient,
    // Hessian, and equality Jacobian entries.
    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = problem.decision_variable();
    let y = problem.decision_variable();

    problem.minimize(math::pow(x - 2.0, 4.0) + math::pow(y - 3.0, 4.0));
    subject_to!(problem, x + y == 5.0);
    subject_to!(problem, x >= 0.0);

    let saw_data = Arc::new(Mutex::new(false));
    let flag = Arc::clone(&saw_data);

    problem.add_callback(move |info| {
        let g = info.gradient();
        let h = info.hessian();
        let a_e = info.equality_jacobian();
        let a_i = info.inequality_jacobian();

        // Dimensions should match the problem shape.
        assert_eq!(g.size, 2);
        assert_eq!(h.rows, 2);
        assert_eq!(h.cols, 2);
        assert_eq!(a_e.rows, 1);
        assert_eq!(a_e.cols, 2);
        assert_eq!(a_i.rows, 1);
        assert_eq!(a_i.cols, 2);

        // Sparse Hessian densifies correctly.
        let dense_h = h.to_dense();
        assert_eq!(dense_h.shape(), &[2, 2]);

        // Equality Jacobian of `x + y - 5 = 0` is the row `[1, 1]`.
        let dense_a_e = a_e.to_dense();
        assert!((dense_a_e[[0, 0]] - 1.0).abs() < 1e-9);
        assert!((dense_a_e[[0, 1]] - 1.0).abs() < 1e-9);

        // We saw at least one iteration with nonzero data.
        if g.nnz() > 0 {
            *flag.lock().unwrap() = true;
        }
        false
    });

    problem.solve(Default::default()).unwrap();

    assert!(
        *saw_data.lock().unwrap(),
        "expected at least one iteration with a nonzero gradient"
    );
}
