//! Rust port of `examples/flywheel_direct_transcription/main.py`.

use hafgufa::{Problem, VariableArena, subject_to};
use ndarray::Array2;

fn main() {
    let total_time = 5.0_f64;
    let dt = 0.005_f64;
    let n: i32 = (total_time / dt) as i32;

    // Flywheel model: states=[velocity], inputs=[voltage].
    let a = (-dt).exp();
    let b = 1.0 - a;

    let arena = VariableArena::new();
    let mut problem = Problem::new(&arena);
    let x = problem.decision_variable_matrix(1, n + 1);
    let u = problem.decision_variable_matrix(1, n);

    // Dynamics constraint for every step. `a * x_curr + b * u_curr` goes
    // through the IntoMatrixOperand-generic ops, so operands pass by value.
    for k in 0..n {
        let x_next = x.block(0, k + 1, 1, 1);
        let x_curr = x.block(0, k, 1, 1);
        let u_curr = u.block(0, k, 1, 1);
        subject_to!(problem, x_next == a * x_curr + b * u_curr);
    }

    // Initial state is zero.
    subject_to!(problem, x.block(0, 0, 1, 1) == 0.0);

    // Input bounds. `u` is still a VariableMatrix (not Copy) so it's
    // borrowed each time.
    subject_to!(problem, &u >= -12.0);
    subject_to!(problem, &u <= 12.0);

    // Cost: track r = 10 at every state.
    let r = arena.array(&Array2::from_elem((1, (n + 1) as usize), 10.0));
    let diff = r - &x;
    let cost = &diff * diff.t();
    problem.minimize_matrix(&cost);

    match problem.solve(Default::default()) {
        Ok(()) => println!("exit status: success"),
        Err(e) => println!("exit status: {e}"),
    }

    let mut x = x;
    let mut u = u;
    println!("x\u{2080} = {}", x.value_at(0, 0));
    println!("u\u{2080} = {}", u.value_at(0, 0));
}
