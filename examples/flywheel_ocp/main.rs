//! Rust port of `examples/flywheel_ocp/main.py`.

use std::time::Duration;

use hafgufa::{OCP, TimestepMethod, VariableArena, VariableMatrix};
use ndarray::Array2;

/// Helper that names the arena lifetime so the closure's three
/// `VariableMatrix<'a>` operands unify. Rust can't introduce a lifetime
/// inside a closure body, so we factor it into a `fn` with a generic `'a`.
fn make_flywheel_dynamics<'a>(
    a: f64,
    b: f64,
) -> impl FnMut(&VariableMatrix<'a>, &VariableMatrix<'a>) -> VariableMatrix<'a> + Send + 'a {
    move |x, u| a * x + b * u
}

fn main() {
    let total_time = 5.0_f64;
    let dt = 0.005_f64;
    let n: i32 = (total_time / dt) as i32;

    let a = (-dt).exp();
    let b = 1.0 - a;

    let arena = VariableArena::new();
    let mut solver = OCP::new_discrete(
        &arena,
        1,
        1,
        Duration::from_secs_f64(dt),
        n,
        make_flywheel_dynamics(a, b),
        TimestepMethod::Fixed,
    );
    solver.constrain_initial_state(0.0);
    solver.set_upper_input_bound(12.0);
    solver.set_lower_input_bound(-12.0);

    let r = 10.0_f64;
    let r_mat = arena.array(&Array2::from_elem((1, (n + 1) as usize), r));
    let diff = &r_mat - &solver.x();
    let cost = &diff * &diff.t();
    solver.minimize_matrix(&cost);

    solver
        .solve(Default::default())
        .expect("flywheel OCP failed to solve");

    let mut x = solver.x();
    let mut u = solver.u();
    println!("x\u{2080} = {}", x.value_at(0, 0));
    println!("u\u{2080} = {}", u.value_at(0, 0));
}
