use hafgufa::{VariableArena, VariableMatrix};
use ndarray::Array1;

/// 4th-order Runge–Kutta integration of `dx/dt = f(x, u)` over timestep
/// `dt` on f64 state vectors. Used by tests that need to simulate the
/// reference trajectory independently of the solver.
pub fn rk4_scalar<F>(mut f: F, x: Array1<f64>, u: &Array1<f64>, dt: f64) -> Array1<f64>
where
    F: FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64>,
{
    let k1 = f(&x, u);
    let k2 = f(&(&x + &(0.5 * dt * &k1)), u);
    let k3 = f(&(&x + &(0.5 * dt * &k2)), u);
    let k4 = f(&(&x + &(dt * &k3)), u);
    &x + &(dt / 6.0 * (&k1 + 2.0 * &k2 + 2.0 * &k3 + &k4))
}

/// RK4 for symbolic `VariableMatrix` states. Used inside the dynamics
/// constraint construction of an OCP. `f` is expected to produce a
/// state-derivative column matching `x`'s shape.
pub fn rk4_variable<'a, F>(
    _arena: &'a VariableArena,
    mut f: F,
    x: VariableMatrix<'a>,
    u: VariableMatrix<'a>,
    dt: f64,
) -> VariableMatrix<'a>
where
    F: FnMut(&VariableMatrix<'a>, &VariableMatrix<'a>) -> VariableMatrix<'a>,
{
    let k1 = f(&x, &u);
    let k2 = f(&(&x + (0.5 * dt) * &k1), &u);
    let k3 = f(&(&x + (0.5 * dt) * &k2), &u);
    let k4 = f(&(&x + dt * &k3), &u);

    // x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    let sum = k1 + 2.0 * k2 + 2.0 * k3 + k4;
    &x + (dt / 6.0) * sum
}
