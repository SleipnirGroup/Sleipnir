//! Rust port of `examples/current_manager/main.py`.

use hafgufa::{Problem, VariableArena, VariableMatrix, subject_to};

struct CurrentManager<'a> {
    problem: Problem<'a>,
    desired_currents: VariableMatrix<'a>,
    allocated_currents: VariableMatrix<'a>,
}

impl<'a> CurrentManager<'a> {
    /// Builds the QP once. [`calculate`](Self::calculate) then reuses it
    /// with new desired-current values on every call.
    fn new(arena: &'a VariableArena, current_tolerances: &[f64], max_current: f64) -> Self {
        let n = current_tolerances.len() as i32;

        let mut problem = Problem::new(arena);

        // Fresh 1x1 "constant" slots for desired currents; values get
        // updated each tick via `set_value_at` so the cost expression
        // built here keeps referencing them. Matches the Python binding's
        // pattern of using a decision-variable-shaped matrix initialized
        // to +inf so Sleipnir doesn't constant-fold during graph
        // construction.
        let mut desired_currents = arena.zeros(n, 1);
        for i in 0..n {
            desired_currents.set_scalar(i, 0, f64::INFINITY);
        }

        let allocated_currents = problem.decision_variable_matrix(n, 1);

        // Cost: sum_i ((desired_i - allocated_i) / tol_i)^2.
        let mut cost = arena.constant(0.0);
        let mut current_sum = arena.constant(0.0);

        for i in 0..n {
            let desired = desired_currents.get(i, 0);
            let allocated = allocated_currents.get(i, 0);
            let error = desired - allocated;
            let tol = current_tolerances[i as usize];
            cost = cost + error * error / (tol * tol);

            current_sum = current_sum + allocated;

            subject_to!(problem, allocated >= 0.0);
        }
        problem.minimize(cost);

        subject_to!(problem, current_sum <= max_current);

        Self {
            problem,
            desired_currents,
            allocated_currents,
        }
    }

    fn calculate(&mut self, desired_currents: &[f64]) -> Vec<f64> {
        let n = self.desired_currents.rows() as usize;
        assert_eq!(n, desired_currents.len());

        // Update the desired-current constant nodes in place so the cost
        // expression built in `new()` still references them.
        for (i, value) in desired_currents.iter().enumerate() {
            self.desired_currents.set_value_at(i as i32, 0, *value);
        }

        self.problem
            .solve(Default::default())
            .expect("current-manager QP failed to solve");

        (0..n)
            .map(|i| self.allocated_currents.value_at(i as i32, 0).max(0.0))
            .collect()
    }
}

fn main() {
    let arena = VariableArena::new();
    let mut manager = CurrentManager::new(&arena, &[1.0, 5.0, 10.0, 5.0], 40.0);
    let currents = manager.calculate(&[25.0, 10.0, 5.0, 0.0]);
    println!("currents = {currents:?}");
}
