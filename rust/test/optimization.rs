//! Aggregator for the `optimization/` integration test folder. Cargo's
//! `[[test]]` entry points at this file; sibling `.rs` files are pulled
//! in as child modules so they compile into a single test binary.

#[path = "optimization/common/mod.rs"]
mod common;

#[path = "optimization/arm_on_elevator_problem.rs"]
mod arm_on_elevator_problem;
#[path = "optimization/cart_pole_ocp.rs"]
mod cart_pole_ocp;
#[path = "optimization/cart_pole_problem.rs"]
mod cart_pole_problem;
#[path = "optimization/constraints.rs"]
mod constraints;
#[path = "optimization/decision_variable.rs"]
mod decision_variable;
#[path = "optimization/differential_drive_ocp.rs"]
mod differential_drive_ocp;
#[path = "optimization/differential_drive_problem.rs"]
mod differential_drive_problem;
#[path = "optimization/double_integrator_problem.rs"]
mod double_integrator_problem;
#[path = "optimization/exit_status.rs"]
mod exit_status;
#[path = "optimization/flywheel_ocp.rs"]
mod flywheel_ocp;
#[path = "optimization/flywheel_problem.rs"]
mod flywheel_problem;
#[path = "optimization/linear_problem.rs"]
mod linear_problem;
#[path = "optimization/multistart.rs"]
mod multistart;
#[path = "optimization/nonlinear_problem.rs"]
mod nonlinear_problem;
#[path = "optimization/quadratic_problem.rs"]
mod quadratic_problem;
#[path = "optimization/trivial_problem.rs"]
mod trivial_problem;
