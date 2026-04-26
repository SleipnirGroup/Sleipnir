//! Aggregator for the `autodiff/` integration test folder. Cargo's
//! `[[test]]` entry points at this file; sibling `.rs` files are pulled
//! in as child modules so they compile into a single test binary.

#[path = "autodiff/gradient.rs"]
mod gradient;
#[path = "autodiff/hessian.rs"]
mod hessian;
#[path = "autodiff/jacobian.rs"]
mod jacobian;
#[path = "autodiff/variable.rs"]
mod variable;
#[path = "autodiff/variable_matrix.rs"]
mod variable_matrix;
