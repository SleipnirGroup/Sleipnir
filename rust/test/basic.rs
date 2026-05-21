//! Aggregator for the `basic/` integration test folder. Cargo's
//! `[[test]]` entry points at this file; each sibling `.rs` file is
//! pulled in as a child module so all `#[test]` functions register
//! under a single compiled test binary.

#[path = "basic/callbacks.rs"]
mod callbacks;
#[path = "basic/dsl_macros.rs"]
mod dsl_macros;
#[path = "basic/ergonomics.rs"]
mod ergonomics;
#[path = "basic/gradient.rs"]
mod gradient;
#[path = "basic/hessian_jacobian.rs"]
mod hessian_jacobian;
#[path = "basic/iteration_info.rs"]
mod iteration_info;
#[path = "basic/marker_traits.rs"]
mod marker_traits;
#[path = "basic/math.rs"]
mod math;
#[path = "basic/multistart.rs"]
mod multistart;
#[path = "basic/ocp.rs"]
mod ocp;
#[path = "basic/quadratic.rs"]
mod quadratic;
#[path = "basic/stacking.rs"]
mod stacking;
#[path = "basic/variable_matrix.rs"]
mod variable_matrix;
