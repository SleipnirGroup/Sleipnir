//! Helpers shared across optimization tests. Each sub-module mirrors a
//! header in `test/include/` in the C++ suite.

#![allow(dead_code)]

#[path = "cart_pole.rs"]
pub mod cart_pole;
#[path = "differential_drive.rs"]
pub mod differential_drive;
#[path = "lerp.rs"]
pub mod lerp;
#[path = "rk4.rs"]
pub mod rk4;
