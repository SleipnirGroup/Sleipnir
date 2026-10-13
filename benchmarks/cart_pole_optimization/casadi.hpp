// Copyright (c) Sleipnir contributors

#pragma once

#include <span>

/// Registers CasADi benchmarks for the cart-pole nonlinear optimization
/// problem.
///
/// @param nums_samples The numbers of samples for which to record results.
/// @param diagnostics True if diagnostic prints should be enabled.
void register_casadi_benchmarks(std::span<const int> nums_samples,
                                bool diagnostics);
