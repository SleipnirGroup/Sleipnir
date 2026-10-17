// Copyright (c) Sleipnir contributors

#pragma once

#include <span>

/// Registers CasADi benchmarks for the lower-triangular Hessian.
///
/// @param nums_variables The numbers of variables for which to record results.
void register_casadi_benchmarks(std::span<const int> nums_variables);
