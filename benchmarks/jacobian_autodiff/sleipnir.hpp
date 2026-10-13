// Copyright (c) Sleipnir contributors

#pragma once

#include <span>

/// Registers Sleipnir benchmarks for the Jacobian.
///
/// @param nums_variables The numbers of variables for which to record results.
void register_sleipnir_benchmarks(std::span<const int> nums_variables);
