// Copyright (c) Sleipnir contributors

#pragma once

#include <span>

/// Registers Sleipnir benchmarks for the cart-pole nonlinear optimization
/// problem.
///
/// @param nums_samples The numbers of samples for which to record results.
/// @param diagnostics True if diagnostic prints should be enabled.
void register_sleipnir_benchmarks(std::span<const int> nums_samples,
                                  bool diagnostics);
