// Copyright (c) Joshua Nichols and Tyler Veness

#pragma once

#include <chrono>
#include <limits>

#include <units/time.h>

namespace sleipnir {

/**
 * Solver configuration.
 */
struct SolverConfig {
  /// The solver will stop once the error is below this tolerance.
  double tolerance = 1e-6;

  /// The maximum number of solver iterations before returning a solution.
  int maxIterations = 1000;

  /// The maximum elapsed wall clock time before returning a solution.
  units::second_t timeout{std::numeric_limits<double>::infinity()};

  /// Enables diagnostic prints.
  bool diagnostics = false;
};

}  // namespace sleipnir
