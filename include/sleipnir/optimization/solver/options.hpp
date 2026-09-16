// Copyright (c) Sleipnir contributors

#pragma once

#include <chrono>
#include <limits>

#include "sleipnir/util/symbol_exports.hpp"

namespace slp {

/// Solver options.
struct SLEIPNIR_DLLEXPORT Options {
  /// The solver will stop once the error is below this tolerance.
  double tolerance = 1e-8;

  /// The maximum number of solver iterations before returning a solution.
  int max_iterations = 5000;

  /// The maximum elapsed wall clock time before returning a solution.
  std::chrono::duration<double> timeout{
      std::numeric_limits<double>::infinity()};

  /// Enables diagnostic output.
  ///
  /// See https://sleipnirgroup.github.io/Sleipnir/md_usage.html#output for more
  /// information.
  bool diagnostics = false;
};

}  // namespace slp
