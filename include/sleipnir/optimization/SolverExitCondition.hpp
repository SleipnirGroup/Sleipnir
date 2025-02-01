// Copyright (c) Sleipnir contributors

#pragma once

#include <stdint.h>

#include <string_view>

#include "sleipnir/util/SymbolExports.hpp"

namespace sleipnir {

/**
 * Solver exit condition.
 */
enum class SolverExitCondition : int8_t {
  /// Solved the problem to the desired tolerance.
  kSuccess = 0,
  /// Solved the problem to an acceptable tolerance, but not the desired one.
  kSolvedToAcceptableTolerance = 1,
  /// The solver returned its solution so far after the user requested a stop.
  kCallbackRequestedStop = 2,
  /// The solver determined the problem to be overconstrained and gave up.
  kTooFewDOFs = -1,
  /// The solver determined the problem to be locally infeasible and gave up.
  kLocallyInfeasible = -2,
  /// The linear system factorization failed.
  kFactorizationFailed = -3,
  /// The backtracking line search failed, and the problem isn't locally
  /// infeasible.
  kLineSearchFailed = -4,
  /// The solver encountered nonfinite initial cost or constraints and gave up.
  kNonfiniteInitialCostOrConstraints = -5,
  /// The solver encountered diverging primal iterates xₖ and/or sₖ and gave up.
  kDivergingIterates = -6,
  /// The solver returned its solution so far after exceeding the maximum number
  /// of iterations.
  kMaxIterationsExceeded = -7,
  /// The solver returned its solution so far after exceeding the maximum
  /// elapsed wall clock time.
  kTimeout = -8
};

/**
 * Returns user-readable message corresponding to the exit condition.
 *
 * @param exitCondition Solver exit condition.
 */
SLEIPNIR_DLLEXPORT constexpr std::string_view ToMessage(
    const SolverExitCondition& exitCondition) {
  using enum SolverExitCondition;

  switch (exitCondition) {
    case kSuccess:
      return "solved to desired tolerance";
    case kSolvedToAcceptableTolerance:
      return "solved to acceptable tolerance";
    case kCallbackRequestedStop:
      return "callback requested stop";
    case kTooFewDOFs:
      return "problem has too few degrees of freedom";
    case kLocallyInfeasible:
      return "problem is locally infeasible";
    case kFactorizationFailed:
      return "linear system factorization failed";
    case kLineSearchFailed:
      return "backtracking line search failed, and the problem isn't locally "
             "infeasible";
    case kNonfiniteInitialCostOrConstraints:
      return "solver encountered nonfinite initial cost or constraints and "
             "gave up";
    case kDivergingIterates:
      return "solver encountered diverging primal iterates xₖ and/or sₖ and "
             "gave up";
    case kMaxIterationsExceeded:
      return "solution returned after maximum iterations exceeded";
    case kTimeout:
      return "solution returned after maximum wall clock time exceeded";
    default:
      return "unknown";
  }
}

}  // namespace sleipnir
