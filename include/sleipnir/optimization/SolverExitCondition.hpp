// Copyright (c) Sleipnir contributors

#pragma once

namespace sleipnir {

/**
 * Solver exit condition.
 */
enum class SolverExitCondition {
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
  /// The solver failed to reach the desired tolerance due to a bad search
  /// direction.
  kBadSearchDirection = -3,
  /// The solver failed to reach the desired tolerance due to the maximum search
  /// direction becoming too small.
  kMaxSearchDirectionTooSmall = -4,
  /// The solver encountered diverging primal iterates pₖˣ and/or pₖˢ and gave
  /// up.
  kDivergingIterates = -5,
  /// The solver returned its solution so far after exceeding the maximum number
  /// of iterations.
  kMaxIterationsExceeded = -6,
  /// The solver returned its solution so far after exceeding the maximum
  /// elapsed wall clock time.
  kMaxWallClockTimeExceeded = -7
};

}  // namespace sleipnir
