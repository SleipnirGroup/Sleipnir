// Copyright (c) Sleipnir contributors

#pragma once

namespace sleipnir {

/**
 * Solver exit condition.
 */
enum class SolverExitCondition {
  /// Solved the problem to the desired tolerance.
  kSuccess,
  /// Solved the problem to an acceptable tolerance, but not the desired one.
  kSolvedToAcceptableTolerance,
  /// The solver determined the problem to be overconstrained and gave up.
  kTooFewDOFs,
  /// The solver determined the problem to be locally infeasible and gave up.
  kLocallyInfeasible,
  /// The solver failed to reach the desired tolerance due to a bad search
  /// direction.
  kBadSearchDirection,
  /// The solver failed to reach the desired tolerance due to the maximum search
  /// direction becoming too small.
  kMaxSearchDirectionTooSmall,
  /// The solver encountered diverging primal iterates pₖˣ and/or pₖˢ and gave
  /// up.
  kDivergingIterates,
  /// The solver returned its solution so far after exceeding the maximum number
  /// of iterations.
  kMaxIterationsExceeded,
  /// The solver returned its solution so far after exceeding the maximum
  /// elapsed wall clock time.
  kMaxWallClockTimeExceeded
};

}  // namespace sleipnir
