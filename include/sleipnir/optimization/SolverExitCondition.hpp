// Copyright (c) Sleipnir contributors

#pragma once

namespace sleipnir {

/**
 * Solver exit condition.
 */
enum class SolverExitCondition {
  /// The solver found an optimal solution.
  kOk,
  /// The solver reached an acceptable tolerance, but not the desired one.
  kReachedAcceptableTolerance,
  /// The solver determined the problem to be overconstrained and gave up.
  kTooFewDOFs,
  /// The solver determined the problem to be locally infeasible and gave up.
  kLocallyInfeasible,
  /// The solver failed to reach the desired tolerance due to a numerical issue
  /// (bad step).
  kNumericalIssue_BadStep,
  /// The solver failed to reach the desired tolerance due to a numerical issue
  /// (max step too small).
  kNumericalIssue_MaxStepTooSmall,
  /// The solver returned its solution so far after exceeding the maximum number
  /// of iterations.
  kMaxIterations,
  /// The solver returned its solution so far after exceeding the maximum
  /// elapsed wall clock time.
  kTimeout
};

}  // namespace sleipnir
