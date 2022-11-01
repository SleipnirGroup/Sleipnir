// Copyright (c) Joshua Nichols and Tyler Veness

#pragma once

namespace sleipnir {

/**
 * Solver exit condition.
 */
enum class SolverExitCondition {
  /// The solver found an optimal solution.
  kOk,
  /// The solver determined the problem to be overconstrained and gave up.
  kTooFewDOFs,
  /// The solver determined the problem to be locally infeasible and gave up.
  kLocallyInfeasible,
  /// The solver returned its solution so far after exceeding the maximum number
  /// of iterations.
  kMaxIterations,
  /// The solver returned its solution so far after exceeding the maximum
  /// elapsed wall clock time.
  kTimeout
};

}  // namespace sleipnir
