// Copyright (c) Joshua Nichols and Tyler Veness

#pragma once

#include "sleipnir/autodiff/Expression.hpp"
#include "sleipnir/optimization/SolverExitCondition.hpp"

namespace sleipnir {

/**
 * Return value of OptimizationProblem::Solve() containing the cost function and
 * constraint types and solver's exit condition.
 */
struct SolverStatus {
  /// The cost function type detected by the solver.
  ExpressionType costFunctionType = ExpressionType::kNone;

  /// The equality constraint type detected by the solver.
  ExpressionType equalityConstraintType = ExpressionType::kNone;

  /// The inequality constraint type detected by the solver.
  ExpressionType inequalityConstraintType = ExpressionType::kNone;

  /// The solver's exit condition.
  SolverExitCondition exitCondition = SolverExitCondition::kOk;
};

}  // namespace sleipnir
