// Copyright (c) Joshua Nichols and Tyler Veness

#pragma once

#include "sleipnir/autodiff/Expression.h"
#include "sleipnir/optimization/SolverExitCondition.h"

namespace sleipnir {

/**
 * Return value of OptimizationProblem::Solve() containing the cost function and
 * constraint types and solver's exit condition.
 */
struct SolverStatus {
  /// The cost function type detected by the solver.
  autodiff::ExpressionType costFunctionType = autodiff::ExpressionType::kNone;

  /// The equality constraint type detected by the solver.
  autodiff::ExpressionType equalityConstraintType =
      autodiff::ExpressionType::kNone;

  /// The inequality constraint type detected by the solver.
  autodiff::ExpressionType inequalityConstraintType =
      autodiff::ExpressionType::kNone;

  /// The solver's exit condition.
  SolverExitCondition exitCondition = SolverExitCondition::kOk;
};

}  // namespace sleipnir
