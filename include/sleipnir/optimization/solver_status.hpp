// Copyright (c) Sleipnir contributors

#pragma once

#include "sleipnir/autodiff/expression_type.hpp"
#include "sleipnir/optimization/solver_exit_condition.hpp"
#include "sleipnir/util/symbol_exports.hpp"

namespace slp {

/**
 * Return value of OptimizationProblem::Solve() containing the cost function and
 * constraint types and solver's exit condition.
 */
struct SLEIPNIR_DLLEXPORT SolverStatus {
  /// The cost function type detected by the solver.
  ExpressionType cost_function_type = ExpressionType::NONE;

  /// The equality constraint type detected by the solver.
  ExpressionType equality_constraint_type = ExpressionType::NONE;

  /// The inequality constraint type detected by the solver.
  ExpressionType inequality_constraint_type = ExpressionType::NONE;

  /// The solver's exit condition.
  SolverExitCondition exit_condition = SolverExitCondition::SUCCESS;

  /// The solution's cost.
  double cost = 0.0;
};

}  // namespace slp
