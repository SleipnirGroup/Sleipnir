// Copyright (c) Sleipnir contributors

#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/OptimizationProblem.hpp>

#include "CatchStringConverters.hpp"

TEST_CASE("TrivialProblem - Empty", "[TrivialProblem]") {
  sleipnir::OptimizationProblem problem;

  auto status = problem.Solve({.diagnostics = true});

  CHECK(status.costFunctionType == sleipnir::ExpressionType::kNone);
  CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kNone);
  CHECK(status.inequalityConstraintType == sleipnir::ExpressionType::kNone);
  CHECK(status.exitCondition == sleipnir::SolverExitCondition::kSuccess);
}

TEST_CASE("TrivialProblem - No cost, unconstrained", "[TrivialProblem]") {
  {
    sleipnir::OptimizationProblem problem;

    auto X = problem.DecisionVariable(2, 3);

    auto status = problem.Solve({.diagnostics = true});

    CHECK(status.costFunctionType == sleipnir::ExpressionType::kNone);
    CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kNone);
    CHECK(status.inequalityConstraintType == sleipnir::ExpressionType::kNone);
    CHECK(status.exitCondition == sleipnir::SolverExitCondition::kSuccess);

    for (int row = 0; row < X.Rows(); ++row) {
      for (int col = 0; col < X.Cols(); ++col) {
        CHECK(X.Value(row, col) == 0.0);
      }
    }
  }

  {
    sleipnir::OptimizationProblem problem;

    auto X = problem.DecisionVariable(2, 3);
    X.SetValue(Eigen::Matrix<double, 2, 3>{{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}});

    auto status = problem.Solve({.diagnostics = true});

    CHECK(status.costFunctionType == sleipnir::ExpressionType::kNone);
    CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kNone);
    CHECK(status.inequalityConstraintType == sleipnir::ExpressionType::kNone);
    CHECK(status.exitCondition == sleipnir::SolverExitCondition::kSuccess);

    for (int row = 0; row < X.Rows(); ++row) {
      for (int col = 0; col < X.Cols(); ++col) {
        CHECK(X.Value(row, col) == 1.0);
      }
    }
  }
}
