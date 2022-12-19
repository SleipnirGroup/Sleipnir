// Copyright (c) Sleipnir contributors

#include <gtest/gtest.h>
#include <sleipnir/optimization/OptimizationProblem.hpp>

TEST(LinearProblemTest, Maximize) {
  sleipnir::OptimizationProblem problem;

  auto x = problem.DecisionVariable();
  x = 1.0;

  auto y = problem.DecisionVariable();
  y = 1.0;

  problem.Maximize(50 * x + 40 * y);

  problem.SubjectTo(x + 1.5 * y <= 750);
  problem.SubjectTo(2 * x + 3 * y <= 1500);
  problem.SubjectTo(2 * x + y <= 1000);
  problem.SubjectTo(x >= 0);
  problem.SubjectTo(y >= 0);

  auto status = problem.Solve({.diagnostics = true});

  EXPECT_EQ(sleipnir::ExpressionType::kLinear, status.costFunctionType);
  EXPECT_EQ(sleipnir::ExpressionType::kNone, status.equalityConstraintType);
  EXPECT_EQ(sleipnir::ExpressionType::kLinear, status.inequalityConstraintType);
  EXPECT_EQ(sleipnir::SolverExitCondition::kOk, status.exitCondition);

  EXPECT_NEAR(375.0, x.Value(0), 1e-6);
  EXPECT_NEAR(250.0, y.Value(0), 1e-6);
}
