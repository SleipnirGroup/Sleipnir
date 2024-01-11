// Copyright (c) Sleipnir contributors

#include <vector>

#include <fmt/core.h>
#include <gtest/gtest.h>
#include <sleipnir/optimization/OptimizationProblem.hpp>

#include "CmdlineArguments.hpp"
#include "Range.hpp"

TEST(NonlinearProblemTest, Quartic) {
  sleipnir::OptimizationProblem problem;

  auto x = problem.DecisionVariable();
  x.SetValue(20.0);

  problem.Minimize(sleipnir::pow(x, 4));

  problem.SubjectTo(x >= 1);

  auto status =
      problem.Solve({.diagnostics = Argv().Contains("--enable-diagnostics")});

  EXPECT_EQ(sleipnir::ExpressionType::kNonlinear, status.costFunctionType);
  EXPECT_EQ(sleipnir::ExpressionType::kNone, status.equalityConstraintType);
  EXPECT_EQ(sleipnir::ExpressionType::kLinear, status.inequalityConstraintType);
  EXPECT_EQ(sleipnir::SolverExitCondition::kSuccess, status.exitCondition);

  EXPECT_NEAR(1.0, x.Value(), 1e-6);
}

TEST(NonlinearProblemTest, RosenbrockWithCubicAndLineConstraint) {
  // https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization
  for (auto x0 : Range(-1.5, 1.5, 0.1)) {
    for (auto y0 : Range(-0.5, 2.5, 0.1)) {
      sleipnir::OptimizationProblem problem;

      auto x = problem.DecisionVariable();
      x.SetValue(x0);
      auto y = problem.DecisionVariable();
      y.SetValue(y0);

      problem.Minimize(sleipnir::pow(1 - x, 2) +
                       100 * sleipnir::pow(y - sleipnir::pow(x, 2), 2));

      problem.SubjectTo(sleipnir::pow(x - 1, 3) - y + 1 <= 0);
      problem.SubjectTo(x + y - 2 <= 0);

      auto status = problem.Solve(
          {.diagnostics = Argv().Contains("--enable-diagnostics")});

      EXPECT_EQ(sleipnir::ExpressionType::kNonlinear, status.costFunctionType);
      EXPECT_EQ(sleipnir::ExpressionType::kNone, status.equalityConstraintType);
      EXPECT_EQ(sleipnir::ExpressionType::kNonlinear,
                status.inequalityConstraintType);
      EXPECT_EQ(sleipnir::SolverExitCondition::kSuccess, status.exitCondition);

      auto Near = [](double expected, double actual, double tolerance) {
        return std::abs(expected - actual) < tolerance;
      };

      // Local minimum at (0.0, 0.0)
      // Global minimum at (1.0, 1.0)
      EXPECT_TRUE(Near(0.0, x.Value(), 1e-2) || Near(1.0, x.Value(), 1e-2))
          << fmt::format("  (x₀, y₀) = ({}, {})\n", x0, y0)
          << fmt::format("  x.Value(0) = {}", x.Value());
      EXPECT_TRUE(Near(0.0, y.Value(), 1e-2) || Near(1.0, y.Value(), 1e-2))
          << fmt::format("  (x₀, y₀) = ({}, {})\n", x0, y0)
          << fmt::format("  y.Value(0) = {}", y.Value());
    }
  }
}

TEST(NonlinearProblemTest, RosenbrockWithDiskConstraint) {
  // https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization
  for (auto x0 : Range(-1.5, 1.5, 0.1)) {
    for (auto y0 : Range(-1.5, 1.5, 0.1)) {
      sleipnir::OptimizationProblem problem;

      auto x = problem.DecisionVariable();
      x.SetValue(x0);
      auto y = problem.DecisionVariable();
      y.SetValue(y0);

      problem.Minimize(sleipnir::pow(1 - x, 2) +
                       100 * sleipnir::pow(y - sleipnir::pow(x, 2), 2));

      problem.SubjectTo(sleipnir::pow(x, 2) + sleipnir::pow(y, 2) <= 2);

      auto status = problem.Solve(
          {.diagnostics = Argv().Contains("--enable-diagnostics")});

      EXPECT_EQ(sleipnir::ExpressionType::kNonlinear, status.costFunctionType);
      EXPECT_EQ(sleipnir::ExpressionType::kNone, status.equalityConstraintType);
      EXPECT_EQ(sleipnir::ExpressionType::kQuadratic,
                status.inequalityConstraintType);
      EXPECT_EQ(sleipnir::SolverExitCondition::kSuccess, status.exitCondition);

      EXPECT_NEAR(1.0, x.Value(), 1e-1)
          << fmt::format("  (x₀, y₀) = ({}, {})\n", x0, y0)
          << fmt::format("  x.Value(0) = {}", x.Value());
      EXPECT_NEAR(1.0, y.Value(), 1e-1)
          << fmt::format("  (x₀, y₀) = ({}, {})\n", x0, y0)
          << fmt::format("  x.Value(0) = {}", x.Value());
    }
  }
}

TEST(NonlinearProblemTest, NarrowFeasibleRegion) {
  sleipnir::OptimizationProblem problem;

  auto x = problem.DecisionVariable();
  x.SetValue(20.0);

  auto y = problem.DecisionVariable();
  y.SetValue(50.0);

  problem.Minimize(sleipnir::sqrt(x * x + y * y));

  problem.SubjectTo(y == -x + 5.0);

  auto status =
      problem.Solve({.diagnostics = Argv().Contains("--enable-diagnostics")});

  EXPECT_EQ(sleipnir::ExpressionType::kNonlinear, status.costFunctionType);
  EXPECT_EQ(sleipnir::ExpressionType::kLinear, status.equalityConstraintType);
  EXPECT_EQ(sleipnir::ExpressionType::kNone, status.inequalityConstraintType);
  EXPECT_EQ(sleipnir::SolverExitCondition::kSuccess, status.exitCondition);

  EXPECT_NEAR(2.5, x.Value(), 1e-2);
  EXPECT_NEAR(2.5, y.Value(), 1e-2);
}
