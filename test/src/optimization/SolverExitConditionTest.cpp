// Copyright (c) Sleipnir contributors

#include <gtest/gtest.h>
#include <sleipnir/optimization/OptimizationProblem.hpp>

// These tests ensure coverage of the off-nominal solver exit conditions

TEST(SolverExitConditionTest, CallbackRequestedStop) {
  sleipnir::OptimizationProblem problem;

  auto x = problem.DecisionVariable();
  x.SetValue(0.0);
  problem.Minimize(x);

  problem.Callback([](const sleipnir::SolverIterationInfo&) { return true; });

  auto status = problem.Solve({.diagnostics = true});

  EXPECT_EQ(sleipnir::ExpressionType::kLinear, status.costFunctionType);
  EXPECT_EQ(sleipnir::ExpressionType::kNone, status.equalityConstraintType);
  EXPECT_EQ(sleipnir::ExpressionType::kNone, status.inequalityConstraintType);
  EXPECT_EQ(sleipnir::SolverExitCondition::kCallbackRequestedStop,
            status.exitCondition);
}

TEST(SolverExitConditionTest, TooFewDOFs) {
  sleipnir::OptimizationProblem problem;

  auto x = problem.DecisionVariable();
  auto y = problem.DecisionVariable();
  auto z = problem.DecisionVariable();

  problem.SubjectTo(x == 1);
  problem.SubjectTo(x == 2);
  problem.SubjectTo(y == 1);
  problem.SubjectTo(z == 1);

  auto status = problem.Solve({.diagnostics = true});

  EXPECT_EQ(sleipnir::ExpressionType::kNone, status.costFunctionType);
  EXPECT_EQ(sleipnir::ExpressionType::kLinear, status.equalityConstraintType);
  EXPECT_EQ(sleipnir::ExpressionType::kNone, status.inequalityConstraintType);
  EXPECT_EQ(sleipnir::SolverExitCondition::kTooFewDOFs, status.exitCondition);
}

TEST(SolverExitConditionTest, LocallyInfeasible) {
  // Equality constraints
  {
    sleipnir::OptimizationProblem problem;

    auto x = problem.DecisionVariable();
    auto y = problem.DecisionVariable();
    auto z = problem.DecisionVariable();

    problem.SubjectTo(x == y + 1);
    problem.SubjectTo(y == z + 1);
    problem.SubjectTo(z == x + 1);

    auto status = problem.Solve({.diagnostics = true});

    EXPECT_EQ(sleipnir::ExpressionType::kNone, status.costFunctionType);
    EXPECT_EQ(sleipnir::ExpressionType::kLinear, status.equalityConstraintType);
    EXPECT_EQ(sleipnir::ExpressionType::kNone, status.inequalityConstraintType);
    EXPECT_EQ(sleipnir::SolverExitCondition::kLocallyInfeasible,
              status.exitCondition);
  }

  // Inequality constraints
  {
    sleipnir::OptimizationProblem problem;

    auto x = problem.DecisionVariable();
    auto y = problem.DecisionVariable();
    auto z = problem.DecisionVariable();

    problem.SubjectTo(x >= y + 1);
    problem.SubjectTo(y >= z + 1);
    problem.SubjectTo(z >= x + 1);

    auto status = problem.Solve({.diagnostics = true});

    EXPECT_EQ(sleipnir::ExpressionType::kNone, status.costFunctionType);
    EXPECT_EQ(sleipnir::ExpressionType::kNone, status.equalityConstraintType);
    EXPECT_EQ(sleipnir::ExpressionType::kLinear,
              status.inequalityConstraintType);
    EXPECT_EQ(sleipnir::SolverExitCondition::kLocallyInfeasible,
              status.exitCondition);
  }
}

TEST(SolverExitConditionTest, MaxIterationsExceeded) {
  sleipnir::OptimizationProblem problem;

  auto x = problem.DecisionVariable();
  x.SetValue(0.0);
  problem.Minimize(x);

  auto status = problem.Solve({.maxIterations = 0, .diagnostics = true});

  EXPECT_EQ(sleipnir::ExpressionType::kLinear, status.costFunctionType);
  EXPECT_EQ(sleipnir::ExpressionType::kNone, status.equalityConstraintType);
  EXPECT_EQ(sleipnir::ExpressionType::kNone, status.inequalityConstraintType);
  EXPECT_EQ(sleipnir::SolverExitCondition::kMaxIterationsExceeded,
            status.exitCondition);
}

TEST(SolverExitConditionTest, Timeout) {
  using namespace std::chrono_literals;

  sleipnir::OptimizationProblem problem;

  auto x = problem.DecisionVariable();
  x.SetValue(0.0);
  problem.Minimize(x);

  auto status = problem.Solve({.timeout = 0s, .diagnostics = true});

  EXPECT_EQ(sleipnir::ExpressionType::kLinear, status.costFunctionType);
  EXPECT_EQ(sleipnir::ExpressionType::kNone, status.equalityConstraintType);
  EXPECT_EQ(sleipnir::ExpressionType::kNone, status.inequalityConstraintType);
  EXPECT_EQ(sleipnir::SolverExitCondition::kMaxWallClockTimeExceeded,
            status.exitCondition);
}
