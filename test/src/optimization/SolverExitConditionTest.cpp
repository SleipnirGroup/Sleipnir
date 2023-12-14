// Copyright (c) Sleipnir contributors

#include <gtest/gtest.h>
#include <sleipnir/optimization/OptimizationProblem.hpp>

#include "CmdlineArguments.hpp"

// These tests ensure coverage of the off-nominal solver exit conditions

TEST(SolverExitConditionTest, CallbackRequestedStop) {
  sleipnir::OptimizationProblem problem;

  auto x = problem.DecisionVariable();
  problem.Minimize(x * x);

  problem.Callback([](const sleipnir::SolverIterationInfo&) {});
  auto status =
      problem.Solve({.diagnostics = CmdlineArgPresent(kEnableDiagnostics)});

  EXPECT_EQ(sleipnir::ExpressionType::kQuadratic, status.costFunctionType);
  EXPECT_EQ(sleipnir::ExpressionType::kNone, status.equalityConstraintType);
  EXPECT_EQ(sleipnir::ExpressionType::kNone, status.inequalityConstraintType);
  EXPECT_EQ(sleipnir::SolverExitCondition::kSuccess, status.exitCondition);

  problem.Callback([](const sleipnir::SolverIterationInfo&) { return false; });
  status =
      problem.Solve({.diagnostics = CmdlineArgPresent(kEnableDiagnostics)});

  EXPECT_EQ(sleipnir::ExpressionType::kQuadratic, status.costFunctionType);
  EXPECT_EQ(sleipnir::ExpressionType::kNone, status.equalityConstraintType);
  EXPECT_EQ(sleipnir::ExpressionType::kNone, status.inequalityConstraintType);
  EXPECT_EQ(sleipnir::SolverExitCondition::kSuccess, status.exitCondition);

  problem.Callback([](const sleipnir::SolverIterationInfo&) { return true; });
  status =
      problem.Solve({.diagnostics = CmdlineArgPresent(kEnableDiagnostics)});

  EXPECT_EQ(sleipnir::ExpressionType::kQuadratic, status.costFunctionType);
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

  auto status =
      problem.Solve({.diagnostics = CmdlineArgPresent(kEnableDiagnostics)});

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

    auto status =
        problem.Solve({.diagnostics = CmdlineArgPresent(kEnableDiagnostics)});

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

    auto status =
        problem.Solve({.diagnostics = CmdlineArgPresent(kEnableDiagnostics)});

    EXPECT_EQ(sleipnir::ExpressionType::kNone, status.costFunctionType);
    EXPECT_EQ(sleipnir::ExpressionType::kNone, status.equalityConstraintType);
    EXPECT_EQ(sleipnir::ExpressionType::kLinear,
              status.inequalityConstraintType);
    EXPECT_EQ(sleipnir::SolverExitCondition::kLocallyInfeasible,
              status.exitCondition);
  }
}

TEST(SolverExitConditionTest, DivergingIterates) {
  sleipnir::OptimizationProblem problem;

  auto x = problem.DecisionVariable();
  problem.Minimize(x);

  auto status =
      problem.Solve({.diagnostics = CmdlineArgPresent(kEnableDiagnostics)});

  EXPECT_EQ(sleipnir::ExpressionType::kLinear, status.costFunctionType);
  EXPECT_EQ(sleipnir::ExpressionType::kNone, status.equalityConstraintType);
  EXPECT_EQ(sleipnir::ExpressionType::kNone, status.inequalityConstraintType);
  EXPECT_EQ(sleipnir::SolverExitCondition::kDivergingIterates,
            status.exitCondition);
}

TEST(SolverExitConditionTest, MaxIterationsExceeded) {
  sleipnir::OptimizationProblem problem;

  auto x = problem.DecisionVariable();
  problem.Minimize(x * x);

  auto status =
      problem.Solve({.maxIterations = 0,
                     .diagnostics = CmdlineArgPresent(kEnableDiagnostics)});

  EXPECT_EQ(sleipnir::ExpressionType::kQuadratic, status.costFunctionType);
  EXPECT_EQ(sleipnir::ExpressionType::kNone, status.equalityConstraintType);
  EXPECT_EQ(sleipnir::ExpressionType::kNone, status.inequalityConstraintType);
  EXPECT_EQ(sleipnir::SolverExitCondition::kMaxIterationsExceeded,
            status.exitCondition);
}

TEST(SolverExitConditionTest, Timeout) {
  using namespace std::chrono_literals;

  sleipnir::OptimizationProblem problem;

  auto x = problem.DecisionVariable();
  problem.Minimize(x * x);

  auto status = problem.Solve(
      {.timeout = 0s, .diagnostics = CmdlineArgPresent(kEnableDiagnostics)});

  EXPECT_EQ(sleipnir::ExpressionType::kQuadratic, status.costFunctionType);
  EXPECT_EQ(sleipnir::ExpressionType::kNone, status.equalityConstraintType);
  EXPECT_EQ(sleipnir::ExpressionType::kNone, status.inequalityConstraintType);
  EXPECT_EQ(sleipnir::SolverExitCondition::kMaxWallClockTimeExceeded,
            status.exitCondition);
}
