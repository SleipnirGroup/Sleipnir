// Copyright (c) Joshua Nichols and Tyler Veness

#include <gtest/gtest.h>
#include <sleipnir/optimization/OptimizationProblem.hpp>

// These tests ensure coverage of the off-nominal solver exit conditions

TEST(SolverExitConditionTest, TooFewDOFs) {
  sleipnir::OptimizationProblem problem;

  auto x = problem.DecisionVariable();
  auto y = problem.DecisionVariable();
  auto z = problem.DecisionVariable();

  problem.SubjectTo(x == 1);
  problem.SubjectTo(x == 2);
  problem.SubjectTo(y == 1);
  problem.SubjectTo(z == 1);

  sleipnir::SolverConfig config;
  config.diagnostics = true;

  auto status = problem.Solve(config);

  EXPECT_EQ(sleipnir::autodiff::ExpressionType::kNone, status.costFunctionType);
  EXPECT_EQ(sleipnir::autodiff::ExpressionType::kLinear,
            status.equalityConstraintType);
  EXPECT_EQ(sleipnir::autodiff::ExpressionType::kNone,
            status.inequalityConstraintType);
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

    sleipnir::SolverConfig config;
    config.diagnostics = true;

    auto status = problem.Solve(config);

    EXPECT_EQ(sleipnir::autodiff::ExpressionType::kNone,
              status.costFunctionType);
    EXPECT_EQ(sleipnir::autodiff::ExpressionType::kLinear,
              status.equalityConstraintType);
    EXPECT_EQ(sleipnir::autodiff::ExpressionType::kNone,
              status.inequalityConstraintType);
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

    sleipnir::SolverConfig config;
    config.diagnostics = true;

    auto status = problem.Solve(config);

    EXPECT_EQ(sleipnir::autodiff::ExpressionType::kNone,
              status.costFunctionType);
    EXPECT_EQ(sleipnir::autodiff::ExpressionType::kNone,
              status.equalityConstraintType);
    EXPECT_EQ(sleipnir::autodiff::ExpressionType::kLinear,
              status.inequalityConstraintType);
    EXPECT_EQ(sleipnir::SolverExitCondition::kLocallyInfeasible,
              status.exitCondition);
  }
}

TEST(SolverExitConditionTest, MaxIterations) {
  sleipnir::OptimizationProblem problem;

  auto x = problem.DecisionVariable();
  x = 0.0;
  problem.Minimize(x);

  sleipnir::SolverConfig config;
  config.diagnostics = true;
  config.maxIterations = 0;

  auto status = problem.Solve(config);

  EXPECT_EQ(sleipnir::autodiff::ExpressionType::kLinear,
            status.costFunctionType);
  EXPECT_EQ(sleipnir::autodiff::ExpressionType::kNone,
            status.equalityConstraintType);
  EXPECT_EQ(sleipnir::autodiff::ExpressionType::kNone,
            status.inequalityConstraintType);
  EXPECT_EQ(sleipnir::SolverExitCondition::kMaxIterations,
            status.exitCondition);
}

TEST(SolverExitConditionTest, Timeout) {
  using namespace std::chrono_literals;

  sleipnir::OptimizationProblem problem;

  auto x = problem.DecisionVariable();
  x = 0.0;
  problem.Minimize(x);

  sleipnir::SolverConfig config;
  config.diagnostics = true;
  config.timeout = 0s;

  auto status = problem.Solve(config);

  EXPECT_EQ(sleipnir::autodiff::ExpressionType::kLinear,
            status.costFunctionType);
  EXPECT_EQ(sleipnir::autodiff::ExpressionType::kNone,
            status.equalityConstraintType);
  EXPECT_EQ(sleipnir::autodiff::ExpressionType::kNone,
            status.inequalityConstraintType);
  EXPECT_EQ(sleipnir::SolverExitCondition::kTimeout, status.exitCondition);
}
