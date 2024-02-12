// Copyright (c) Sleipnir contributors

#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/OptimizationProblem.hpp>

// These tests ensure coverage of the off-nominal solver exit conditions

TEST_CASE("Callback requested stop", "[SolverExitCondition]") {
  sleipnir::OptimizationProblem problem;

  auto x = problem.DecisionVariable();
  problem.Minimize(x * x);

  problem.Callback([](const sleipnir::SolverIterationInfo&) {});
  auto status = problem.Solve({.diagnostics = true});

  CHECK(status.costFunctionType == sleipnir::ExpressionType::kQuadratic);
  CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kNone);
  CHECK(status.inequalityConstraintType == sleipnir::ExpressionType::kNone);
  CHECK(status.exitCondition == sleipnir::SolverExitCondition::kSuccess);

  problem.Callback([](const sleipnir::SolverIterationInfo&) { return false; });
  status = problem.Solve({.diagnostics = true});

  CHECK(status.costFunctionType == sleipnir::ExpressionType::kQuadratic);
  CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kNone);
  CHECK(status.inequalityConstraintType == sleipnir::ExpressionType::kNone);
  CHECK(status.exitCondition == sleipnir::SolverExitCondition::kSuccess);

  problem.Callback([](const sleipnir::SolverIterationInfo&) { return true; });
  status = problem.Solve({.diagnostics = true});

  CHECK(status.costFunctionType == sleipnir::ExpressionType::kQuadratic);
  CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kNone);
  CHECK(status.inequalityConstraintType == sleipnir::ExpressionType::kNone);
  CHECK(status.exitCondition ==
        sleipnir::SolverExitCondition::kCallbackRequestedStop);
}

TEST_CASE("Too few DOFs", "[SolverExitCondition]") {
  sleipnir::OptimizationProblem problem;

  auto x = problem.DecisionVariable();
  auto y = problem.DecisionVariable();
  auto z = problem.DecisionVariable();

  problem.SubjectTo(x == 1);
  problem.SubjectTo(x == 2);
  problem.SubjectTo(y == 1);
  problem.SubjectTo(z == 1);

  auto status = problem.Solve({.diagnostics = true});

  CHECK(status.costFunctionType == sleipnir::ExpressionType::kNone);
  CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kLinear);
  CHECK(status.inequalityConstraintType == sleipnir::ExpressionType::kNone);
  CHECK(status.exitCondition == sleipnir::SolverExitCondition::kTooFewDOFs);
}

TEST_CASE("Locally infeasible", "[SolverExitCondition]") {
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

    CHECK(status.costFunctionType == sleipnir::ExpressionType::kNone);
    CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kLinear);
    CHECK(status.inequalityConstraintType == sleipnir::ExpressionType::kNone);
    CHECK(status.exitCondition ==
          sleipnir::SolverExitCondition::kLocallyInfeasible);
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

    CHECK(status.costFunctionType == sleipnir::ExpressionType::kNone);
    CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kNone);
    CHECK(status.inequalityConstraintType == sleipnir::ExpressionType::kLinear);
    CHECK(status.exitCondition ==
          sleipnir::SolverExitCondition::kLocallyInfeasible);
  }
}

TEST_CASE("Diverging iterates", "[SolverExitCondition]") {
  sleipnir::OptimizationProblem problem;

  auto x = problem.DecisionVariable();
  problem.Minimize(x);

  auto status = problem.Solve({.diagnostics = true});

  CHECK(status.costFunctionType == sleipnir::ExpressionType::kLinear);
  CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kNone);
  CHECK(status.inequalityConstraintType == sleipnir::ExpressionType::kNone);
  CHECK(status.exitCondition ==
        sleipnir::SolverExitCondition::kDivergingIterates);
}

TEST_CASE("Max iterations exceeded", "[SolverExitCondition]") {
  sleipnir::OptimizationProblem problem;

  auto x = problem.DecisionVariable();
  problem.Minimize(x * x);

  auto status = problem.Solve({.maxIterations = 0, .diagnostics = true});

  CHECK(status.costFunctionType == sleipnir::ExpressionType::kQuadratic);
  CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kNone);
  CHECK(status.inequalityConstraintType == sleipnir::ExpressionType::kNone);
  CHECK(status.exitCondition ==
        sleipnir::SolverExitCondition::kMaxIterationsExceeded);
}

TEST_CASE("Timeout", "[SolverExitCondition]") {
  using namespace std::chrono_literals;

  sleipnir::OptimizationProblem problem;

  auto x = problem.DecisionVariable();
  problem.Minimize(x * x);

  auto status = problem.Solve({.timeout = 0s, .diagnostics = true});

  CHECK(status.costFunctionType == sleipnir::ExpressionType::kQuadratic);
  CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kNone);
  CHECK(status.inequalityConstraintType == sleipnir::ExpressionType::kNone);
  CHECK(status.exitCondition ==
        sleipnir::SolverExitCondition::kMaxWallClockTimeExceeded);
}
