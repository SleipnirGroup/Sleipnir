// Copyright (c) Sleipnir contributors

#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/optimization_problem.hpp>

#include "catch_string_converters.hpp"

// These tests ensure coverage of the off-nominal solver exit conditions

TEST_CASE("SolverExitCondition - Callback requested stop",
          "[SolverExitCondition]") {
  sleipnir::OptimizationProblem problem;

  auto x = problem.decision_variable();
  problem.minimize(x * x);

  problem.add_callback([](const sleipnir::SolverIterationInfo&) {});
  CHECK(problem.solve({.diagnostics = true}).exit_condition ==
        sleipnir::SolverExitCondition::SUCCESS);

  problem.add_callback(
      [](const sleipnir::SolverIterationInfo&) { return false; });
  CHECK(problem.solve({.diagnostics = true}).exit_condition ==
        sleipnir::SolverExitCondition::SUCCESS);

  problem.add_callback(
      [](const sleipnir::SolverIterationInfo&) { return true; });
  CHECK(problem.solve({.diagnostics = true}).exit_condition ==
        sleipnir::SolverExitCondition::CALLBACK_REQUESTED_STOP);

  problem.clear_callbacks();
  problem.add_callback(
      [](const sleipnir::SolverIterationInfo&) { return false; });
  CHECK(problem.solve({.diagnostics = true}).exit_condition ==
        sleipnir::SolverExitCondition::SUCCESS);
}

TEST_CASE("SolverExitCondition - Too few DOFs", "[SolverExitCondition]") {
  sleipnir::OptimizationProblem problem;

  auto x = problem.decision_variable();
  auto y = problem.decision_variable();
  auto z = problem.decision_variable();

  problem.subject_to(x == 1);
  problem.subject_to(x == 2);
  problem.subject_to(y == 1);
  problem.subject_to(z == 1);

  auto status = problem.solve({.diagnostics = true});

  CHECK(status.cost_function_type == sleipnir::ExpressionType::NONE);
  CHECK(status.equality_constraint_type == sleipnir::ExpressionType::LINEAR);
  CHECK(status.inequality_constraint_type == sleipnir::ExpressionType::NONE);
  CHECK(status.exit_condition == sleipnir::SolverExitCondition::TOO_FEW_DOFS);
}

TEST_CASE("SolverExitCondition - Locally infeasible", "[SolverExitCondition]") {
  // Equality constraints
  {
    sleipnir::OptimizationProblem problem;

    auto x = problem.decision_variable();
    auto y = problem.decision_variable();
    auto z = problem.decision_variable();

    problem.subject_to(x == y + 1);
    problem.subject_to(y == z + 1);
    problem.subject_to(z == x + 1);

    auto status = problem.solve({.diagnostics = true});

    CHECK(status.cost_function_type == sleipnir::ExpressionType::NONE);
    CHECK(status.equality_constraint_type == sleipnir::ExpressionType::LINEAR);
    CHECK(status.inequality_constraint_type == sleipnir::ExpressionType::NONE);
    CHECK(status.exit_condition ==
          sleipnir::SolverExitCondition::LOCALLY_INFEASIBLE);
  }

  // Inequality constraints
  {
    sleipnir::OptimizationProblem problem;

    auto x = problem.decision_variable();
    auto y = problem.decision_variable();
    auto z = problem.decision_variable();

    problem.subject_to(x >= y + 1);
    problem.subject_to(y >= z + 1);
    problem.subject_to(z >= x + 1);

    auto status = problem.solve({.diagnostics = true});

    CHECK(status.cost_function_type == sleipnir::ExpressionType::NONE);
    CHECK(status.equality_constraint_type == sleipnir::ExpressionType::NONE);
    CHECK(status.inequality_constraint_type ==
          sleipnir::ExpressionType::LINEAR);
    CHECK(status.exit_condition ==
          sleipnir::SolverExitCondition::LOCALLY_INFEASIBLE);
  }
}

TEST_CASE("SolverExitCondition - Nonfinite initial cost or constraints",
          "[SolverExitCondition]") {
  sleipnir::OptimizationProblem problem;

  auto x = problem.decision_variable();
  x.set_value(-1.0);
  problem.minimize(sleipnir::sqrt(x));

  auto status = problem.solve({.diagnostics = true});

  CHECK(status.cost_function_type == sleipnir::ExpressionType::NONLINEAR);
  CHECK(status.equality_constraint_type == sleipnir::ExpressionType::NONE);
  CHECK(status.inequality_constraint_type == sleipnir::ExpressionType::NONE);
  CHECK(status.exit_condition ==
        sleipnir::SolverExitCondition::NONFINITE_INITIAL_COST_OR_CONSTRAINTS);
}

TEST_CASE("SolverExitCondition - Diverging iterates", "[SolverExitCondition]") {
  sleipnir::OptimizationProblem problem;

  auto x = problem.decision_variable();
  problem.minimize(x);

  auto status = problem.solve({.diagnostics = true});

  CHECK(status.cost_function_type == sleipnir::ExpressionType::LINEAR);
  CHECK(status.equality_constraint_type == sleipnir::ExpressionType::NONE);
  CHECK(status.inequality_constraint_type == sleipnir::ExpressionType::NONE);
  CHECK(status.exit_condition ==
        sleipnir::SolverExitCondition::DIVERGING_ITERATES);
}

TEST_CASE("SolverExitCondition - Max iterations exceeded",
          "[SolverExitCondition]") {
  sleipnir::OptimizationProblem problem;

  auto x = problem.decision_variable();
  problem.minimize(x * x);

  auto status = problem.solve({.max_iterations = 0, .diagnostics = true});

  CHECK(status.cost_function_type == sleipnir::ExpressionType::QUADRATIC);
  CHECK(status.equality_constraint_type == sleipnir::ExpressionType::NONE);
  CHECK(status.inequality_constraint_type == sleipnir::ExpressionType::NONE);
  CHECK(status.exit_condition ==
        sleipnir::SolverExitCondition::MAX_ITERATIONS_EXCEEDED);
}

TEST_CASE("SolverExitCondition - Timeout", "[SolverExitCondition]") {
  using namespace std::chrono_literals;

  sleipnir::OptimizationProblem problem;

  auto x = problem.decision_variable();
  problem.minimize(x * x);

  auto status = problem.solve({.timeout = 0s, .diagnostics = true});

  CHECK(status.cost_function_type == sleipnir::ExpressionType::QUADRATIC);
  CHECK(status.equality_constraint_type == sleipnir::ExpressionType::NONE);
  CHECK(status.inequality_constraint_type == sleipnir::ExpressionType::NONE);
  CHECK(status.exit_condition == sleipnir::SolverExitCondition::TIMEOUT);
}
