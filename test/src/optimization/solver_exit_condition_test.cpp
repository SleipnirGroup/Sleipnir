// Copyright (c) Sleipnir contributors

#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "catch_string_converters.hpp"

// These tests ensure coverage of the off-nominal solver exit conditions

TEST_CASE("SolverExitCondition - Callback requested stop",
          "[SolverExitCondition]") {
  slp::Problem problem;

  auto x = problem.decision_variable();
  problem.minimize(x * x);

  problem.add_callback([](const slp::SolverIterationInfo&) {});
  CHECK(problem.solve({.diagnostics = true}).exit_condition ==
        slp::SolverExitCondition::SUCCESS);

  problem.add_callback([](const slp::SolverIterationInfo&) { return false; });
  CHECK(problem.solve({.diagnostics = true}).exit_condition ==
        slp::SolverExitCondition::SUCCESS);

  problem.add_callback([](const slp::SolverIterationInfo&) { return true; });
  CHECK(problem.solve({.diagnostics = true}).exit_condition ==
        slp::SolverExitCondition::CALLBACK_REQUESTED_STOP);

  problem.clear_callbacks();
  problem.add_callback([](const slp::SolverIterationInfo&) { return false; });
  CHECK(problem.solve({.diagnostics = true}).exit_condition ==
        slp::SolverExitCondition::SUCCESS);
}

TEST_CASE("SolverExitCondition - Too few DOFs", "[SolverExitCondition]") {
  slp::Problem problem;

  auto x = problem.decision_variable();
  auto y = problem.decision_variable();
  auto z = problem.decision_variable();

  problem.subject_to(x == 1);
  problem.subject_to(x == 2);
  problem.subject_to(y == 1);
  problem.subject_to(z == 1);

  auto status = problem.solve({.diagnostics = true});

  CHECK(status.cost_function_type == slp::ExpressionType::NONE);
  CHECK(status.equality_constraint_type == slp::ExpressionType::LINEAR);
  CHECK(status.inequality_constraint_type == slp::ExpressionType::NONE);
  CHECK(status.exit_condition == slp::SolverExitCondition::TOO_FEW_DOFS);
}

TEST_CASE("SolverExitCondition - Locally infeasible", "[SolverExitCondition]") {
  // Equality constraints
  {
    slp::Problem problem;

    auto x = problem.decision_variable();
    auto y = problem.decision_variable();
    auto z = problem.decision_variable();

    problem.subject_to(x == y + 1);
    problem.subject_to(y == z + 1);
    problem.subject_to(z == x + 1);

    auto status = problem.solve({.diagnostics = true});

    CHECK(status.cost_function_type == slp::ExpressionType::NONE);
    CHECK(status.equality_constraint_type == slp::ExpressionType::LINEAR);
    CHECK(status.inequality_constraint_type == slp::ExpressionType::NONE);
    CHECK(status.exit_condition ==
          slp::SolverExitCondition::LOCALLY_INFEASIBLE);
  }

  // Inequality constraints
  {
    slp::Problem problem;

    auto x = problem.decision_variable();
    auto y = problem.decision_variable();
    auto z = problem.decision_variable();

    problem.subject_to(x >= y + 1);
    problem.subject_to(y >= z + 1);
    problem.subject_to(z >= x + 1);

    auto status = problem.solve({.diagnostics = true});

    CHECK(status.cost_function_type == slp::ExpressionType::NONE);
    CHECK(status.equality_constraint_type == slp::ExpressionType::NONE);
    CHECK(status.inequality_constraint_type == slp::ExpressionType::LINEAR);
    CHECK(status.exit_condition ==
          slp::SolverExitCondition::LOCALLY_INFEASIBLE);
  }
}

TEST_CASE("SolverExitCondition - Nonfinite initial cost or constraints",
          "[SolverExitCondition]") {
  slp::Problem problem;

  auto x = problem.decision_variable();
  x.set_value(-1.0);
  problem.minimize(slp::sqrt(x));

  auto status = problem.solve({.diagnostics = true});

  CHECK(status.cost_function_type == slp::ExpressionType::NONLINEAR);
  CHECK(status.equality_constraint_type == slp::ExpressionType::NONE);
  CHECK(status.inequality_constraint_type == slp::ExpressionType::NONE);
  CHECK(status.exit_condition ==
        slp::SolverExitCondition::NONFINITE_INITIAL_COST_OR_CONSTRAINTS);
}

TEST_CASE("SolverExitCondition - Diverging iterates", "[SolverExitCondition]") {
  slp::Problem problem;

  auto x = problem.decision_variable();
  problem.minimize(x);

  auto status = problem.solve({.diagnostics = true});

  CHECK(status.cost_function_type == slp::ExpressionType::LINEAR);
  CHECK(status.equality_constraint_type == slp::ExpressionType::NONE);
  CHECK(status.inequality_constraint_type == slp::ExpressionType::NONE);
  CHECK(status.exit_condition == slp::SolverExitCondition::DIVERGING_ITERATES);
}

TEST_CASE("SolverExitCondition - Max iterations exceeded",
          "[SolverExitCondition]") {
  slp::Problem problem;

  auto x = problem.decision_variable();
  problem.minimize(x * x);

  auto status = problem.solve({.max_iterations = 0, .diagnostics = true});

  CHECK(status.cost_function_type == slp::ExpressionType::QUADRATIC);
  CHECK(status.equality_constraint_type == slp::ExpressionType::NONE);
  CHECK(status.inequality_constraint_type == slp::ExpressionType::NONE);
  CHECK(status.exit_condition ==
        slp::SolverExitCondition::MAX_ITERATIONS_EXCEEDED);
}

TEST_CASE("SolverExitCondition - Timeout", "[SolverExitCondition]") {
  using namespace std::chrono_literals;

  slp::Problem problem;

  auto x = problem.decision_variable();
  problem.minimize(x * x);

  auto status = problem.solve({.timeout = 0s, .diagnostics = true});

  CHECK(status.cost_function_type == slp::ExpressionType::QUADRATIC);
  CHECK(status.equality_constraint_type == slp::ExpressionType::NONE);
  CHECK(status.inequality_constraint_type == slp::ExpressionType::NONE);
  CHECK(status.exit_condition == slp::SolverExitCondition::TIMEOUT);
}
