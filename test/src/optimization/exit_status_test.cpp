// Copyright (c) Sleipnir contributors

#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "catch_string_converters.hpp"

// These tests ensure coverage of the off-nominal exit statuses

TEST_CASE("ExitStatus - Callback requested stop", "[ExitStatus]") {
  slp::Problem problem;

  auto x = problem.decision_variable();
  problem.minimize(x * x);

  problem.add_callback([](const slp::IterationInfo&) {});
  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  problem.add_callback([](const slp::IterationInfo&) { return false; });
  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  problem.add_callback([](const slp::IterationInfo&) { return true; });
  CHECK(problem.solve({.diagnostics = true}) ==
        slp::ExitStatus::CALLBACK_REQUESTED_STOP);

  problem.clear_callbacks();
  problem.add_callback([](const slp::IterationInfo&) { return false; });
  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);
}

TEST_CASE("ExitStatus - Too few DOFs", "[ExitStatus]") {
  slp::Problem problem;

  auto x = problem.decision_variable();
  auto y = problem.decision_variable();
  auto z = problem.decision_variable();

  problem.subject_to(x == 1);
  problem.subject_to(x == 2);
  problem.subject_to(y == 1);
  problem.subject_to(z == 1);

  CHECK(problem.cost_function_type() == slp::ExpressionType::NONE);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::LINEAR);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::TOO_FEW_DOFS);
}

TEST_CASE("ExitStatus - Locally infeasible", "[ExitStatus]") {
  // Equality constraints
  {
    slp::Problem problem;

    auto x = problem.decision_variable();
    auto y = problem.decision_variable();
    auto z = problem.decision_variable();

    problem.subject_to(x == y + 1);
    problem.subject_to(y == z + 1);
    problem.subject_to(z == x + 1);

    CHECK(problem.cost_function_type() == slp::ExpressionType::NONE);
    CHECK(problem.equality_constraint_type() == slp::ExpressionType::LINEAR);
    CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

    CHECK(problem.solve({.diagnostics = true}) ==
          slp::ExitStatus::LOCALLY_INFEASIBLE);
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

    CHECK(problem.cost_function_type() == slp::ExpressionType::NONE);
    CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
    CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

    CHECK(problem.solve({.diagnostics = true}) ==
          slp::ExitStatus::LOCALLY_INFEASIBLE);
  }
}

TEST_CASE("ExitStatus - Nonfinite initial cost or constraints",
          "[ExitStatus]") {
  slp::Problem problem;

  auto x = problem.decision_variable();
  x.set_value(-1.0);
  problem.minimize(slp::sqrt(x));

  CHECK(problem.cost_function_type() == slp::ExpressionType::NONLINEAR);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

  CHECK(problem.solve({.diagnostics = true}) ==
        slp::ExitStatus::NONFINITE_INITIAL_COST_OR_CONSTRAINTS);
}

TEST_CASE("ExitStatus - Diverging iterates", "[ExitStatus]") {
  slp::Problem problem;

  auto x = problem.decision_variable();
  problem.minimize(x);

  CHECK(problem.cost_function_type() == slp::ExpressionType::LINEAR);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

  CHECK(problem.solve({.diagnostics = true}) ==
        slp::ExitStatus::DIVERGING_ITERATES);
}

TEST_CASE("ExitStatus - Max iterations exceeded", "[ExitStatus]") {
  slp::Problem problem;

  auto x = problem.decision_variable();
  problem.minimize(x * x);

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

  CHECK(problem.solve({.max_iterations = 0, .diagnostics = true}) ==
        slp::ExitStatus::MAX_ITERATIONS_EXCEEDED);
}

TEST_CASE("ExitStatus - Timeout", "[ExitStatus]") {
  using namespace std::chrono_literals;

  slp::Problem problem;

  auto x = problem.decision_variable();
  problem.minimize(x * x);

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

  CHECK(problem.solve({.timeout = 0s, .diagnostics = true}) ==
        slp::ExitStatus::TIMEOUT);
}
