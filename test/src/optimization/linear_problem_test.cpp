// Copyright (c) Sleipnir contributors

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/optimization_problem.hpp>

#include "catch_string_converters.hpp"

TEST_CASE("linear_problem - Maximize", "[linear_problem]") {
  slp::OptimizationProblem problem;

  auto x = problem.decision_variable();
  x.set_value(1.0);

  auto y = problem.decision_variable();
  y.set_value(1.0);

  problem.maximize(50 * x + 40 * y);

  problem.subject_to(x + 1.5 * y <= 750);
  problem.subject_to(2 * x + 3 * y <= 1500);
  problem.subject_to(2 * x + y <= 1000);
  problem.subject_to(x >= 0);
  problem.subject_to(y >= 0);

  auto status = problem.solve({.diagnostics = true});

  CHECK(status.cost_function_type == slp::ExpressionType::LINEAR);
  CHECK(status.equality_constraint_type == slp::ExpressionType::NONE);
  CHECK(status.inequality_constraint_type == slp::ExpressionType::LINEAR);
  CHECK(status.exit_condition == slp::SolverExitCondition::SUCCESS);

  CHECK(x.value() == Catch::Approx(375.0).margin(1e-6));
  CHECK(y.value() == Catch::Approx(250.0).margin(1e-6));
}

TEST_CASE("linear_problem - Free variable", "[linear_problem]") {
  slp::OptimizationProblem problem;

  auto x = problem.decision_variable(2);
  x[0].set_value(1.0);
  x[1].set_value(2.0);

  problem.subject_to(x[0] == 0);

  auto status = problem.solve({.diagnostics = true});

  CHECK(status.cost_function_type == slp::ExpressionType::NONE);
  CHECK(status.equality_constraint_type == slp::ExpressionType::LINEAR);
  CHECK(status.inequality_constraint_type == slp::ExpressionType::NONE);
  CHECK(status.exit_condition == slp::SolverExitCondition::SUCCESS);

  CHECK(x[0].value() == Catch::Approx(0.0).margin(1e-6));
  CHECK(x[1].value() == Catch::Approx(2.0).margin(1e-6));
}
