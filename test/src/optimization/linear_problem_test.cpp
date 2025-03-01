// Copyright (c) Sleipnir contributors

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "catch_string_converters.hpp"

TEST_CASE("Problem - Maximize", "[Problem]") {
  slp::Problem problem;

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

  CHECK(problem.cost_function_type() == slp::ExpressionType::LINEAR);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  CHECK(x.value() == Catch::Approx(375.0).margin(1e-6));
  CHECK(y.value() == Catch::Approx(250.0).margin(1e-6));
}

TEST_CASE("Problem - Free variable", "[Problem]") {
  slp::Problem problem;

  auto x = problem.decision_variable(2);
  x[0].set_value(1.0);
  x[1].set_value(2.0);

  problem.subject_to(x[0] == 0);

  CHECK(problem.cost_function_type() == slp::ExpressionType::NONE);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::LINEAR);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  CHECK(x[0].value() == Catch::Approx(0.0).margin(1e-6));
  CHECK(x[1].value() == Catch::Approx(2.0).margin(1e-6));
}
