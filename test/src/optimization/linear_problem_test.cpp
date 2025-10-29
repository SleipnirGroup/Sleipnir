// Copyright (c) Sleipnir contributors

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "scalar_types_under_test.hpp"

TEMPLATE_TEST_CASE("Problem - Maximize", "[Problem]", SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.decision_variable();
  x.set_value(T(1));

  auto y = problem.decision_variable();
  y.set_value(T(1));

  problem.maximize(T(50) * x + T(40) * y);

  problem.subject_to(x + T(1.5) * y <= T(750));
  problem.subject_to(T(2) * x + T(3) * y <= T(1500));
  problem.subject_to(T(2) * x + y <= T(1000));
  problem.subject_to(x >= T(0));
  problem.subject_to(y >= T(0));

  CHECK(problem.cost_function_type() == slp::ExpressionType::LINEAR);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  CHECK_THAT(x.value(), WithinAbs(T(375), T(1e-6)));
  CHECK_THAT(y.value(), WithinAbs(T(250), T(1e-6)));
}

TEMPLATE_TEST_CASE("Problem - Free variable", "[Problem]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.decision_variable(2);
  x[0].set_value(T(1));
  x[1].set_value(T(2));

  problem.subject_to(x[0] == T(0));

  CHECK(problem.cost_function_type() == slp::ExpressionType::NONE);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::LINEAR);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  CHECK_THAT(x[0].value(), WithinAbs(T(0), T(1e-6)));
  CHECK_THAT(x[1].value(), WithinAbs(T(2), T(1e-6)));
}
