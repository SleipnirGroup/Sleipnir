// Copyright (c) Sleipnir contributors

#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/multistart.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "catch_string_converters.hpp"

TEST_CASE("multistart - Mishra's Bird function", "[nonlinear_problem]") {
  struct decision_variables {
    double x;
    double y;
  };

  auto Solve = [](const decision_variables& input)
      -> slp::MultistartResult<decision_variables> {
    slp::Problem problem;

    auto x = problem.decision_variable();
    x.set_value(input.x);
    auto y = problem.decision_variable();
    y.set_value(input.y);

    // https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization
    problem.minimize(slp::sin(y) * slp::exp(slp::pow(1 - slp::cos(x), 2)) +
                     slp::cos(x) * slp::exp(slp::pow(1 - slp::sin(y), 2)) +
                     slp::pow(x - y, 2));

    problem.subject_to(slp::pow(x + 5, 2) + slp::pow(y + 5, 2) < 25);

    return {problem.solve(), decision_variables{x.value(), y.value()}};
  };

  auto [status, variables] = slp::Multistart<decision_variables>(
      Solve,
      std::vector{decision_variables{-3, -8}, decision_variables{-3, -1.5}});

  CHECK(status.cost_function_type == slp::ExpressionType::NONLINEAR);
  CHECK(status.equality_constraint_type == slp::ExpressionType::NONE);
  CHECK(status.inequality_constraint_type == slp::ExpressionType::QUADRATIC);
  CHECK(status.exit_condition == slp::SolverExitCondition::SUCCESS);

  CHECK(variables.x == Catch::Approx(-3.130246803458174).margin(1e-15));
  CHECK(variables.y == Catch::Approx(-1.5821421769364057).margin(1e-15));
}
