// Copyright (c) Sleipnir contributors

#include <format>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "catch_string_converters.hpp"
#include "range.hpp"

TEST_CASE("Problem - Quartic", "[Problem]") {
  slp::Problem problem;

  auto x = problem.decision_variable();
  x.set_value(20.0);

  problem.minimize(slp::pow(x, 4));

  problem.subject_to(x >= 1);

  CHECK(problem.cost_function_type() == slp::ExpressionType::NONLINEAR);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  CHECK(x.value() == Catch::Approx(1.0).margin(1e-6));
}

TEST_CASE("Problem - Rosenbrock with cubic and line constraint", "[Problem]") {
  // https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization
  for (auto x0 : range(-1.5, 1.5, 0.1)) {
    for (auto y0 : range(-0.5, 2.5, 0.1)) {
      slp::Problem problem;

      auto x = problem.decision_variable();
      x.set_value(x0);
      auto y = problem.decision_variable();
      y.set_value(y0);

      problem.minimize(slp::pow(1 - x, 2) +
                       100 * slp::pow(y - slp::pow(x, 2), 2));

      problem.subject_to(slp::pow(x - 1, 3) - y + 1 <= 0);
      problem.subject_to(x + y - 2 <= 0);

      CHECK(problem.cost_function_type() == slp::ExpressionType::NONLINEAR);
      CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
      CHECK(problem.inequality_constraint_type() ==
            slp::ExpressionType::NONLINEAR);

      CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

      auto Near = [](double expected, double actual, double tolerance) {
        return std::abs(expected - actual) < tolerance;
      };

      // Local minimum at (0.0, 0.0)
      // Global minimum at (1.0, 1.0)
      CHECK((Near(0.0, x.value(), 1e-2) || Near(1.0, x.value(), 1e-2)));
      INFO(std::format("  (x₀, y₀) = ({}, {})\n", x0, y0));
      INFO(std::format("  x.value(0) = {}", x.value()));
      CHECK((Near(0.0, y.value(), 1e-2) || Near(1.0, y.value(), 1e-2)));
      INFO(std::format("  (x₀, y₀) = ({}, {})\n", x0, y0));
      INFO(std::format("  y.value(0) = {}", y.value()));
    }
  }
}

TEST_CASE("Problem - Rosenbrock with disk constraint", "[Problem]") {
  // https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization
  for (auto x0 : range(-1.5, 1.5, 0.1)) {
    for (auto y0 : range(-1.5, 1.5, 0.1)) {
      slp::Problem problem;

      auto x = problem.decision_variable();
      x.set_value(x0);
      auto y = problem.decision_variable();
      y.set_value(y0);

      problem.minimize(slp::pow(1 - x, 2) +
                       100 * slp::pow(y - slp::pow(x, 2), 2));

      problem.subject_to(slp::pow(x, 2) + slp::pow(y, 2) <= 2);

      CHECK(problem.cost_function_type() == slp::ExpressionType::NONLINEAR);
      CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
      CHECK(problem.inequality_constraint_type() ==
            slp::ExpressionType::QUADRATIC);

      CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

      CHECK(x.value() == Catch::Approx(1.0).margin(1e-1));
      INFO(std::format("  (x₀, y₀) = ({}, {})\n", x0, y0));
      INFO(std::format("  x.value(0) = {}", x.value()));
      CHECK(y.value() == Catch::Approx(1.0).margin(1e-1));
      INFO(std::format("  (x₀, y₀) = ({}, {})\n", x0, y0));
      INFO(std::format("  x.value(0) = {}", x.value()));
    }
  }
}

TEST_CASE("Problem - Narrow feasible region", "[Problem]") {
  slp::Problem problem;

  auto x = problem.decision_variable();
  x.set_value(20.0);

  auto y = problem.decision_variable();
  y.set_value(50.0);

  problem.minimize(slp::sqrt(x * x + y * y));

  problem.subject_to(y == -x + 5.0);

  CHECK(problem.cost_function_type() == slp::ExpressionType::NONLINEAR);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::LINEAR);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

#if defined(__linux__) && defined(__aarch64__)
  // FIXME: Fails on Linux aarch64 with "diverging iterates"
  CHECK(problem.solve({.diagnostics = true}) ==
        slp::ExitStatus::DIVERGING_ITERATES);
  SKIP("Fails with \"diverging iterates\"");
#else
  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);
#endif

  CHECK(x.value() == Catch::Approx(2.5).margin(1e-2));
  CHECK(y.value() == Catch::Approx(2.5).margin(1e-2));
}
