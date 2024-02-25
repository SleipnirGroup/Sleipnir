// Copyright (c) Sleipnir contributors

#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fmt/core.h>
#include <sleipnir/optimization/OptimizationProblem.hpp>

#include "Range.hpp"

TEST_CASE("NonlinearProblem - Quartic", "[NonlinearProblem]") {
  sleipnir::OptimizationProblem problem;

  auto x = problem.DecisionVariable();
  x.SetValue(20.0);

  problem.Minimize(sleipnir::pow(x, 4));

  problem.SubjectTo(x >= 1);

  auto status = problem.Solve({.diagnostics = true});

  CHECK(status.costFunctionType == sleipnir::ExpressionType::kNonlinear);
  CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kNone);
  CHECK(status.inequalityConstraintType == sleipnir::ExpressionType::kLinear);
  CHECK(status.exitCondition == sleipnir::SolverExitCondition::kSuccess);

  CHECK(x.Value() == Catch::Approx(1.0).margin(1e-6));
}

TEST_CASE("NonlinearProblem - Rosenbrock with cubic and line constraint",
          "[NonlinearProblem]") {
  // https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization
  for (auto x0 : Range(-1.5, 1.5, 0.1)) {
    for (auto y0 : Range(-0.5, 2.5, 0.1)) {
      sleipnir::OptimizationProblem problem;

      auto x = problem.DecisionVariable();
      x.SetValue(x0);
      auto y = problem.DecisionVariable();
      y.SetValue(y0);

      problem.Minimize(sleipnir::pow(1 - x, 2) +
                       100 * sleipnir::pow(y - sleipnir::pow(x, 2), 2));

      problem.SubjectTo(sleipnir::pow(x - 1, 3) - y + 1 <= 0);
      problem.SubjectTo(x + y - 2 <= 0);

      auto status = problem.Solve({.diagnostics = true});

      CHECK(status.costFunctionType == sleipnir::ExpressionType::kNonlinear);
      CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kNone);
      CHECK(status.inequalityConstraintType ==
            sleipnir::ExpressionType::kNonlinear);
      CHECK(status.exitCondition == sleipnir::SolverExitCondition::kSuccess);

      auto Near = [](double expected, double actual, double tolerance) {
        return std::abs(expected - actual) < tolerance;
      };

      // Local minimum at (0.0, 0.0)
      // Global minimum at (1.0, 1.0)
      CHECK((Near(0.0, x.Value(), 1e-2) || Near(1.0, x.Value(), 1e-2)));
      INFO(fmt::format("  (x₀, y₀) = ({}, {})\n", x0, y0));
      INFO(fmt::format("  x.Value(0) = {}", x.Value()));
      CHECK((Near(0.0, y.Value(), 1e-2) || Near(1.0, y.Value(), 1e-2)));
      INFO(fmt::format("  (x₀, y₀) = ({}, {})\n", x0, y0));
      INFO(fmt::format("  y.Value(0) = {}", y.Value()));
    }
  }
}

TEST_CASE("NonlinearProblem - Rosenbrock with disk constraint",
          "[NonlinearProblem]") {
  // https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization
  for (auto x0 : Range(-1.5, 1.5, 0.1)) {
    for (auto y0 : Range(-1.5, 1.5, 0.1)) {
      sleipnir::OptimizationProblem problem;

      auto x = problem.DecisionVariable();
      x.SetValue(x0);
      auto y = problem.DecisionVariable();
      y.SetValue(y0);

      problem.Minimize(sleipnir::pow(1 - x, 2) +
                       100 * sleipnir::pow(y - sleipnir::pow(x, 2), 2));

      problem.SubjectTo(sleipnir::pow(x, 2) + sleipnir::pow(y, 2) <= 2);

      auto status = problem.Solve({.diagnostics = true});

      CHECK(status.costFunctionType == sleipnir::ExpressionType::kNonlinear);
      CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kNone);
      CHECK(status.inequalityConstraintType ==
            sleipnir::ExpressionType::kQuadratic);
      CHECK(status.exitCondition == sleipnir::SolverExitCondition::kSuccess);

      CHECK(x.Value() == Catch::Approx(1.0).margin(1e-1));
      INFO(fmt::format("  (x₀, y₀) = ({}, {})\n", x0, y0));
      INFO(fmt::format("  x.Value(0) = {}", x.Value()));
      CHECK(y.Value() == Catch::Approx(1.0).margin(1e-1));
      INFO(fmt::format("  (x₀, y₀) = ({}, {})\n", x0, y0));
      INFO(fmt::format("  x.Value(0) = {}", x.Value()));
    }
  }
}

TEST_CASE("NonlinearProblem - Narrow feasible region", "[NonlinearProblem]") {
  sleipnir::OptimizationProblem problem;

  auto x = problem.DecisionVariable();
  x.SetValue(20.0);

  auto y = problem.DecisionVariable();
  y.SetValue(50.0);

  problem.Minimize(sleipnir::sqrt(x * x + y * y));

  problem.SubjectTo(y == -x + 5.0);

  auto status = problem.Solve({.diagnostics = true});

  CHECK(status.costFunctionType == sleipnir::ExpressionType::kNonlinear);
  CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kLinear);
  CHECK(status.inequalityConstraintType == sleipnir::ExpressionType::kNone);

#if defined(__APPLE__) && defined(__aarch64__)
  // FIXME: Fails on macOS arm64 with "diverging iterates"
  CHECK(status.exitCondition ==
        sleipnir::SolverExitCondition::kDivergingIterates);
  SKIP("Fails on macOS arm64 with \"diverging iterates\"");
#else
  CHECK(status.exitCondition == sleipnir::SolverExitCondition::kSuccess);
#endif

  CHECK(x.Value() == Catch::Approx(2.5).margin(1e-2));
  CHECK(y.Value() == Catch::Approx(2.5).margin(1e-2));
}
