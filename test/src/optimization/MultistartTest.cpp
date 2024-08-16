// Copyright (c) Sleipnir contributors

#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/Multistart.hpp>
#include <sleipnir/optimization/OptimizationProblem.hpp>

#include "CatchStringConverters.hpp"

TEST_CASE("Multistart - Mishra's Bird function", "[NonlinearProblem]") {
  struct DecisionVariables {
    double x;
    double y;
  };

  auto Solve = [](const DecisionVariables& input)
      -> sleipnir::MultistartResult<DecisionVariables> {
    sleipnir::OptimizationProblem problem;

    auto x = problem.DecisionVariable();
    x.SetValue(input.x);
    auto y = problem.DecisionVariable();
    y.SetValue(input.y);

    // https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_constrained_optimization
    problem.Minimize(sleipnir::sin(y) *
                         sleipnir::exp(sleipnir::pow(1 - sleipnir::cos(x), 2)) +
                     sleipnir::cos(x) *
                         sleipnir::exp(sleipnir::pow(1 - sleipnir::sin(y), 2)) +
                     sleipnir::pow(x - y, 2));

    problem.SubjectTo(sleipnir::pow(x + 5, 2) + sleipnir::pow(y + 5, 2) < 25);

    return {problem.Solve(), DecisionVariables{x.Value(), y.Value()}};
  };

  auto [status, variables] = sleipnir::Multistart<DecisionVariables>(
      Solve,
      std::vector{DecisionVariables{-3, -8}, DecisionVariables{-3, -1.5}});

  CHECK(status.costFunctionType == sleipnir::ExpressionType::kNonlinear);
  CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kNone);
  CHECK(status.inequalityConstraintType ==
        sleipnir::ExpressionType::kQuadratic);
  CHECK(status.exitCondition == sleipnir::SolverExitCondition::kSuccess);

  CHECK(variables.x == Catch::Approx(-3.130246803458174).margin(1e-15));
  CHECK(variables.y == Catch::Approx(-1.5821421769364057).margin(1e-15));
}
