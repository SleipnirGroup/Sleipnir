// Copyright (c) Sleipnir contributors

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/OptimizationProblem.hpp>

#include "CmdlineArguments.hpp"

TEST_CASE("Maximize", "[LinearProblem]") {
  sleipnir::OptimizationProblem problem;

  auto x = problem.DecisionVariable();
  x.SetValue(1.0);

  auto y = problem.DecisionVariable();
  y.SetValue(1.0);

  problem.Maximize(50 * x + 40 * y);

  problem.SubjectTo(x + 1.5 * y <= 750);
  problem.SubjectTo(2 * x + 3 * y <= 1500);
  problem.SubjectTo(2 * x + y <= 1000);
  problem.SubjectTo(x >= 0);
  problem.SubjectTo(y >= 0);

  auto status =
      problem.Solve({.diagnostics = Argv().Contains("--enable-diagnostics")});

  CHECK(status.costFunctionType == sleipnir::ExpressionType::kLinear);
  CHECK(status.equalityConstraintType == sleipnir::ExpressionType::kNone);
  CHECK(status.inequalityConstraintType == sleipnir::ExpressionType::kLinear);
  CHECK(status.exitCondition == sleipnir::SolverExitCondition::kSuccess);

  CHECK(x.Value() == Catch::Approx(375.0).margin(1e-6));
  CHECK(y.Value() == Catch::Approx(250.0).margin(1e-6));
}
