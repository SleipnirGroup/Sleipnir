// Copyright (c) Sleipnir contributors

#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/optimization_problem.hpp>

#include "catch_string_converters.hpp"

TEST_CASE("trivial_problem - Empty", "[trivial_problem]") {
  slp::OptimizationProblem problem;

  auto status = problem.solve({.diagnostics = true});

  CHECK(status.cost_function_type == slp::ExpressionType::NONE);
  CHECK(status.equality_constraint_type == slp::ExpressionType::NONE);
  CHECK(status.inequality_constraint_type == slp::ExpressionType::NONE);
  CHECK(status.exit_condition == slp::SolverExitCondition::SUCCESS);
}

TEST_CASE("trivial_problem - No cost, unconstrained", "[trivial_problem]") {
  {
    slp::OptimizationProblem problem;

    auto X = problem.decision_variable(2, 3);

    auto status = problem.solve({.diagnostics = true});

    CHECK(status.cost_function_type == slp::ExpressionType::NONE);
    CHECK(status.equality_constraint_type == slp::ExpressionType::NONE);
    CHECK(status.inequality_constraint_type == slp::ExpressionType::NONE);
    CHECK(status.exit_condition == slp::SolverExitCondition::SUCCESS);

    for (int row = 0; row < X.rows(); ++row) {
      for (int col = 0; col < X.cols(); ++col) {
        CHECK(X.value(row, col) == 0.0);
      }
    }
  }

  {
    slp::OptimizationProblem problem;

    auto X = problem.decision_variable(2, 3);
    X.set_value(Eigen::Matrix<double, 2, 3>{{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}});

    auto status = problem.solve({.diagnostics = true});

    CHECK(status.cost_function_type == slp::ExpressionType::NONE);
    CHECK(status.equality_constraint_type == slp::ExpressionType::NONE);
    CHECK(status.inequality_constraint_type == slp::ExpressionType::NONE);
    CHECK(status.exit_condition == slp::SolverExitCondition::SUCCESS);

    for (int row = 0; row < X.rows(); ++row) {
      for (int col = 0; col < X.cols(); ++col) {
        CHECK(X.value(row, col) == 1.0);
      }
    }
  }
}
