// Copyright (c) Sleipnir contributors

#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "catch_string_converters.hpp"

TEST_CASE("Problem - Empty", "[Problem]") {
  slp::Problem problem;

  CHECK(problem.cost_function_type() == slp::ExpressionType::NONE);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);
}

TEST_CASE("Problem - No cost, unconstrained", "[Problem]") {
  {
    slp::Problem problem;

    auto X = problem.decision_variable(2, 3);

    CHECK(problem.cost_function_type() == slp::ExpressionType::NONE);
    CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
    CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

    CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

    for (int row = 0; row < X.rows(); ++row) {
      for (int col = 0; col < X.cols(); ++col) {
        CHECK(X.value(row, col) == 0.0);
      }
    }
  }

  {
    slp::Problem problem;

    auto X = problem.decision_variable(2, 3);
    X.set_value(Eigen::Matrix<double, 2, 3>::Ones());

    CHECK(problem.cost_function_type() == slp::ExpressionType::NONE);
    CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
    CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

    CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

    for (int row = 0; row < X.rows(); ++row) {
      for (int col = 0; col < X.cols(); ++col) {
        CHECK(X.value(row, col) == 1.0);
      }
    }
  }
}
