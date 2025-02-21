// Copyright (c) Sleipnir contributors

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/optimization_problem.hpp>

#include "catch_string_converters.hpp"

TEST_CASE("quadratic_problem - Unconstrained 1D", "[quadratic_problem]") {
  sleipnir::OptimizationProblem problem;

  auto x = problem.decision_variable();
  x.set_value(2.0);

  problem.minimize(x * x - 6.0 * x);

  auto status = problem.solve({.diagnostics = true});

  CHECK(status.cost_function_type == sleipnir::ExpressionType::QUADRATIC);
  CHECK(status.equality_constraint_type == sleipnir::ExpressionType::NONE);
  CHECK(status.inequality_constraint_type == sleipnir::ExpressionType::NONE);
  CHECK(status.exit_condition == sleipnir::SolverExitCondition::SUCCESS);

  CHECK(x.value() == Catch::Approx(3.0).margin(1e-6));
}

TEST_CASE("quadratic_problem - Unconstrained 2D", "[quadratic_problem]") {
  {
    sleipnir::OptimizationProblem problem;

    auto x = problem.decision_variable();
    x.set_value(1.0);
    auto y = problem.decision_variable();
    y.set_value(2.0);

    problem.minimize(x * x + y * y);

    auto status = problem.solve({.diagnostics = true});

    CHECK(status.cost_function_type == sleipnir::ExpressionType::QUADRATIC);
    CHECK(status.equality_constraint_type == sleipnir::ExpressionType::NONE);
    CHECK(status.inequality_constraint_type == sleipnir::ExpressionType::NONE);
    CHECK(status.exit_condition == sleipnir::SolverExitCondition::SUCCESS);

    CHECK(x.value() == Catch::Approx(0.0).margin(1e-6));
    CHECK(y.value() == Catch::Approx(0.0).margin(1e-6));
  }

  {
    sleipnir::OptimizationProblem problem;

    auto x = problem.decision_variable(2);
    x[0].set_value(1.0);
    x[1].set_value(2.0);

    problem.minimize(x.T() * x);

    auto status = problem.solve({.diagnostics = true});

    CHECK(status.cost_function_type == sleipnir::ExpressionType::QUADRATIC);
    CHECK(status.equality_constraint_type == sleipnir::ExpressionType::NONE);
    CHECK(status.inequality_constraint_type == sleipnir::ExpressionType::NONE);
    CHECK(status.exit_condition == sleipnir::SolverExitCondition::SUCCESS);

    CHECK(x.value(0) == Catch::Approx(0.0).margin(1e-6));
    CHECK(x.value(1) == Catch::Approx(0.0).margin(1e-6));
  }
}

TEST_CASE("quadratic_problem - Equality-constrained", "[quadratic_problem]") {
  // Maximize xy subject to x + 3y = 36.
  //
  // Maximize f(x,y) = xy
  // subject to g(x,y) = x + 3y - 36 = 0
  //
  //         value func  constraint
  //              |          |
  //              v          v
  // L(x,y,λ) = f(x,y) - λg(x,y)
  // L(x,y,λ) = xy - λ(x + 3y - 36)
  // L(x,y,λ) = xy - xλ - 3yλ + 36λ
  //
  // ∇_x,y,λ L(x,y,λ) = 0
  //
  // ∂L/∂x = y - λ
  // ∂L/∂y = x - 3λ
  // ∂L/∂λ = -x - 3y + 36
  //
  //  0x + 1y - 1λ = 0
  //  1x + 0y - 3λ = 0
  // -1x - 3y + 0λ + 36 = 0
  //
  // [ 0  1 -1][x]   [  0]
  // [ 1  0 -3][y] = [  0]
  // [-1 -3  0][λ]   [-36]
  //
  // Solve with:
  // ```python
  //   np.linalg.solve(
  //     np.array([[0,1,-1],
  //               [1,0,-3],
  //               [-1,-3,0]]),
  //     np.array([[0], [0], [-36]]))
  // ```
  //
  // [x]   [18]
  // [y] = [ 6]
  // [λ]   [ 6]
  {
    sleipnir::OptimizationProblem problem;

    auto x = problem.decision_variable();
    auto y = problem.decision_variable();

    problem.maximize(x * y);

    problem.subject_to(x + 3 * y == 36);

    auto status = problem.solve({.diagnostics = true});

    CHECK(status.cost_function_type == sleipnir::ExpressionType::QUADRATIC);
    CHECK(status.equality_constraint_type == sleipnir::ExpressionType::LINEAR);
    CHECK(status.inequality_constraint_type == sleipnir::ExpressionType::NONE);
    CHECK(status.exit_condition == sleipnir::SolverExitCondition::SUCCESS);

    CHECK(x.value() == Catch::Approx(18.0).margin(1e-5));
    CHECK(y.value() == Catch::Approx(6.0).margin(1e-5));
  }

  {
    sleipnir::OptimizationProblem problem;

    auto x = problem.decision_variable(2);
    x[0].set_value(1.0);
    x[1].set_value(2.0);

    problem.minimize(x.T() * x);

    problem.subject_to(x == Eigen::Matrix<double, 2, 1>{{3.0, 3.0}});

    auto status = problem.solve({.diagnostics = true});

    CHECK(status.cost_function_type == sleipnir::ExpressionType::QUADRATIC);
    CHECK(status.equality_constraint_type == sleipnir::ExpressionType::LINEAR);
    CHECK(status.inequality_constraint_type == sleipnir::ExpressionType::NONE);
    CHECK(status.exit_condition == sleipnir::SolverExitCondition::SUCCESS);

    CHECK(x.value(0) == Catch::Approx(3.0).margin(1e-5));
    CHECK(x.value(1) == Catch::Approx(3.0).margin(1e-5));
  }
}
