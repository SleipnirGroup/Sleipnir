// Copyright (c) Sleipnir contributors

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/optimization/problem.hpp>

#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "scalar_types_under_test.hpp"

TEMPLATE_TEST_CASE("Problem - Unconstrained 1D", "[Problem]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.decision_variable();
  x.set_value(T(2));

  problem.minimize(x * x - T(6) * x);

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  CHECK_THAT(x.value(), WithinAbs(T(3), T(1e-6)));
}

TEMPLATE_TEST_CASE("Problem - Unconstrained 2D", "[Problem]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  {
    slp::Problem<T> problem;

    auto x = problem.decision_variable();
    auto y = problem.decision_variable();
    x.set_value(T(1));
    y.set_value(T(2));

    problem.minimize(x * x + y * y);

    CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
    CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
    CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

    CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

    CHECK_THAT(x.value(), WithinAbs(T(0), T(1e-6)));
    CHECK_THAT(y.value(), WithinAbs(T(0), T(1e-6)));
  }

  {
    slp::Problem<T> problem;

    auto x = problem.decision_variable(2);
    x[0].set_value(T(1));
    x[1].set_value(T(2));

    problem.minimize(x.T() * x);

    CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
    CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
    CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

    CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

    CHECK_THAT(x.value(0), WithinAbs(T(0), T(1e-6)));
    CHECK_THAT(x.value(1), WithinAbs(T(0), T(1e-6)));
  }
}

TEMPLATE_TEST_CASE("Problem - Equality-constrained", "[Problem]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

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
    slp::Problem<T> problem;

    auto x = problem.decision_variable();
    auto y = problem.decision_variable();

    problem.maximize(x * y);

    problem.subject_to(x + T(3) * y == T(36));

    CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
    CHECK(problem.equality_constraint_type() == slp::ExpressionType::LINEAR);
    CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

    CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

    CHECK_THAT(x.value(), WithinAbs(T(18), T(1e-5)));
    CHECK_THAT(y.value(), WithinAbs(T(6), T(1e-5)));
  }

  {
    slp::Problem<T> problem;

    auto x = problem.decision_variable(2);
    x[0].set_value(T(1));
    x[1].set_value(T(2));

    problem.minimize(x.T() * x);

    problem.subject_to(x == Eigen::Matrix<T, 2, 1>{{T(3)}, {T(3)}});

    CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
    CHECK(problem.equality_constraint_type() == slp::ExpressionType::LINEAR);
    CHECK(problem.inequality_constraint_type() == slp::ExpressionType::NONE);

    CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

    CHECK_THAT(x.value(0), WithinAbs(T(3), T(1e-5)));
    CHECK_THAT(x.value(1), WithinAbs(T(3), T(1e-5)));
  }
}

TEMPLATE_TEST_CASE("Problem - Inequality-constrained 2D", "[Problem]",
                   SCALAR_TYPES_UNDER_TEST) {
  using T = TestType;

  slp::Problem<T> problem;

  auto x = problem.decision_variable();
  auto y = problem.decision_variable();
  x.set_value(T(5));
  y.set_value(T(5));

  problem.minimize(x * x + y * T(2) * y);
  problem.subject_to(y >= -x + T(5));

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);

  CHECK_THAT(x.value(), WithinAbs(T(3) + T(1.0 / 3.0), T(1e-6)));
  CHECK_THAT(y.value(), WithinAbs(T(1) + T(2.0 / 3.0), T(1e-6)));
}
