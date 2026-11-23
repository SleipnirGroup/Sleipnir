// Copyright (c) Sleipnir contributors

#include <cmath>

#include <Eigen/Core>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <sleipnir/autodiff/expression_type.hpp>
#include <sleipnir/autodiff/hessian.hpp>
#include <sleipnir/autodiff/jacobian.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/optimization/solver/exit_status.hpp>

#include "catch_matchers.hpp"
#include "catch_string_converters.hpp"
#include "scalar_types.hpp"
#include "tolerance.hpp"

TEMPLATE_TEST_CASE("Problem decision variable - T init assign", "[Problem]",
                   SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  // T zero init
  auto x = problem.decision_variable();
  CHECK(x.value() == T(0));

  // T assignment
  x.set_value(T(1));
  CHECK(x.value() == T(1));
  x.set_value(T(2));
  CHECK(x.value() == T(2));
}

TEMPLATE_TEST_CASE("Problem decision variable - Vector init assign",
                   "[Problem]", SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  // Vector zero init
  auto y = problem.decision_variable(2);
  CHECK(y.value(0) == T(0));
  CHECK(y.value(1) == T(0));

  // Vector assignment
  y[0].set_value(T(1));
  y[1].set_value(T(2));
  CHECK(y.value(0) == T(1));
  CHECK(y.value(1) == T(2));
  y[0].set_value(T(3));
  y[1].set_value(T(4));
  CHECK(y.value(0) == T(3));
  CHECK(y.value(1) == T(4));
}

TEMPLATE_TEST_CASE("Problem decision variable - Static matrix init assign",
                   "[Problem]", SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  // Matrix zero init
  auto z = problem.decision_variable(3, 2);
  CHECK(z.value(0, 0) == T(0));
  CHECK(z.value(0, 1) == T(0));
  CHECK(z.value(1, 0) == T(0));
  CHECK(z.value(1, 1) == T(0));
  CHECK(z.value(2, 0) == T(0));
  CHECK(z.value(2, 1) == T(0));

  // Matrix assignment; element comparison
  z.set_value(Eigen::Matrix<T, 3, 2>{{T(1), T(2)}, {T(3), T(4)}, {T(5), T(6)}});
  CHECK(z.value(0, 0) == T(1));
  CHECK(z.value(0, 1) == T(2));
  CHECK(z.value(1, 0) == T(3));
  CHECK(z.value(1, 1) == T(4));
  CHECK(z.value(2, 0) == T(5));
  CHECK(z.value(2, 1) == T(6));

  // Matrix assignment; matrix comparison
  {
    Eigen::Matrix<T, 3, 2> expected{
        {T(7), T(8)}, {T(9), T(10)}, {T(11), T(12)}};
    z.set_value(expected);
    CHECK(z.value() == expected);
  }

  // Block assignment
  z.block(0, 0, 2, 1).set_value(Eigen::Vector2<T>{{T(1)}, {T(1)}});
  CHECK(z.value() ==
        Eigen::Matrix<T, 3, 2>{{T(1), T(8)}, {T(1), T(10)}, {T(11), T(12)}});

  // Segment assignment
  z.block(0, 0, 3, 1)
      .segment(0, 2)
      .set_value(Eigen::Vector2<T>{{T(2)}, {T(2)}});
  CHECK(z.value() ==
        Eigen::Matrix<T, 3, 2>{{T(2), T(8)}, {T(2), T(10)}, {T(11), T(12)}});
}

TEMPLATE_TEST_CASE("Problem decision variable - Dynamic matrix init assign",
                   "[Problem]", SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  // Matrix zero init
  auto z = problem.decision_variable(3, 2);
  CHECK(z.value(0, 0) == T(0));
  CHECK(z.value(0, 1) == T(0));
  CHECK(z.value(1, 0) == T(0));
  CHECK(z.value(1, 1) == T(0));
  CHECK(z.value(2, 0) == T(0));
  CHECK(z.value(2, 1) == T(0));

  // Matrix assignment; element comparison
  z.set_value(Eigen::MatrixX<T>{{T(1), T(2)}, {T(3), T(4)}, {T(5), T(6)}});
  CHECK(z.value(0, 0) == T(1));
  CHECK(z.value(0, 1) == T(2));
  CHECK(z.value(1, 0) == T(3));
  CHECK(z.value(1, 1) == T(4));
  CHECK(z.value(2, 0) == T(5));
  CHECK(z.value(2, 1) == T(6));

  // Matrix assignment; matrix comparison
  Eigen::MatrixX<T> expected{{T(7), T(8)}, {T(9), T(10)}, {T(11), T(12)}};
  z.set_value(expected);
  CHECK(z.value() == expected);

  // Block assignment
  z.block(0, 0, 2, 1).set_value(Eigen::MatrixX<T>{{T(1)}, {T(1)}});
  CHECK(z.value() ==
        Eigen::MatrixX<T>{{T(1), T(8)}, {T(1), T(10)}, {T(11), T(12)}});

  // Segment assignment
  z.block(0, 0, 3, 1)
      .segment(0, 2)
      .set_value(Eigen::MatrixX<T>{{T(2)}, {T(2)}});
  CHECK(z.value() ==
        Eigen::MatrixX<T>{{T(2), T(8)}, {T(2), T(10)}, {T(11), T(12)}});
}

TEMPLATE_TEST_CASE("Problem decision variable - Symmetric matrix", "[Problem]",
                   SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  // Matrix zero init
  auto A = problem.symmetric_decision_variable(2);
  CHECK(A.value(0, 0) == T(0));
  CHECK(A.value(0, 1) == T(0));
  CHECK(A.value(1, 0) == T(0));
  CHECK(A.value(1, 1) == T(0));

  // Assign to lower triangle
  A[0, 0].set_value(T(1));
  A[1, 0].set_value(T(2));
  A[1, 1].set_value(T(3));

  // Confirm whole matrix changed
  CHECK(A.value(0, 0) == T(1));
  CHECK(A.value(0, 1) == T(2));
  CHECK(A.value(1, 0) == T(2));
  CHECK(A.value(1, 1) == T(3));
}

TEMPLATE_TEST_CASE("Problem parameter - Init assign", "[Problem]",
                   SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto p = problem.parameter();
  CHECK(p.type() == slp::ExpressionType::PARAMETER);
  CHECK(p.value() == T(0));
  p.set_value(T(2));
  CHECK(p.value() == T(2));

  auto P = problem.parameter(2, 3);
  CHECK(P.rows() == 2);
  CHECK(P.cols() == 3);
  for (const auto& elem : P) {
    CHECK(elem.type() == slp::ExpressionType::PARAMETER);
  }
  P.set_value(Eigen::Matrix<T, 2, 3>{{T(1), T(2), T(3)}, {T(4), T(5), T(6)}});
  CHECK(P.value(1, 2) == T(6));
}

TEMPLATE_TEST_CASE("Problem parameter - Expression types", "[Problem]",
                   SCALAR_TYPES) {
  using T = TestType;
  using enum slp::ExpressionType;

  slp::Problem<T> problem;

  auto p = problem.parameter();
  auto x = problem.decision_variable();

  CHECK((T(2) * p).type() == PARAMETER);
  CHECK((p + T(1)).type() == PARAMETER);
  CHECK((p * p).type() == PARAMETER);
  CHECK((T(1) / p).type() == PARAMETER);
  CHECK((-p).type() == PARAMETER);
  CHECK(slp::pow(p, T(2)).type() == PARAMETER);
  CHECK(slp::pow(p, p).type() == PARAMETER);
  CHECK(slp::sin(p).type() == PARAMETER);
  CHECK(slp::sqrt(p).type() == PARAMETER);
  CHECK(slp::atan2(p, slp::Variable<T>{T(1)}).type() == PARAMETER);
  CHECK(slp::hypot(p, p, p).type() == PARAMETER);
  CHECK(slp::max(p, slp::Variable<T>{T(1)}).type() == PARAMETER);
  CHECK(slp::if_else(p < T(1), p, T(2)).type() == PARAMETER);
  CHECK((p * x).type() == LINEAR);
  CHECK((x / p).type() == LINEAR);
  CHECK((slp::sin(p) * x).type() == LINEAR);
  CHECK((p * x * x).type() == QUADRATIC);
  CHECK((p / x).type() == NONLINEAR);
  CHECK(slp::sin(x).type() == NONLINEAR);
  CHECK(slp::pow(p, x).type() == NONLINEAR);
  CHECK(slp::max(p, x).type() == NONLINEAR);
  CHECK(slp::if_else(p < T(1), x, T(2)).type() == NONLINEAR);
}

TEMPLATE_TEST_CASE("Problem parameter - Not folded", "[Problem]",
                   SCALAR_TYPES) {
  using T = TestType;
  using std::sqrt;

  slp::Problem<T> problem;

  // Parameters start at zero and one here, which constants would fold
  auto p = problem.parameter();
  auto q = problem.parameter();
  q.set_value(T(1));
  auto x = problem.decision_variable();

  auto product = p * x;
  auto quotient = x / q;
  auto sum = p + T(3) * q;
  auto function = slp::sqrt(p) * x;

  p.set_value(T(2));
  q.set_value(T(4));
  x.set_value(T(3));

  CHECK(product.value() == T(6));
  CHECK(quotient.value() == T(0.75));
  CHECK(sum.value() == T(14));
  CHECK(function.value() == T(3) * sqrt(T(2)));
}

TEMPLATE_TEST_CASE("Problem parameter - Resolve with new values", "[Problem]",
                   SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto target = problem.parameter();
  auto upper_bound = problem.parameter();
  auto x = problem.decision_variable();

  problem.minimize((x - target) * (x - target));
  problem.subject_to(x <= upper_bound);

  CHECK(problem.cost_function_type() == slp::ExpressionType::QUADRATIC);
  CHECK(problem.equality_constraint_type() == slp::ExpressionType::NONE);
  CHECK(problem.inequality_constraint_type() == slp::ExpressionType::LINEAR);

  // Unconstrained minimum is feasible
  target.set_value(T(1));
  upper_bound.set_value(T(5));
  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);
  CHECK_THAT(x.value(), WithinAbs(T(1), solution_tolerance<T>(1)));

  // Unconstrained minimum is infeasible, so the bound is active
  target.set_value(T(10));
  upper_bound.set_value(T(3));
  CHECK(problem.solve({.diagnostics = true}) == slp::ExitStatus::SUCCESS);
  CHECK_THAT(x.value(), WithinAbs(T(3), solution_tolerance<T>(1)));
}

TEMPLATE_TEST_CASE("Problem parameter - Jacobian updates", "[Problem]",
                   SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto p = problem.parameter();
  auto x = problem.decision_variable(2);

  // Both rows are linear, but the first row's coefficients depend on p
  slp::VariableMatrix<T> y{{p * x[0] + x[1]}, {T(3) * x[1]}};
  slp::Jacobian J{y, x};

  p.set_value(T(2));
  Eigen::Matrix<T, 2, 2> expected{{T(2), T(1)}, {T(0), T(3)}};
  CHECK(Eigen::Matrix<T, 2, 2>{J.value()} == expected);

  p.set_value(T(5));
  expected(0, 0) = T(5);
  CHECK(Eigen::Matrix<T, 2, 2>{J.value()} == expected);
}

TEMPLATE_TEST_CASE("Problem parameter - Hessian updates", "[Problem]",
                   SCALAR_TYPES) {
  using T = TestType;

  slp::Problem<T> problem;

  auto p = problem.parameter();
  auto x = problem.decision_variable(2);

  // The cost is quadratic, but its Hessian depends on p
  slp::Hessian<T> H{p * x[0] * x[0] + x[0] * x[1], x};

  p.set_value(T(2));
  Eigen::Matrix<T, 2, 2> expected{{T(4), T(1)}, {T(1), T(0)}};
  CHECK(Eigen::Matrix<T, 2, 2>{H.value()} == expected);

  p.set_value(T(5));
  expected(0, 0) = T(10);
  CHECK(Eigen::Matrix<T, 2, 2>{H.value()} == expected);
}
