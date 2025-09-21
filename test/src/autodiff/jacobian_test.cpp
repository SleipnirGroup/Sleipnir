// Copyright (c) Sleipnir contributors

#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/jacobian.hpp>

#include "util/scope_exit.hpp"

TEST_CASE("Jacobian - y = x", "[Jacobian]") {
  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::VariableMatrix x{3};
  for (int i = 0; i < 3; ++i) {
    x[i].set_value(i + 1);
  }

  // y = x
  //
  //         [1  0  0]
  // dy/dx = [0  1  0]
  //         [0  0  1]
  auto y = x;
  auto J = slp::Jacobian(y, x);

  Eigen::MatrixXd expected_J{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
  CHECK(J.get().value() == expected_J);
  CHECK(J.value().toDense() == expected_J);
}

TEST_CASE("Jacobian - y = 3x", "[Jacobian]") {
  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::VariableMatrix x{3};
  for (int i = 0; i < 3; ++i) {
    x[i].set_value(i + 1);
  }

  // y = 3x
  //
  //         [3  0  0]
  // dy/dx = [0  3  0]
  //         [0  0  3]
  auto y = 3 * x;
  auto J = slp::Jacobian(y, x);

  Eigen::MatrixXd expected_J{{3.0, 0.0, 0.0}, {0.0, 3.0, 0.0}, {0.0, 0.0, 3.0}};
  CHECK(J.get().value() == expected_J);
  CHECK(J.value().toDense() == expected_J);
}

TEST_CASE("Jacobian - Products", "[Jacobian]") {
  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::VariableMatrix x{3};
  for (int i = 0; i < 3; ++i) {
    x[i].set_value(i + 1);
  }

  //     [x₁x₂]
  // y = [x₂x₃]
  //     [x₁x₃]
  //
  //         [x₂  x₁  0 ]
  // dy/dx = [0   x₃  x₂]
  //         [x₃  0   x₁]
  //
  //         [2  1  0]
  // dy/dx = [0  3  2]
  //         [3  0  1]
  slp::VariableMatrix y{3};
  y[0] = x[0] * x[1];
  y[1] = x[1] * x[2];
  y[2] = x[0] * x[2];
  auto J = slp::Jacobian(y, x);

  Eigen::MatrixXd expected_J{{2.0, 1.0, 0.0}, {0.0, 3.0, 2.0}, {3.0, 0.0, 1.0}};
  CHECK(J.get().value() == expected_J);
  CHECK(J.value().toDense() == expected_J);
}

TEST_CASE("Jacobian - Nested products", "[Jacobian]") {
  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::VariableMatrix x{1};
  x[0].set_value(3);
  CHECK(x.value(0) == 3.0);

  //     [ 5x]   [15]
  // y = [ 7x] = [21]
  //     [11x]   [33]
  slp::VariableMatrix y{3};
  y[0] = 5 * x[0];
  y[1] = 7 * x[0];
  y[2] = 11 * x[0];
  CHECK(y.value(0) == 15.0);
  CHECK(y.value(1) == 21.0);
  CHECK(y.value(2) == 33.0);

  //     [y₁y₂]   [15⋅21]   [315]
  // z = [y₂y₃] = [21⋅33] = [693]
  //     [y₁y₃]   [15⋅33]   [495]
  slp::VariableMatrix z{3};
  z[0] = y[0] * y[1];
  z[1] = y[1] * y[2];
  z[2] = y[0] * y[2];
  CHECK(z.value(0) == 315.0);
  CHECK(z.value(1) == 693.0);
  CHECK(z.value(2) == 495.0);

  //     [ 5x]
  // y = [ 7x]
  //     [11x]
  //
  //         [ 5]
  // dy/dx = [ 7]
  //         [11]
  auto J = slp::Jacobian(y, x);
  CHECK(J.get().value(0, 0) == 5.0);
  CHECK(J.get().value(1, 0) == 7.0);
  CHECK(J.get().value(2, 0) == 11.0);
  CHECK(J.value().coeff(0, 0) == 5.0);
  CHECK(J.value().coeff(1, 0) == 7.0);
  CHECK(J.value().coeff(2, 0) == 11.0);

  //     [y₁y₂]
  // z = [y₂y₃]
  //     [y₁y₃]
  //
  //         [y₂  y₁  0 ]   [21  15   0]
  // dz/dy = [0   y₃  y₂] = [ 0  33  21]
  //         [y₃  0   y₁]   [33   0  15]
  J = slp::Jacobian(z, y);
  Eigen::MatrixXd expected_J{
      {21.0, 15.0, 0.0}, {0.0, 33.0, 21.0}, {33.0, 0.0, 15.0}};
  CHECK(J.get().value() == expected_J);
  CHECK(J.value().toDense() == expected_J);

  //     [y₁y₂]   [5x⋅ 7x]   [35x²]
  // z = [y₂y₃] = [7x⋅11x] = [77x²]
  //     [y₁y₃]   [5x⋅11x]   [55x²]
  //
  //         [ 70x]   [210]
  // dz/dx = [154x] = [462]
  //         [110x] = [330]
  J = slp::Jacobian(z, x);
  expected_J = Eigen::MatrixXd{{210.0}, {462.0}, {330.0}};
  CHECK(J.get().value() == expected_J);
  CHECK(J.value().toDense() == expected_J);
}

TEST_CASE("Jacobian - Non-square", "[Jacobian]") {
  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::VariableMatrix x{3};
  for (int i = 0; i < 3; ++i) {
    x[i].set_value(i + 1);
  }

  // y = [x₁ + 3x₂ − 5x₃]
  //
  // dy/dx = [1  3  −5]
  slp::VariableMatrix y{1};
  y[0] = x[0] + 3 * x[1] - 5 * x[2];
  auto J = slp::Jacobian(y, x);

  Eigen::MatrixXd expected_J{{1.0, 3.0, -5.0}};

  auto J_get_value = J.get().value();
  CHECK(J_get_value.rows() == 1);
  CHECK(J_get_value.cols() == 3);
  CHECK(J_get_value == expected_J);

  auto J_value = J.value();
  CHECK(J_value.rows() == 1);
  CHECK(J_value.cols() == 3);
  CHECK(J_value.toDense() == expected_J);
}

TEST_CASE("Jacobian - Variable reuse", "[Jacobian]") {
  slp::scope_exit exit{
      [] { CHECK(slp::global_pool_resource().blocks_in_use() == 0u); }};

  slp::VariableMatrix x{2};
  for (int i = 0; i < 2; ++i) {
    x[i].set_value(i + 1);
  }

  // y = [x₁x₂]
  slp::VariableMatrix y{1};
  y[0] = x[0] * x[1];

  slp::Jacobian jacobian{y, x};

  // dy/dx = [x₂  x₁]
  // dy/dx = [2  1]
  Eigen::MatrixXd J = jacobian.value();

  CHECK(J.rows() == 1);
  CHECK(J.cols() == 2);
  CHECK(J(0, 0) == 2.0);
  CHECK(J(0, 1) == 1.0);

  x[0].set_value(2);
  x[1].set_value(1);
  // dy/dx = [x₂  x₁]
  // dy/dx = [1  2]
  J = jacobian.value();

  CHECK(J.rows() == 1);
  CHECK(J.cols() == 2);
  CHECK(J(0, 0) == 1.0);
  CHECK(J(0, 1) == 2.0);
}
