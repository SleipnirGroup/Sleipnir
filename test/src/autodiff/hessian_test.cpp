// Copyright (c) Sleipnir contributors

#include <cmath>
#include <functional>
#include <numeric>

#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sleipnir/autodiff/gradient.hpp>
#include <sleipnir/autodiff/hessian.hpp>
#include <sleipnir/autodiff/variable.hpp>

#include "range.hpp"
#include "util/scope_exit.hpp"

TEST_CASE("Hessian - Linear", "[Hessian]") {
  sleipnir::scope_exit exit{
      [] { CHECK(sleipnir::global_pool_resource().blocks_in_use() == 0u); }};

  // y = x
  sleipnir::VariableMatrix x{1};
  x(0).set_value(3);
  sleipnir::Variable y = x(0);

  // dy/dx = 1
  double g = sleipnir::Gradient(y, x(0)).value().coeff(0);
  CHECK(g == 1.0);

  // d²y/dx² = 0
  auto H = sleipnir::Hessian(y, x);
  CHECK(H.get().value(0, 0) == 0.0);
  CHECK(H.value().coeff(0, 0) == 0.0);
}

TEST_CASE("Hessian - Quadratic", "[Hessian]") {
  sleipnir::scope_exit exit{
      [] { CHECK(sleipnir::global_pool_resource().blocks_in_use() == 0u); }};

  // y = x²
  // y = x * x
  sleipnir::VariableMatrix x{1};
  x(0).set_value(3);
  sleipnir::Variable y = x(0) * x(0);

  // dy/dx = x (rhs) + x (lhs)
  //       = (3) + (3)
  //       = 6
  double g = sleipnir::Gradient(y, x(0)).value().coeff(0);
  CHECK(g == 6.0);

  // d²y/dx² = d/dx(x (rhs) + x (lhs))
  //         = 1 + 1
  //         = 2
  auto H = sleipnir::Hessian(y, x);
  CHECK(H.get().value(0, 0) == 2.0);
  CHECK(H.value().coeff(0, 0) == 2.0);
}

TEST_CASE("Hessian - Sum", "[Hessian]") {
  sleipnir::scope_exit exit{
      [] { CHECK(sleipnir::global_pool_resource().blocks_in_use() == 0u); }};

  sleipnir::VariableMatrix x{5};
  x(0).set_value(1);
  x(1).set_value(2);
  x(2).set_value(3);
  x(3).set_value(4);
  x(4).set_value(5);

  // y = sum(x)
  auto y = std::accumulate(x.begin(), x.end(), sleipnir::Variable{0.0});
  CHECK(y.value() == 15.0);

  auto g = sleipnir::Gradient(y, x);
  CHECK(g.get().value() == Eigen::MatrixXd::Constant(5, 1, 1.0));
  CHECK(g.value().toDense() == Eigen::MatrixXd::Constant(5, 1, 1.0));

  auto H = sleipnir::Hessian(y, x);
  CHECK(H.get().value() == Eigen::MatrixXd::Zero(5, 5));
  CHECK(H.value().toDense() == Eigen::MatrixXd::Zero(5, 5));
}

TEST_CASE("Hessian - Sum of products", "[Hessian]") {
  sleipnir::scope_exit exit{
      [] { CHECK(sleipnir::global_pool_resource().blocks_in_use() == 0u); }};

  sleipnir::VariableMatrix x{5};
  x(0).set_value(1);
  x(1).set_value(2);
  x(2).set_value(3);
  x(3).set_value(4);
  x(4).set_value(5);

  // y = ||x||²
  sleipnir::Variable y = x.T() * x;
  CHECK(y.value() == 1 + 2 * 2 + 3 * 3 + 4 * 4 + 5 * 5);

  auto g = sleipnir::Gradient(y, x);
  CHECK(g.get().value() == 2 * x.value());
  CHECK(g.value().toDense() == 2 * x.value());

  auto H = sleipnir::Hessian(y, x);

  Eigen::MatrixXd expected_H =
      Eigen::VectorXd::Constant(5, 1, 2.0).asDiagonal();
  CHECK(H.get().value() == expected_H);
  CHECK(H.value().toDense() == expected_H);
}

TEST_CASE("Hessian - Product of sines", "[Hessian]") {
  sleipnir::scope_exit exit{
      [] { CHECK(sleipnir::global_pool_resource().blocks_in_use() == 0u); }};

  sleipnir::VariableMatrix x{5};
  x(0).set_value(1);
  x(1).set_value(2);
  x(2).set_value(3);
  x(3).set_value(4);
  x(4).set_value(5);

  // y = prod(sin(x))
  auto temp = x.cwise_transform(sleipnir::sin);
  auto y = std::accumulate(temp.begin(), temp.end(), sleipnir::Variable{1.0},
                           std::multiplies{});
  CHECK(y.value() == Catch::Approx(std::sin(1) * std::sin(2) * std::sin(3) *
                                   std::sin(4) * std::sin(5))
                         .margin(1e-15));

  auto g = sleipnir::Gradient(y, x);
  for (int i = 0; i < x.rows(); ++i) {
    CHECK(g.get().value(i) ==
          Catch::Approx((y / sleipnir::tan(x(i))).value()).margin(1e-15));
    CHECK(g.value().coeff(i) ==
          Catch::Approx((y / sleipnir::tan(x(i))).value()).margin(1e-15));
  }

  auto H = sleipnir::Hessian(y, x);

  Eigen::MatrixXd expected_H{5, 5};
  for (int i = 0; i < x.rows(); ++i) {
    for (int j = 0; j < x.rows(); ++j) {
      if (i == j) {
        expected_H(i, j) =
            (g.value().coeff(i) / sleipnir::tan(x(i))).value() *
            (1.0 - 1.0 / (sleipnir::cos(x(i)) * sleipnir::cos(x(i)))).value();
      } else {
        expected_H(i, j) = (g.value().coeff(j) / sleipnir::tan(x(i))).value();
      }
    }
  }

  auto actual_H = H.get().value();
  for (int i = 0; i < x.rows(); ++i) {
    for (int j = 0; j < x.rows(); ++j) {
      CHECK(actual_H(i, j) == Catch::Approx(expected_H(i, j)).margin(1e-15));
    }
  }

  actual_H = H.value();
  for (int i = 0; i < x.rows(); ++i) {
    for (int j = 0; j < x.rows(); ++j) {
      CHECK(actual_H(i, j) == Catch::Approx(expected_H(i, j)).margin(1e-15));
    }
  }
}

TEST_CASE("Hessian - Sum of squared residuals", "[Hessian]") {
  sleipnir::scope_exit exit{
      [] { CHECK(sleipnir::global_pool_resource().blocks_in_use() == 0u); }};

  sleipnir::Variable y;
  Eigen::VectorXd g;
  Eigen::MatrixXd H;
  sleipnir::VariableMatrix x{5};
  x(0).set_value(1);
  x(1).set_value(1);
  x(2).set_value(1);
  x(3).set_value(1);
  x(4).set_value(1);

  // y = sum(diff(x).^2)
  auto temp = (x.block(0, 0, 4, 1) - x.block(1, 0, 4, 1))
                  .cwise_transform([](const sleipnir::Variable& x) {
                    return sleipnir::pow(x, 2);
                  });
  y = std::accumulate(temp.begin(), temp.end(), sleipnir::Variable{0.0});
  g = sleipnir::Gradient(y, x).value();

  CHECK(y.value() == 0.0);
  CHECK(g(0) == (2 * x(0) - 2 * x(1)).value());
  CHECK(g(1) == (-2 * x(0) + 4 * x(1) - 2 * x(2)).value());
  CHECK(g(2) == (-2 * x(1) + 4 * x(2) - 2 * x(3)).value());
  CHECK(g(3) == (-2 * x(2) + 4 * x(3) - 2 * x(4)).value());
  CHECK(g(4) == (-2 * x(3) + 2 * x(4)).value());

  H = sleipnir::Hessian(y, x).value();

  Eigen::MatrixXd expected_H{{2.0, -2.0, 0.0, 0.0, 0.0},
                             {-2.0, 4.0, -2.0, 0.0, 0.0},
                             {0.0, -2.0, 4.0, -2.0, 0.0},
                             {0.0, 0.0, -2.0, 4.0, -2.0},
                             {0.0, 0.0, 0.0, -2.0, 2.0}};
  for (int i = 0; i < x.rows(); ++i) {
    for (int j = 0; j < x.rows(); ++j) {
      CHECK(H(i, j) == expected_H(i, j));
    }
  }
}

TEST_CASE("Hessian - Sum of squares", "[Hessian]") {
  sleipnir::scope_exit exit{
      [] { CHECK(sleipnir::global_pool_resource().blocks_in_use() == 0u); }};

  sleipnir::VariableMatrix r{4};
  r(0).set_value(25.0);
  r(1).set_value(10.0);
  r(2).set_value(5.0);
  r(3).set_value(0.0);
  sleipnir::VariableMatrix x{4};
  x(0).set_value(0.0);
  x(1).set_value(0.0);
  x(2).set_value(0.0);
  x(3).set_value(0.0);

  sleipnir::Variable J = 0.0;
  for (int i = 0; i < 4; ++i) {
    J += (r(i) - x(i)) * (r(i) - x(i));
  }

  auto H = sleipnir::Hessian(J, x);

  Eigen::MatrixXd expected_H =
      Eigen::VectorXd::Constant(4, 1, 2.0).asDiagonal();
  CHECK(H.get().value() == expected_H);
  CHECK(H.value().toDense() == expected_H);
}

TEST_CASE("Hessian - Rosenbrock", "[Hessian]") {
  sleipnir::scope_exit exit{
      [] { CHECK(sleipnir::global_pool_resource().blocks_in_use() == 0u); }};

  sleipnir::VariableMatrix input{2};
  auto& x = input(0);
  auto& y = input(1);

  for (auto x0 : range(-2.5, 2.5, 0.1)) {
    for (auto y0 : range(-2.5, 2.5, 0.1)) {
      x.set_value(x0);
      y.set_value(y0);
      auto z = sleipnir::pow(1 - x, 2) +
               100 * sleipnir::pow(y - sleipnir::pow(x, 2), 2);

      Eigen::MatrixXd H = sleipnir::Hessian(z, input).value();
      CHECK(H(0, 0) == Catch::Approx(-400 * (y0 - x0 * x0) + 800 * x0 * x0 + 2)
                           .margin(1e-12));
      CHECK(H(0, 1) == -400 * x0);
      CHECK(H(1, 0) == -400 * x0);
      CHECK(H(1, 1) == 200);
    }
  }
}

TEST_CASE("Hessian - Variable reuse", "[Hessian]") {
  sleipnir::scope_exit exit{
      [] { CHECK(sleipnir::global_pool_resource().blocks_in_use() == 0u); }};

  sleipnir::Variable y;
  sleipnir::VariableMatrix x{1};

  // y = x³
  x(0).set_value(1);
  y = x(0) * x(0) * x(0);

  sleipnir::Hessian hessian{y, x};

  // d²y/dx² = 6x
  // H = 6
  Eigen::MatrixXd H = hessian.value();

  CHECK(H.rows() == 1);
  CHECK(H.cols() == 1);
  CHECK(H(0, 0) == 6.0);

  x(0).set_value(2);
  // d²y/dx² = 6x
  // H = 12
  H = hessian.value();

  CHECK(H.rows() == 1);
  CHECK(H.cols() == 1);
  CHECK(H(0, 0) == 12.0);
}
